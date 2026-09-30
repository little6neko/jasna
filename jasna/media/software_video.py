"""Software encoding after GPU decoding, restoration, blending and conversion.

The GPU remains responsible for detection/restoration. Source-stream muxing
is shared with VideoEncoder so audio, subtitles and metadata survive export.
"""
from __future__ import annotations

import logging
from fractions import Fraction
from pathlib import Path

import av
import numpy as np
import torch
from av.video.reformatter import ColorRange

from jasna.media.container_utils import MOV_SUFFIXES
from jasna.media.encoder_settings import is_software_codec, software_codec, validate_encoder_settings
from jasna.media.rgb_to_yuv import RgbToYuvConverter
from jasna.media.video_encoder import (
    VideoEncoder, _COLOR_TAGS, _COLOR_PRIMARIES, _COLOR_TRANSFERS,
    _mov_container_options, _option_value, _COLOR_VARIANTS,
)

log = logging.getLogger(__name__)


class CpuVideoEncoder(VideoEncoder):
    """Synchronous CPU encoder with bounded input buffering (no RGB queue)."""

    def __init__(self, file, device, metadata, *, codec, encoder_settings,
                 lut_path=None, sharpen_strength=0.0, output_fps=None,
                 pts_origin=0, smart_fragment=False, fmp4=False):
        if not is_software_codec(codec):
            raise ValueError(f"Unsupported software codec: {codec}")
        if lut_path or sharpen_strength or smart_fragment:
            raise ValueError("Software output does not support LUT, sharpening or smart fragments")
        self.device = torch.device(device)
        self.metadata = metadata
        self.output_path = Path(file)
        self.codec = codec
        self.encoder_name = codec
        self.output_fps = Fraction(metadata.video_fps_exact if output_fps is None else output_fps)
        self.pts_origin = int(pts_origin)
        self.smart_fragment = False
        self.fmp4 = bool(fmp4)
        self._source_chapters = ()
        settings = validate_encoder_settings(dict(encoder_settings), codec=codec, vendor="nvidia")
        descriptor = software_codec(codec)
        self.canonical_codec = descriptor.canonical_name
        supported = [fmt.name for fmt in (descriptor.video_formats or [])]
        preferred = ["yuv420p10le", "yuv422p10le", "yuv444p10le"] if metadata.is_10bit else []
        preferred += ["yuv420p", "yuv422p", "yuv444p", "yuv420p10le", "yuv422p10le", "rgb24", "bgr0"]
        default_format = next(
            (fmt for fmt in preferred if not supported or fmt in supported),
            supported[0] if supported else "yuv420p",
        )
        self.pixel_format = str(settings.pop("pix_fmt", default_format))
        if supported and self.pixel_format not in supported:
            raise ValueError(f"{codec} does not support pix_fmt={self.pixel_format}; supported: {', '.join(supported)}")
        self.rgb_output = av.VideoFormat(self.pixel_format).is_rgb
        self.encoder_options = {"preset": "medium", "crf": "22"} if codec in {"libx264", "libx265"} else {}
        self.encoder_options.update({k: _option_value(v) for k, v in settings.items()})
        if codec == "libx265":
            params = self.encoder_options.get("x265-params", "")
            has_log_level = any(
                option.partition("=")[0].strip().replace("_", "-") == "log-level"
                for option in params.split(":")
            )
            if not has_log_level:
                level = log.getEffectiveLevel()
                name = ("error" if level >= logging.ERROR else
                        "warning" if level >= logging.WARNING else
                        "info" if level >= logging.INFO else "debug")
                self.encoder_options["x265-params"] = ":".join(
                    part for part in (params, f"log-level={name}") if part
                )
        self._converter = None
        self._packed = None
        self._planar = None

    def __enter__(self):
        self._src = av.open(self.metadata.video_file)
        self.dst = None
        try:
            self.dst = av.open(
                str(self.output_path), "w",
                container_options=_mov_container_options(self.output_path.suffix, fmp4=self.fmp4),
            )
            inv = self._src.streams.video[0]
            out = self.dst.add_stream(self.encoder_name, rate=self.output_fps, options=dict(self.encoder_options))
            if self.canonical_codec == "hevc" and self.output_path.suffix.lower() in MOV_SUFFIXES:
                out.codec_tag = "hvc1"
            out.width, out.height = self.metadata.video_width, self.metadata.video_height
            out.time_base = self.metadata.time_base
            ctx = out.codec_context
            ctx.time_base = self.metadata.time_base
            ctx.framerate = self.output_fps
            ctx.pix_fmt = self.pixel_format
            if self.metadata.sample_aspect_ratio != 1:
                ctx.sample_aspect_ratio = self.metadata.sample_aspect_ratio
            matrix, primaries, transfer = _COLOR_TAGS[self.metadata.color_space]
            ctx.colorspace = 0 if self.rgb_output else matrix
            ctx.color_range = int(ColorRange.JPEG if self.rgb_output else self.metadata.color_range)
            ctx.color_primaries = _COLOR_PRIMARIES.get(self.metadata.color_primaries.lower(), primaries)
            ctx.color_trc = _COLOR_TRANSFERS.get(self.metadata.color_transfer.lower(), transfer)
            self.out_stream = out
            self._video_started = False
            self._options_validated = False
            self._last_emitted_pts = None
            self._copy_source_metadata(inv, out)
            self._setup_source_streams(inv)
            ctx.open()
            self._validate_encoder_options()
            self._options_validated = True
            log.info("CPU encoder: %s, pixel format=%s, options=%s", self.encoder_name, self.pixel_format, self.encoder_options)
            return self
        except BaseException:
            try:
                if self.dst is not None:
                    self.dst.close()
            finally:
                self._src.close()
            raise

    def _prepare_frame(self, frame: torch.Tensor) -> av.VideoFrame:
        height, width = frame.shape[-2:]
        if frame.device.type != "cpu" and self.pixel_format in {"yuv420p", "yuv420p10le"}:
            if height % 2 or width % 2:
                raise ValueError("GPU YUV420 conversion requires even frame dimensions")
            ten_bit = self.pixel_format == "yuv420p10le"
            if self._converter is None:
                variant = _COLOR_VARIANTS[(self.metadata.color_space, self.metadata.color_range)]
                self._converter = RgbToYuvConverter(
                    ("p010_" if ten_bit else "nv12_") + variant, device=frame.device,
                )
                self._packed = torch.empty(
                    (height + height // 2, width), dtype=self._converter.sample_dtype, device=frame.device,
                )
                self._planar = torch.empty_like(self._packed).flatten()
            packed, planar = self._packed, self._planar
            self._converter.convert_into(frame.contiguous(), packed[:height], packed[height:])
            if ten_bit:
                # P010 uses the high 10 bits; planar 10-bit uses the low bits.
                # Mask after the signed shift to avoid sign extension.
                packed.bitwise_right_shift_(6).bitwise_and_(1023)
            y_size = height * width
            c_size = y_size // 4
            planar[:y_size].view(height, width).copy_(packed[:height])
            planar[y_size:y_size + c_size].view(height // 2, width // 2).copy_(packed[height:, ::2])
            planar[y_size + c_size:].view(height // 2, width // 2).copy_(packed[height:, 1::2])
            # One blocking device-to-host copy, after GPU color conversion.
            host = planar.cpu().numpy()
            if ten_bit:
                host = host.view(np.uint16)
            output = av.VideoFrame(width, height, self.pixel_format)
            offset = 0
            for plane in output.planes:
                count = plane.width * plane.height
                values = host[offset:offset + count].reshape(plane.height, plane.width)
                if plane.line_size == plane.width * host.dtype.itemsize:
                    plane.update(values.tobytes())
                else:
                    padded = np.zeros((plane.height, plane.line_size // host.dtype.itemsize), dtype=host.dtype)
                    padded[:, :plane.width] = values
                    plane.update(padded.tobytes())
                offset += count
            return output

        # Other FFmpeg pixel formats remain supported. Download RGB once at
        # the encoder boundary, then let libswscale adapt it to that codec.
        host_rgb = frame.permute(1, 2, 0).contiguous().cpu().numpy()
        rgb = av.VideoFrame.from_ndarray(host_rgb, format="rgb24")
        return rgb.reformat(
            format=self.pixel_format, dst_colorspace=None if self.rgb_output else self.metadata.color_space,
            src_color_range=ColorRange.JPEG,
            dst_color_range=ColorRange.JPEG if self.rgb_output else self.metadata.color_range,
        )

    def encode(self, frame, pts, *, apply_lut=True):
        yuv = self._prepare_frame(frame)
        yuv.pts = self._clamp_pts_monotonic(int(pts) - self.pts_origin)
        yuv.time_base = self.metadata.time_base
        for packet in self.out_stream.encode(yuv):
            self._mux_video(packet)

    def __exit__(self, exc_type, exc_value, traceback):
        try:
            if exc_type is None:
                for packet in self.out_stream.encode(None):
                    self._mux_video(packet)
                self._drain_source_streams()
        finally:
            self._converter = self._packed = self._planar = None
            try:
                self.dst.close()
            finally:
                self._src.close()
