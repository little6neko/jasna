from __future__ import annotations

import logging
import subprocess
import threading
from types import SimpleNamespace
from unittest.mock import MagicMock

import av
import pytest
import torch

from jasna.main import _resolve_cli_encoder_settings, build_parser
from jasna.media.encoder_settings import parse_encoder_settings, software_codec
from jasna.media.probe import get_video_meta_data
from jasna.media.software_video import CpuVideoEncoder
from jasna.os_utils import resolve_executable


@pytest.mark.parametrize("text", ['-preset medium -crf 22', 'preset=medium,crf=22', '{"preset":"medium","crf":22}'])
def test_software_options_ignore_cli_cq(text):
    assert _resolve_cli_encoder_settings(text, cq=999, codec="libx265", vendor="nvidia") == {"preset": "medium", "crf": 22}


def test_ffmpeg_option_quoting_and_negative_values():
    assert parse_encoder_settings('-x265-params "pools=4:frame-threads=2" -threads -1') == {"x265-params": "pools=4:frame-threads=2", "threads": -1}
    assert _resolve_cli_encoder_settings("-b:v 1M -pix_fmt:v:0 yuv420p", cq=None,
                                         codec="mpeg4", vendor="nvidia") == {"b": "1M", "pix_fmt": "yuv420p"}


@pytest.mark.parametrize("text", ["-preset", "preset medium", "-crf 22 -crf 24"])
def test_invalid_options(text):
    with pytest.raises(ValueError):
        parse_encoder_settings(text)


@pytest.mark.parametrize("name", ["libx264", "libx265", "ffv1", "mpeg4", "prores_ks"])
def test_software_encoder_discovery(name):
    assert software_codec(name).type == "video"
    assert build_parser().parse_args(["--codec", name]).codec == name


@pytest.mark.parametrize("name", ["h264_nvenc", "aac", "not_a_real_encoder"])
def test_reject_hardware_audio_and_unknown_encoders(name):
    with pytest.raises(ValueError):
        software_codec(name)


@pytest.fixture
def source(tmp_path):
    path = tmp_path / "source.mkv"
    subtitle = tmp_path / "subtitle.srt"
    subtitle.write_text("1\n00:00:00,100 --> 00:00:01,000\nCPU output test\n")
    subprocess.run([
        resolve_executable("ffmpeg"), "-v", "error", "-y",
        "-f", "lavfi", "-i", "testsrc2=size=128x96:rate=12:duration=2",
        "-f", "lavfi", "-i", "sine=frequency=440:duration=2", "-i", str(subtitle),
        "-map", "0:v", "-map", "1:a", "-map", "2:s", "-c:v", "libx264",
        "-pix_fmt", "yuv420p10le", "-colorspace", "bt709", "-color_range", "tv",
        "-color_primaries", "bt709", "-color_trc", "bt709",
        "-c:a", "aac", "-c:s", "srt", "-metadata", "title=CPU output test", str(path),
    ], check=True)
    return path


@pytest.mark.parametrize("codec,suffix,pix_fmt", [
    ("libx265", ".mp4", "yuv420p10le"), ("libx264", ".mkv", "yuv420p10le"),
    ("ffv1", ".mkv", "yuv420p10le"), ("prores_ks", ".mov", "yuv422p10le"),
    ("mpeg4", ".mkv", "yuv420p"),
])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_cpu_roundtrip_preserves_frames_audio_subtitles_and_color(source, tmp_path, monkeypatch, codec, suffix, pix_fmt, device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA required")
    monkeypatch.setenv("JASNA_DECODE_BACKEND", "pyav-hw")
    metadata = get_video_meta_data(str(source))
    output = tmp_path / (codec + suffix)
    with av.open(str(source)) as reader:
        with CpuVideoEncoder(str(output), torch.device("cpu"), metadata, codec=codec, encoder_settings={}) as writer:
            for frame in reader.decode(video=0):
                rgb = torch.from_numpy(frame.to_ndarray(format="rgb24")).permute(2, 0, 1)
                writer.encode(rgb.to(device), frame.pts)
    with av.open(str(output)) as container:
        stream = container.streams.video[0]
        assert stream.codec_context.pix_fmt == pix_fmt
        assert stream.codec_context.colorspace == 1
        assert stream.codec_context.color_range == 1
        assert len(list(container.decode(video=0))) == 24
        assert container.metadata["title"] == "CPU output test"
    with av.open(str(output)) as container:
        audio = list(container.decode(audio=0))
        assert sum(f.samples / f.sample_rate for f in audio) >= 1.9
        assert len(container.streams.subtitles) == 1
    with av.open(str(output)) as container:
        assert sum(p.size > 0 for p in container.demux(container.streams.subtitles[0])) >= 1


def test_explicit_pixel_format_and_encoder_defaults(source, tmp_path, caplog):
    caplog.set_level(logging.ERROR)
    metadata = get_video_meta_data(str(source))
    writer = CpuVideoEncoder(str(tmp_path / "out.mkv"), torch.device("cpu"), metadata,
                             codec="libx265", encoder_settings={"pix_fmt": "yuv420p", "crf": 25})
    assert writer.pixel_format == "yuv420p"
    assert writer.encoder_options == {"preset": "medium", "crf": "25", "x265-params": "log-level=error"}
    with pytest.raises(ValueError, match="does not support"):
        CpuVideoEncoder(str(tmp_path / "out.mkv"), torch.device("cpu"), metadata,
                        codec="prores_ks", encoder_settings={"pix_fmt": "yuv420p"})


def test_software_encoding_keeps_gpu_vr_projector():
    from factories import make_pipeline
    from pathlib import Path
    pipeline = make_pipeline(codec="libx265", input_video=Path("FSVSS-0001.mp4"),
                             vr_mode="sbs-fisheye", encoder_settings={})
    metadata = SimpleNamespace(video_width=200, video_height=100, sample_aspect_ratio=1,
                               stereo_layout="", spherical_projection="")
    pipeline.configure_vr(metadata)
    assert pipeline.vr_projector.device == pipeline.device
    assert not hasattr(pipeline, "output_vr_projector")


def test_unknown_encoder_option_fails_before_writing_frames(source, tmp_path):
    metadata = get_video_meta_data(str(source))
    with pytest.raises(ValueError, match="did not accept"):
        with CpuVideoEncoder(str(tmp_path / "bad.mkv"), torch.device("cpu"), metadata,
                             codec="libx265", encoder_settings={"not_an_option": 1}):
            pytest.fail("invalid option accepted")


def test_rgb_encoder_uses_full_range_rgb_tags(source, tmp_path):
    metadata = get_video_meta_data(str(source))
    output = tmp_path / "rgb.mkv"
    with av.open(str(source)) as reader:
        with CpuVideoEncoder(str(output), torch.device("cpu"), metadata,
                             codec="libx264rgb", encoder_settings={"crf": 0}) as writer:
            frame = next(reader.decode(video=0))
            expected = frame.to_ndarray(format="rgb24")
            writer.encode(torch.from_numpy(expected).permute(2, 0, 1), frame.pts)
    with av.open(str(output)) as container:
        ctx = container.streams.video[0].codec_context
        assert ctx.color_range == 2
        assert ctx.colorspace == 0
        decoded = next(container.decode(video=0)).to_ndarray(format="rgb24")
        assert (decoded == expected).all()


def test_failed_consumer_cancels_and_drains_blocked_producer(monkeypatch):
    import jasna.pipeline_threads as threads
    pipeline = SimpleNamespace(
        device=torch.device("cpu"), restoration_pipeline=SimpleNamespace(secondary_num_workers=1),
        max_clip_size=1, cpu_output=False, vr_projector=None, input_video="unused",
        batch_size=1, temporal_overlap=0, max_detection_gap=0, min_detection_duration=1,
        enable_crossfade=False, scene_detection=False, job_detection_model=None,
        vr_resolution=SimpleNamespace(resolved="off"),
    )
    full = threading.Event()
    error = RuntimeError("injected consumer failure")
    def producer(**kw):
        kw["encode_queue"].put("first", frame_count=1)
        full.set()
        kw["encode_queue"].put("second", frame_count=1)
    def consumer(**kw):
        assert full.wait(1)
        kw["error_holder"].append(error)
    monkeypatch.setattr(threads, "decode_detect_loop", lambda **kw: None)
    monkeypatch.setattr(threads, "primary_restore_loop", lambda **kw: None)
    monkeypatch.setattr(threads, "secondary_restore_loop", producer)
    monkeypatch.setattr(threads, "blend_encode_loop", consumer)
    monkeypatch.setattr(threads, "VramOffloader", MagicMock())
    monkeypatch.setattr(threads, "empty_cache", lambda *_: None)
    monkeypatch.setattr(threads, "ipc_collect", lambda *_: None)
    cancel = threading.Event()
    result = []
    task = threading.Thread(target=lambda: result.append(threads.run_restoration_pass(
        pipeline, None, None, cancel, seek_ts=None, use_async_secondary=False)), daemon=True)
    task.start()
    task.join(2)
    try:
        assert not task.is_alive(), "failed consumer left producer blocked"
        assert result == [error]
    finally:
        cancel.set()
        task.join(2)


@pytest.mark.parametrize("pix_fmt", ["yuv420p", "yuv420p10le"])
def test_gpu_conversion_matches_reference_with_plane_padding(source, tmp_path, pix_fmt):
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    import numpy as np
    from jasna.media.rgb_to_yuv import RgbToYuvConverter
    metadata = get_video_meta_data(str(source))
    writer = CpuVideoEncoder(str(tmp_path / "out.mkv"), torch.device("cuda"), metadata,
                             codec="ffv1", encoder_settings={"pix_fmt": pix_fmt})
    # An unaligned width exercises PyAV plane padding; saturated values test
    # P010 sign handling as well as luma/chroma plane order.
    frame = torch.zeros((3, 98, 130), dtype=torch.uint8)
    frame[:, :, :32] = 255
    frame[0, :, 32:64] = 255
    frame[1, :, 64:96] = 255
    frame[2, :, 96:] = 255
    ten = pix_fmt == "yuv420p10le"
    reference = RgbToYuvConverter(("p010_" if ten else "nv12_") + "bt709_limited", device=torch.device("cpu"))
    packed = torch.empty((147, 130), dtype=reference.sample_dtype)
    reference.convert_into(frame, packed[:98], packed[98:])
    if ten:
        packed.bitwise_right_shift_(6).bitwise_and_(1023)
    expected = [packed[:98], packed[98:, ::2], packed[98:, 1::2]]
    output = writer._prepare_frame(frame.cuda())
    for plane, target in zip(output.planes, expected):
        dtype = np.uint16 if ten else np.uint8
        values = np.frombuffer(plane, dtype=dtype).reshape(plane.height, -1)[:, :plane.width]
        assert np.abs(values.astype(np.int32) - target.numpy().astype(np.int32)).max() <= 1


@pytest.mark.parametrize("level", ["error", "warning", "info", "debug"])
def test_x265_native_logs_follow_log_level(source, tmp_path, caplog, capfd, level):
    caplog.set_level(getattr(logging, level.upper()))
    metadata = get_video_meta_data(str(source))
    with av.open(str(source)) as reader:
        with CpuVideoEncoder(str(tmp_path / "log-test.mp4"), "cpu", metadata,
                             codec="libx265", encoder_settings={
                                 "preset": "ultrafast", "x265-params": "pools=1:frame-threads=1",
                             }) as writer:
            frame = next(reader.decode(video=0))
            rgb = torch.from_numpy(frame.to_ndarray(format="rgb24")).permute(2, 0, 1)
            writer.encode(rgb, frame.pts)
    stderr = capfd.readouterr().err
    assert ("x265 [info]" in stderr) == (level in {"info", "debug"})
    assert "x265 [error]" not in stderr


@pytest.mark.parametrize("key", ["log-level", "log_level"])
def test_x265_explicit_log_level_preserved(source, tmp_path, caplog, key):
    caplog.set_level(logging.ERROR)
    params = f"pools=2:{key}=debug:frame-threads=1"
    writer = CpuVideoEncoder(str(tmp_path / "explicit.mp4"), "cpu", get_video_meta_data(str(source)),
                             codec="libx265", encoder_settings={"x265-params": params})
    assert writer.encoder_options["x265-params"] == params
