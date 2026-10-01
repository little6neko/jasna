from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from jasna.mosaic.detections import Detections

log = logging.getLogger(__name__)

VR_MODES = ("auto", "off", "sbs", "sbs-fisheye")
PROJECTION_KINDS = ("raw", "fisheye", "gnomonic")
PROJECTION_CHOICES = ("auto", *PROJECTION_KINDS)
_SBS_ASPECT_MIN = 1.90
_SBS_ASPECT_MAX = 2.10
_AUTO_SBS_MIN_HEIGHT = 1080

VR_PROJECTIONS_PATH = Path(__file__).resolve().parent.parent / "vr-projections.json"


@dataclass(frozen=True)
class VrModeResolution:
    requested: str
    resolved: str
    reason: str
    display_aspect: float
    projection: str

    @property
    def is_sbs(self) -> bool:
        return self.resolved == "sbs"


def _match_projection(input_path: Path) -> tuple[str, str] | None:
    with VR_PROJECTIONS_PATH.open(encoding="utf-8") as f:
        mappings = json.load(f)
    if not isinstance(mappings, dict):
        raise ValueError(f"{VR_PROJECTIONS_PATH}: expected a prefix-to-projection object")
    normalized = {}
    for prefix, kind in mappings.items():
        if not prefix.strip() or kind not in PROJECTION_KINDS:
            raise ValueError(f"{VR_PROJECTIONS_PATH}: invalid projection entry {prefix!r}: {kind!r}")
        key = prefix.strip().upper()
        if key in normalized:
            raise ValueError(f"{VR_PROJECTIONS_PATH}: duplicate prefix {prefix!r}")
        normalized[key] = kind
    stem = re.sub(r"^\[[^\]]*\]\s*", "", input_path.stem).upper()
    for prefix in sorted(normalized, key=len, reverse=True):
        if stem.startswith(prefix):
            return prefix, normalized[prefix]
    return None


def _normalize_projection(value: str) -> str:
    projection = str(value).strip().lower()
    if projection not in PROJECTION_CHOICES:
        raise ValueError(
            f"Unknown VR projection '{projection}'. "
            f"Valid projections: {', '.join(PROJECTION_CHOICES)}"
        )
    return projection


def resolve_projection(input_path: Path, requested: str = "auto") -> str:
    projection = _normalize_projection(requested)
    if projection != "auto":
        return projection
    match = _match_projection(input_path)
    return match[1] if match else "raw"


def _display_aspect(metadata) -> float:
    sar = metadata.sample_aspect_ratio
    return (
        float(metadata.video_width)
        * float(sar.numerator)
        / float(sar.denominator)
        / float(metadata.video_height)
    )


def _has_sbs_spatial_metadata(metadata) -> bool:
    stereo = str(getattr(metadata, "stereo_layout", "")).lower()
    projection = str(getattr(metadata, "spherical_projection", "")).lower()
    is_sbs = "side by side" in stereo or stereo in {
        "sbs",
        "left-right",
        "left_right",
    }
    is_equirectangular = "equirect" in projection
    return is_sbs and is_equirectangular


def resolve_vr_mode(
    requested: str,
    metadata,
    input_path: Path,
    *,
    projection: str = "auto",
    announce: bool = False,
) -> VrModeResolution:
    requested = str(requested).strip().lower()
    if requested not in VR_MODES:
        raise ValueError(
            f"Unknown VR mode '{requested}'. Valid modes: {', '.join(VR_MODES)}"
        )
    projection = _normalize_projection(projection)

    match = None
    prefix_checked = False
    width = int(metadata.video_width)
    height = int(metadata.video_height)
    aspect = _display_aspect(metadata)
    is_high_resolution_2_to_1 = (
        width == height * 2 and height > _AUTO_SBS_MIN_HEIGHT
    )
    if requested == "off":
        resolved, reason = "off", "explicit mode"
    elif requested != "auto":
        if width % 2:
            raise ValueError(
                f"VR SBS processing requires an even frame width, got {width}"
            )
        resolved, reason = "sbs", "explicit mode"
        if not (_SBS_ASPECT_MIN <= aspect <= _SBS_ASPECT_MAX):
            reason += f"; unusual SBS display aspect {aspect:.3f}"
    elif width % 2:
        resolved, reason = "off", f"odd frame width {width}"
    elif (
        not is_high_resolution_2_to_1
        and not (_SBS_ASPECT_MIN <= aspect <= _SBS_ASPECT_MAX)
    ):
        resolved, reason = "off", f"display aspect {aspect:.3f} is outside the SBS gate"
    else:
        match = _match_projection(input_path)
        prefix_checked = True
        if match:
            resolved, reason = "sbs", f"configured VR prefix {match[0]}"
        elif _has_sbs_spatial_metadata(metadata):
            resolved, reason = "sbs", "side-by-side equirectangular spatial metadata"
        elif is_high_resolution_2_to_1:
            resolved, reason = "sbs", f"2:1 frame above {_AUTO_SBS_MIN_HEIGHT}p"
        else:
            resolved, reason = "off", "no trusted studio token or spatial metadata"

    if resolved != "sbs":
        resolved_projection = "none"
        projection_source = "VR disabled"
    elif projection != "auto":
        resolved_projection = projection
        projection_source = "explicit --vr-projection"
    elif requested == "sbs-fisheye":
        resolved_projection = "fisheye"
        projection_source = "--vr-mode sbs-fisheye"
    else:
        if not prefix_checked:
            match = _match_projection(input_path)
            prefix_checked = True
        resolved_projection = match[1] if match else "raw"
        projection_source = "prefix mapping" if match else "auto fallback"

    result = VrModeResolution(
        requested,
        resolved,
        reason,
        aspect,
        resolved_projection,
    )
    if resolved != "sbs":
        message = f"[VR: off] {reason}"
    elif projection_source == "prefix mapping":
        message = f"[VR: SBS] prefix={match[0]} -> projection={result.projection}"
    elif projection_source == "auto fallback":
        message = f"[VR: SBS] no matching prefix -> projection={result.projection} (auto)"
    else:
        prefix_info = (
            f"prefix={match[0]} ({match[1]}) -> "
            if match else "no matching prefix -> " if prefix_checked else ""
        )
        message = (
            f"[VR: SBS] {prefix_info}projection={result.projection} "
            f"({projection_source})"
        )
    if "unusual SBS" in result.reason:
        message += f"; {result.reason}"
    if announce:
        print(message, flush=True)
    elif "unusual SBS" in result.reason:
        log.warning(message)
    else:
        log.info(message)
    return result


class SbsDetectionAdapter:
    def __init__(self, detector) -> None:
        self.detector = detector

    @staticmethod
    def _eye_width(frames: torch.Tensor) -> int:
        width = int(frames.shape[-1])
        if width % 2:
            raise ValueError(
                f"VR SBS processing requires an even frame width, got {width}"
            )
        return width // 2

    def __call__(
        self,
        frames: torch.Tensor,
        *,
        target_hw: tuple[int, int],
    ) -> Detections:
        eye_width = self._eye_width(frames)
        target_h, target_w = map(int, target_hw)
        if target_w % 2:
            raise ValueError(
                f"VR SBS processing requires an even target width, got {target_w}"
            )
        target_eye_width = target_w // 2
        left = self.detector(
            frames[:, :, :, :eye_width],
            target_hw=(target_h, target_eye_width),
        )
        right = self.detector(
            frames[:, :, :, eye_width:],
            target_hw=(target_h, target_eye_width),
        )

        boxes: list[np.ndarray] = []
        masks: list[torch.Tensor] = []
        offset = np.array(
            [target_eye_width, 0, target_eye_width, 0],
            dtype=np.float32,
        )
        for left_boxes, right_boxes, left_masks, right_masks in zip(
            left.boxes_xyxy,
            right.boxes_xyxy,
            left.masks,
            right.masks,
        ):
            if left_masks.shape[-2:] != right_masks.shape[-2:]:
                raise RuntimeError(
                    "Per-eye detector masks have mismatched shapes: "
                    f"{tuple(left_masks.shape)} vs {tuple(right_masks.shape)}"
                )
            mask_width = int(left_masks.shape[-1])
            boxes.append(
                np.concatenate(
                    (left_boxes, right_boxes + offset),
                    axis=0,
                ).astype(np.float32, copy=False)
            )
            masks.append(
                torch.cat(
                    (
                        F.pad(left_masks, (0, mask_width)),
                        F.pad(right_masks, (mask_width, 0)),
                    ),
                    dim=0,
                )
            )
        return Detections(boxes_xyxy=boxes, masks=masks)

    def scan_scores_masks(
        self,
        frames: torch.Tensor,
        *,
        mask_hw: tuple[int, int],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        eye_width = self._eye_width(frames)
        mask_h, mask_w = map(int, mask_hw)
        left_mask_w = mask_w // 2
        right_mask_w = mask_w - left_mask_w
        left_scores, left_masks = self.detector.scan_scores_masks(
            frames[:, :, :, :eye_width],
            mask_hw=(mask_h, left_mask_w),
        )
        right_scores, right_masks = self.detector.scan_scores_masks(
            frames[:, :, :, eye_width:],
            mask_hw=(mask_h, right_mask_w),
        )
        return torch.maximum(left_scores, right_scores), torch.cat(
            (left_masks, right_masks),
            dim=-1,
        )

    def close(self) -> None:
        if hasattr(self.detector, "close"):
            self.detector.close()
