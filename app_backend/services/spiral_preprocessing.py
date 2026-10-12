from __future__ import annotations

import logging
import tempfile
import io
from pathlib import Path

from PIL import Image, UnidentifiedImageError

from app_backend.errors import APIError

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}
VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".webm"}
ALLOWED_IMAGE_MIMES = {"image/png", "image/jpeg", "application/octet-stream"}
ALLOWED_VIDEO_MIMES = {
    "video/mp4", "video/quicktime", "video/x-msvideo", "video/webm", "application/octet-stream"
}


def validate_spiral_image(data: bytes, filename: str, content_type: str | None, config) -> dict:
    suffix = Path(filename).suffix.lower()
    if suffix not in IMAGE_EXTENSIONS:
        raise APIError(415, "unsupported_spiral_type", "Supported spiral images are PNG and JPEG.")
    if content_type and content_type.lower() not in ALLOWED_IMAGE_MIMES:
        raise APIError(415, "unsupported_spiral_type", "The uploaded image media type is not supported.")
    if not data:
        raise APIError(400, "empty_upload", "The spiral image upload is empty.")
    if len(data) > config["SPIRAL_MAX_BYTES"]:
        raise APIError(413, "spiral_upload_too_large", "The spiral upload exceeds the configured size limit.")
    try:
        with Image.open(io.BytesIO(data)) as image:
            image.verify()
        with Image.open(io.BytesIO(data)) as image:
            width, height = image.size
            fmt = image.format
            if width < 16 or height < 16 or width > config["SPIRAL_MAX_DIMENSION"] or height > config["SPIRAL_MAX_DIMENSION"]:
                raise APIError(422, "invalid_spiral_dimensions", "Image dimensions are outside the supported range.")
            image.load()
    except (UnidentifiedImageError, Image.DecompressionBombError, OSError, SyntaxError):
        raise APIError(400, "invalid_spiral_image", "The upload is not a decodable PNG or JPEG image.") from None
    if (suffix == ".png" and fmt != "PNG") or (suffix in {".jpg", ".jpeg"} and fmt != "JPEG"):
        raise APIError(415, "spiral_type_mismatch", "The file extension does not match its image content.")
    return {"media_type": fmt.lower(), "width": width, "height": height, "byte_count": len(data)}


def validate_spiral_video(data: bytes, filename: str, content_type: str | None, config) -> dict:
    suffix = Path(filename).suffix.lower()
    if suffix not in VIDEO_EXTENSIONS:
        raise APIError(415, "unsupported_spiral_type", "Supported spiral videos are MP4, MOV, AVI, and WebM.")
    if content_type and content_type.lower() not in ALLOWED_VIDEO_MIMES:
        raise APIError(415, "unsupported_spiral_type", "The uploaded video media type is not supported.")
    if not data:
        raise APIError(400, "empty_upload", "The spiral video upload is empty.")
    if len(data) > config["SPIRAL_MAX_BYTES"]:
        raise APIError(413, "spiral_upload_too_large", "The spiral upload exceeds the configured size limit.")

    temp_dir = Path(config["UPLOAD_TEMP_DIR"])
    temp_dir.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(prefix="pd-spiral-", suffix=suffix, dir=temp_dir, delete=False)
    temp_path = Path(handle.name)
    try:
        with handle:
            handle.write(data)
        try:
            import cv2
        except ImportError:
            raise APIError(503, "video_decoder_unavailable", "Video validation is unavailable on this server.") from None
        capture = cv2.VideoCapture(str(temp_path))
        try:
            if not capture.isOpened():
                raise APIError(400, "invalid_spiral_video", "The uploaded video could not be decoded.")
            fps = float(capture.get(cv2.CAP_PROP_FPS))
            declared_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            if fps <= 0 or declared_frames <= 0:
                raise APIError(400, "invalid_spiral_video", "The video has no usable frame-rate or frame-count metadata.")
            duration = declared_frames / fps
            if duration <= 0 or duration > config["SPIRAL_MAX_SECONDS"]:
                raise APIError(422, "invalid_spiral_duration", "Spiral videos must be 30 seconds or shorter.")
            if declared_frames > config["SPIRAL_MAX_FRAMES"]:
                raise APIError(422, "too_many_video_frames", "The video exceeds the configured frame-count limit.")
            decoded = 0
            while decoded <= config["SPIRAL_MAX_FRAMES"]:
                ok, frame = capture.read()
                if not ok:
                    break
                if frame is None or frame.size == 0:
                    raise APIError(400, "invalid_spiral_video", "The video contains an undecodable frame.")
                decoded += 1
            if decoded == 0 or decoded != declared_frames:
                raise APIError(400, "invalid_spiral_video", "The uploaded video is truncated or has inconsistent frame metadata.")
            return {
                "media_type": suffix.lstrip("."),
                "frame_count": decoded,
                "fps": round(fps, 4),
                "duration_seconds": round(duration, 4),
                "byte_count": len(data),
            }
        except APIError:
            raise
        except Exception:
            raise APIError(400, "invalid_spiral_video", "The uploaded video could not be decoded.") from None
        finally:
            capture.release()
    finally:
        try:
            temp_path.unlink(missing_ok=True)
        except OSError:
            logging.getLogger("app_backend.uploads").error(
                "temporary_upload_cleanup_failed",
                extra={"event": "upload_cleanup_error", "component": "spiral_video"},
            )
            raise APIError(500, "upload_cleanup_failed", "The temporary upload could not be securely removed.") from None


def validate_spiral_upload(data: bytes, filename: str, content_type: str | None, config) -> dict:
    suffix = Path(filename).suffix.lower()
    if suffix in IMAGE_EXTENSIONS:
        return {"kind": "image", **validate_spiral_image(data, filename, content_type, config)}
    if suffix in VIDEO_EXTENSIONS:
        return {"kind": "video", **validate_spiral_video(data, filename, content_type, config)}
    raise APIError(415, "unsupported_spiral_type", "Upload a PNG/JPEG image or MP4/MOV/AVI/WebM video.")


def camera_features_supported(_model_feature_names: list[str]) -> bool:
    """Camera data cannot currently supply tablet pressure/grip/pen-up signals."""
    return False
