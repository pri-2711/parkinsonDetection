from __future__ import annotations

import io
import wave

from app_backend.errors import APIError

ALLOWED_EXTENSIONS = {".wav"}
ALLOWED_MIME_TYPES = {"audio/wav", "audio/x-wav", "audio/wave", "application/octet-stream"}


def validate_wav(data: bytes, filename: str, content_type: str | None, config) -> dict:
    suffix = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    if f".{suffix}" not in ALLOWED_EXTENSIONS:
        raise APIError(415, "unsupported_audio_type", "Only PCM WAV audio uploads are accepted.")
    if content_type and content_type.lower() not in ALLOWED_MIME_TYPES:
        raise APIError(415, "unsupported_audio_type", "The uploaded audio media type is not supported.")
    if not data:
        raise APIError(400, "empty_upload", "The audio upload is empty.")
    if len(data) > config["AUDIO_MAX_BYTES"]:
        raise APIError(413, "audio_too_large", "The audio upload exceeds the configured size limit.")
    try:
        with wave.open(io.BytesIO(data), "rb") as audio:
            channels = audio.getnchannels()
            sample_width = audio.getsampwidth()
            sample_rate = audio.getframerate()
            frame_count = audio.getnframes()
            duration = frame_count / sample_rate if sample_rate else 0
            compression = audio.getcomptype()
    except (wave.Error, EOFError, OSError):
        raise APIError(400, "invalid_audio", "The uploaded file is not a decodable PCM WAV recording.") from None
    if compression != "NONE" or channels not in (1, 2) or sample_width not in (1, 2, 3, 4):
        raise APIError(415, "unsupported_audio_encoding", "Use uncompressed mono or stereo PCM WAV audio.")
    if not 8000 <= sample_rate <= 48000:
        raise APIError(422, "unsupported_audio_sample_rate", "The WAV sample rate must be between 8 kHz and 48 kHz.")
    if not config["AUDIO_MIN_SECONDS"] <= duration <= config["AUDIO_MAX_SECONDS"]:
        raise APIError(
            422,
            "invalid_audio_duration",
            f"Audio duration must be between {config['AUDIO_MIN_SECONDS']:g} and {config['AUDIO_MAX_SECONDS']:g} seconds.",
        )
    return {
        "format": "wav",
        "channels": channels,
        "sample_rate_hz": sample_rate,
        "sample_width_bytes": sample_width,
        "duration_seconds": round(duration, 4),
        "byte_count": len(data),
    }
