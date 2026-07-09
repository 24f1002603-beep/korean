import base64
import os
import io
import re
import logging

import fastapi
from fastapi import Body
from pydantic import BaseModel
import pandas as pd
import numpy as np
import requests

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("verify_audio")

app = fastapi.FastAPI()
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

TRANSCRIPTION_URL = "https://openrouter.ai/api/v1/audio/transcriptions"
MODEL = "openai/whisper-large-v3"
REQUEST_TIMEOUT = 120  # seconds - long clips (e.g. 125 numbers) need headroom

# --- Korean number-word -> digit conversion -------------------------------
# Whisper sometimes writes numbers as Korean words instead of digits
# (e.g. "이십삼" instead of "23"). This is a basic native-Korean-number
# and Sino-Korean-number parser to catch those cases as a fallback.

_SINO_DIGITS = {"영": 0, "일": 1, "이": 2, "삼": 3, "사": 4,
                "오": 5, "육": 6, "칠": 7, "팔": 8, "구": 9}
_SINO_UNITS = {"십": 10, "백": 100, "천": 1000, "만": 10000}


def _sino_korean_to_int(token: str):
    """Best-effort parse of a Sino-Korean number word into an int, or None."""
    if not token:
        return None
    total = 0
    section = 0
    num = 0
    i = 0
    matched_any = False
    while i < len(token):
        ch = token[i]
        if ch in _SINO_DIGITS:
            num = _SINO_DIGITS[ch]
            matched_any = True
        elif ch in _SINO_UNITS:
            unit = _SINO_UNITS[ch]
            if num == 0:
                num = 1
            if unit == 10000:
                total += (section + num * unit)
                section = 0
                num = 0
            else:
                section += num * unit
                num = 0
            matched_any = True
        else:
            return None  # unknown character -> not a clean number word
        i += 1
    total += section + num
    return total if matched_any else None


def extract_numbers(text: str):
    """
    Extract numbers from transcribed text.
    1) Prefer plain digit numbers (most common Whisper output for read-out lists).
    2) Fall back to Sino-Korean number words if no digits were found at all.
    """
    digit_numbers = re.findall(r'[-+]?\d*\.\d+|\d+', text)
    if digit_numbers:
        return [float(x) for x in digit_numbers]

    # Fallback: try to parse Korean number-word tokens
    tokens = re.findall(r'[영일이삼사오육칠팔구십백천만]+', text)
    parsed = []
    for tok in tokens:
        val = _sino_korean_to_int(tok)
        if val is not None:
            parsed.append(float(val))
    return parsed


def clean_dict(d):
    return {k: (None if pd.isna(v) else v) for k, v in d.items()}


class AudioRequest(BaseModel):
    audio_id: str
    audio_base64: str


def transcribe_audio(audio_bytes: bytes) -> str:
    files = {
        'file': ('audio.wav', io.BytesIO(audio_bytes), 'audio/wav')
    }
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}"
    }
    data = {
        "model": MODEL,
        "response_format": "json",
        "language": "ko"
    }

    try:
        response = requests.post(
            TRANSCRIPTION_URL,
            headers=headers,
            files=files,
            data=data,
            timeout=REQUEST_TIMEOUT
        )
    except requests.exceptions.RequestException as e:
        logger.error(f"Transcription request failed: {e}")
        return ""

    logger.info(f"OpenRouter status: {response.status_code}")
    logger.info(f"OpenRouter body (first 1000 chars): {response.text[:1000]}")

    if response.status_code != 200:
        logger.error(f"Non-200 response from OpenRouter: {response.status_code}")
        return ""

    try:
        result = response.json()
    except ValueError:
        logger.error("Response was not valid JSON")
        return ""

    spoken_text = (result.get("text") or "").strip()
    logger.info(f"Transcribed text: {spoken_text}")
    return spoken_text


@app.post("/verify")
async def verify_audio(payload: AudioRequest = Body(...)):
    audio_bytes = base64.b64decode(payload.audio_base64)

    spoken_text = transcribe_audio(audio_bytes)

    # Extract the Korean word (column name)
    korean_match = re.search(r'[\uac00-\ud7a3]+', spoken_text)
    col_name = korean_match.group(0).strip() if korean_match else "값"

    # Extract numbers (digits first, Korean-word fallback second)
    numbers = extract_numbers(spoken_text)

    logger.info(f"Column name: {col_name}, numbers found: {len(numbers)}")

    if not numbers:
        df = pd.DataFrame(columns=[col_name])
    else:
        df = pd.DataFrame({col_name: numbers})

    numeric_df = df.select_dtypes(include=[np.number])

    stats = {
        "rows": len(df),
        "columns": list(df.columns),
        "mean": clean_dict(numeric_df.mean().to_dict()),
        "std": clean_dict(numeric_df.std().to_dict()),
        "variance": clean_dict(numeric_df.var().to_dict()),
        "min": clean_dict(numeric_df.min().to_dict()),
        "max": clean_dict(numeric_df.max().to_dict()),
        "median": clean_dict(numeric_df.median().to_dict()),
        "mode": clean_dict(numeric_df.mode().dropna().iloc[0].to_dict())
                if not numeric_df.empty and len(numeric_df.mode().dropna()) > 0 else {},
        "range": clean_dict((numeric_df.max() - numeric_df.min()).to_dict()),
        "allowed_values": {},
        "value_range": {},
        "correlation": []
    }

    return stats


@app.get("/health")
async def health():
    """Simple endpoint to confirm the server + API key are configured."""
    return {"status": "ok", "api_key_set": bool(OPENROUTER_API_KEY)}
