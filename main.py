import base64
import os
import io
import fastapi
from pydantic import BaseModel
import pandas as pd
import numpy as np
import requests

app = fastapi.FastAPI()

# Pull the API key securely from Render's Environment Variables
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

class AudioRequest(BaseModel):
    audio_id: str
    audio_base64: str

@app.post("/verify")
async def verify_audio(payload: AudioRequest):
    # 1. Decode Base64 to raw audio bytes
    audio_bytes = base64.b64decode(payload.audio_base64)
    
    # 2. Send the raw audio to OpenRouter's Whisper API
    # Using files tuple to send in-memory bytes without writing to Render's disk
    files = {
        'file': ('audio.wav', io.BytesIO(audio_bytes), 'audio/wav')
    }
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}"
    }
    data = {
        "model": "openai/whisper-large-v3", # Standard robust whisper model
        "response_format": "verbose_json",  # CRITICAL: This gives us timestamps/segments!
        "language": "ko"                     # Forces Korean transcription
    }
    
    response = requests.post(
        "https://openrouter.ai/api/v1/audio/transcriptions",
        headers=headers,
        files=files,
        data=data
    )
    
    result = response.json()
    
    # 3. Pull segments to build the DataFrame
    segments = result.get("segments", [])
    df = pd.DataFrame(segments)
    
    # Isolate numeric metrics (ids, start, end, temperature, etc.)
    numeric_df = df.select_dtypes(include=[np.number])
    
    # 4. Compute strict format structure
    # If a stat empty (like mode), fall back to an empty dictionary
    stats = {
        "rows": len(df),
        "columns": list(df.columns),
        "mean": numeric_df.mean().to_dict(),
        "std": numeric_df.std().to_dict(),
        "variance": numeric_df.var().to_dict(),
        "min": numeric_df.min().to_dict(),
        "max": numeric_df.max().to_dict(),
        "median": numeric_df.median().to_dict(),
        "mode": numeric_df.mode().dropna().iloc[0].to_dict() if not numeric_df.empty and len(numeric_df.mode().dropna()) > 0 else {},
        "range": (numeric_df.max() - numeric_df.min()).to_dict(),
        "allowed_values": {},  
        "value_range": {},    
        "correlation": numeric_df.corr().values.tolist() if not numeric_df.empty else []
    }
    
    return stats