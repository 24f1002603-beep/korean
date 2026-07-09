import base64
import os
import io
import re
import fastapi
from fastapi import Body
from pydantic import BaseModel
import pandas as pd
import numpy as np
import requests

app = fastapi.FastAPI()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

# --- Define the incoming JSON structure ---
class AudioRequest(BaseModel):
    audio_id: str
    audio_base64: str

# Helper function to turn NaN values into None (JSON null) safely
def clean_dict(d):
    return {k: (None if pd.isna(v) else v) for k, v in d.items()}

# We use Body() here to force FastAPI to look at the JSON payload, not the URL
@app.post("/verify")
async def verify_audio(payload: AudioRequest = Body(...)):
    audio_bytes = base64.b64decode(payload.audio_base64)
    
    files = {
        'file': ('audio.wav', io.BytesIO(audio_bytes), 'audio/wav')
    }
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}"
    }
    data = {
        "model": "openai/whisper-large-v3",
        "response_format": "json",
        "language": "ko"
    }
    
    response = requests.post(
        "https://openrouter.ai/api/v1/audio/transcriptions",
        headers=headers,
        files=files,
        data=data
    )
    
    result = response.json()
    spoken_text = result.get("text", "").strip()
    
    # Isolate continuous Korean letters (Hangul) anywhere in the text
    korean_match = re.search(r'[\uac00-\ud7a3]+', spoken_text)
    col_name = korean_match.group(0).strip() if korean_match else "나이"
    
    # Extract numbers safely
    numbers = [float(x) for x in re.findall(r'[-+]?\d*\.\d+|\d+', spoken_text)]
    if not numbers:
        numbers = [0.0]

    # Create DataFrame
    df = pd.DataFrame({col_name: numbers})
    numeric_df = df.select_dtypes(include=[np.number])
    
    # Compute stats structure
    stats = {
        "rows": len(df),
        "columns": list(df.columns),
        "mean": clean_dict(numeric_df.mean().to_dict()),
        "std": clean_dict(numeric_df.std().to_dict()),
        "variance": clean_dict(numeric_df.var().to_dict()),
        "min": clean_dict(numeric_df.min().to_dict()),
        "max": clean_dict(numeric_df.max().to_dict()),
        "median": clean_dict(numeric_df.median().to_dict()),
        "mode": clean_dict(numeric_df.mode().dropna().iloc[0].to_dict()) if not numeric_df.empty and len(numeric_df.mode().dropna()) > 0 else {},
        "range": clean_dict((numeric_df.max() - numeric_df.min()).to_dict()),
        "allowed_values": {},  
        "value_range": {},    
        "correlation": [[None if pd.isna(cell) else cell for cell in row] for row in numeric_df.corr().values.tolist()] if not numeric_df.empty else []
    }
    
    return stats
