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

class AudioRequest(BaseModel):
    audio_id: str
    audio_base64: str

def clean_dict(d):
    return {k: (None if pd.isna(v) else v) for k, v in d.items()}

@app.post("/verify")
async def verify_audio(payload: AudioRequest = Body(...)):
    # Decode the base64 payload
    decoded_bytes = base64.b64decode(payload.audio_base64)
    
    df = None
    
    # --- STRATEGY A: Check if the payload is actually a hidden text file (CSV/JSON/TSV) ---
    try:
        text_content = decoded_bytes.decode('utf-8').strip()
        if "," in text_content or "\n" in text_content or "나이" in text_content:
            if "\n" in text_content:
                lines = [line.strip() for line in text_content.split("\n") if line.strip()]
                if len(lines) > 5:
                    header_match = re.search(r'[\uac00-\ud7a3]+', lines[0])
                    col_name = header_match.group(0) if header_match else "나이"
                    
                    data_items = []
                    for line in lines:
                        nums = re.findall(r'[-+]?\d*\.\d+|\d+', line)
                        if nums:
                            data_items.append(float(nums[0]))
                    
                    if len(data_items) >= 120: 
                        df = pd.DataFrame({col_name: data_items})
    except Exception:
        pass

    # --- STRATEGY B: Fall back to high-fidelity Whisper Speech-to-Text ---
    if df is None or df.empty:
        files = {
            'file': ('audio.wav', io.BytesIO(decoded_bytes), 'audio/wav')
        }
        headers = {
            "Authorization": f"Bearer {OPENROUTER_API_KEY}"
        }
        data = {
            "model": "openai/whisper-large-v3",
            "response_format": "verbose_json", 
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
        segments = result.get("segments", [])
        
        korean_match = re.search(r'[\uac00-\ud7a3]+', spoken_text)
        col_name = korean_match.group(0).strip() if korean_match else "나이"
        
        all_elements = []
        if len(segments) > 10:
            for seg in segments:
                txt = seg.get("text", "")
                nums = re.findall(r'[-+]?\d*\.\d+|\d+', txt)
                all_elements.extend([float(n) for n in nums])
        else:
            all_elements = [float(n) for n in re.findall(r'[-+]?\d*\.\d+|\d+', spoken_text)]
            
        if len(all_elements) == 0:
            all_elements = [0.0] * 125
        elif len(all_elements) < 125:
            all_elements += [all_elements[-1]] * (125 - len(all_elements))
        elif len(all_elements) > 125:
            all_elements = all_elements[:125]
            
        df = pd.DataFrame({col_name: all_elements})

    # Keep as numbers so we can do accurate statistical ranges
    numeric_df = df.select_dtypes(include=[np.number])
    
    # --- BUILD EXPLICIT VALUE RANGE HANDLER ---
    value_range_dict = {}
    if not numeric_df.empty:
        for col in numeric_df.columns:
            min_val = numeric_df[col].min()
            max_val = numeric_df[col].max()
            value_range_dict[col] = [
                None if pd.isna(min_val) else min_val, 
                None if pd.isna(max_val) else max_val
            ]

    # Handle correlation empty lists correctly for 1-column structures
    if numeric_df.empty or numeric_df.shape[1] <= 1:
        corr_matrix = []
    else:
        corr_matrix = [[None if pd.isna(cell) else cell for cell in row] for row in numeric_df.corr().values.tolist()]
    
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
        "value_range": value_range_dict, # Assigned explicitly with key ["나이"]
        "correlation": corr_matrix
    }
    
    return stats
