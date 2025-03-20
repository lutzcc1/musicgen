from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List, Optional
import tempfile
import os
from .services.music_gen import text_to_music, continue_music, save_audio

app = FastAPI(
    title="MusicGen API",
    description="API for generating music from text descriptions using Meta's AudioCraft MusicGen model",
    version="1.0.0"
)

class TextToMusicRequest(BaseModel):
    descriptions: List[str]
    model_size: str = "small"
    duration: int = 10

class ContinueMusicRequest(BaseModel):
    descriptions: Optional[List[str]] = None
    duration: int = 10

@app.post("/api/v1/generate")
async def generate_music(request: TextToMusicRequest):
    """
    Generate music from text descriptions
    """
    try:
        # Generate the audio
        audio, sr = text_to_music(
            request.descriptions,
            model_size=request.model_size,
            duration=request.duration
        )

        # Save to a temporary file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
            save_audio(audio, sr, temp_file.name)
            return FileResponse(
                temp_file.name,
                media_type="audio/wav",
                filename="generated_music.wav"
            )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/continue")
async def continue_music_endpoint(
    audio_file: UploadFile = File(...),
    request: ContinueMusicRequest = None
):
    """
    Continue music from an uploaded audio file
    """
    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_input:
            content = await audio_file.read()
            temp_input.write(content)
            temp_input.flush()

            # Generate continuation
            audio, sr = continue_music(
                temp_input.name,
                descriptions=request.descriptions if request else None,
                duration=request.duration if request else 10
            )

            # Save result to another temporary file
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_output:
                save_audio(audio, sr, temp_output.name)
                return FileResponse(
                    temp_output.name,
                    media_type="audio/wav",
                    filename="continued_music.wav"
                )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def root():
    return {"message": "Welcome to MusicGen API"}