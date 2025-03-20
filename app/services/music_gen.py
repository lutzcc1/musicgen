"""
Music Generation Service using Meta's AudioCraft MusicGen model
"""

import torch
import torchaudio
from audiocraft.models import MusicGen
import math
from scipy.io import wavfile
import numpy as np

def display_audio_info(audio, sample_rate):
    """Print information about the generated audio"""
    print(f"Generated audio with shape {audio.shape} at {sample_rate}Hz sample rate")
    print(f"Duration: {audio.shape[-1]/sample_rate:.2f} seconds")

def text_to_music(descriptions, model_size='small', duration=10, use_sampling=True, top_k=250):
    """Generate music from text descriptions"""
    print(f"Loading MusicGen {model_size} model...")
    model = MusicGen.get_pretrained(f'facebook/musicgen-{model_size}')

    # Set generation parameters
    model.set_generation_params(
        use_sampling=use_sampling,
        top_k=top_k,
        duration=duration
    )

    # Generate music
    print(f"Generating music from text: {descriptions}")
    output = model.generate(descriptions=descriptions, progress=True)

    # Display info and return the audio
    display_audio_info(output, 32000)
    return output, 32000

def continue_music(audio_file, descriptions=None, prompt_duration=2, duration=10):
    """Continue music from an audio file with optional text descriptions"""
    print(f"Loading MusicGen small model...")
    model = MusicGen.get_pretrained('facebook/musicgen-small')

    # Set generation parameters
    model.set_generation_params(duration=duration)

    # Load and prepare the prompt audio
    print(f"Loading audio prompt from {audio_file}")
    prompt_waveform, prompt_sr = torchaudio.load(audio_file)
    prompt_waveform = prompt_waveform[..., :int(prompt_duration * prompt_sr)]

    # Generate music continuation
    print("Generating music continuation...")
    output = model.generate_continuation(
        prompt_waveform,
        prompt_sample_rate=prompt_sr,
        descriptions=descriptions,
        progress=True
    )

    # Display info and return the audio
    display_audio_info(output, 32000)
    return output, 32000

def save_audio(audio, sample_rate, filename):
    """Save audio to file"""
    audio_np = audio.cpu().numpy()[0]

    # Normalize audio to be in the [-1, 1] range if it's not already
    if audio_np.max() > 1.0 or audio_np.min() < -1.0:
        audio_np = audio_np / max(abs(audio_np.max()), abs(audio_np.min()))

    # Convert to 16-bit PCM
    audio_np = (audio_np * 32767).astype(np.int16)

    # Transpose the audio array if it has more than one channel
    if audio_np.ndim > 1:
        audio_np = audio_np.T

    wavfile.write(filename, sample_rate, audio_np)
    print(f"Audio saved to {filename}")