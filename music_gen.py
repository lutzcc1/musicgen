"""
Simple Music Generation using Meta's AudioCraft MusicGen model

This script implements text-to-music generation and music continuation from the official musicgen_demo.ipynb
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

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate music using MusicGen")
    parser.add_argument("--mode", type=str, default="text", choices=["text", "continue"],
                        help="Generation mode: text or continue")
    parser.add_argument("--descriptions", type=str, nargs="+",
                        default=["drum and bass beat with intense percussions"],
                        help="Text descriptions for the music")
    parser.add_argument("--audio_file", type=str, default=None,
                        help="Path to audio file for continuation mode")
    parser.add_argument("--duration", type=int, default=10,
                        help="Duration of generated audio in seconds")
    parser.add_argument("--model", type=str, default="small",
                        choices=["small", "medium", "large", "melody"],
                        help="Model size to use")
    parser.add_argument("--output", type=str, default="output.wav",
                        help="Output file name")

    args = parser.parse_args()

    # Run the appropriate function based on the mode
    if args.mode == "text":
        audio, sr = text_to_music(args.descriptions, args.model, args.duration)
    elif args.mode == "continue":
        if args.audio_file is None:
            raise ValueError("Audio file is required for continuation mode")
        audio, sr = continue_music(args.audio_file, args.descriptions, duration=args.duration)

    # Save the generated audio
    save_audio(audio, sr, args.output)
