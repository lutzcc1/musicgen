# MusicGen Demo Script

This is a lean Python script implementation based on Meta's official [musicgen_demo.ipynb](https://github.com/facebookresearch/audiocraft/blob/main/demos/musicgen_demo.ipynb) notebook for generating music using AudioCraft's MusicGen model.

## Features

This script supports the following generation modes:
- Text-to-music generation
- Music continuation

## Prerequisites

- Python 3.9 or higher
- FFmpeg (for audio processing)
- CUDA-compatible GPU (recommended for faster generation)

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/your-username/musicgen.git
   cd musicgen
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

The script provides a command-line interface with different modes:

```
python music_gen.py --mode [text|continue] [OPTIONS]
```

### Text-to-Music Generation

Generate music based on text descriptions:

```
python music_gen.py --mode text --descriptions "drum and bass beat with intense percussions" --duration 30 --model medium --output generated_music.wav
```

### Music Continuation

Continue music from an audio prompt with optional text guidance:

```
python music_gen.py --mode continue --audio_file path/to/audio.mp3 --descriptions "Jazz with piano solo" --duration 15 --output continued_music.wav
```

### Command Line Arguments

- `--mode`: Generation mode (text or continue)
- `--descriptions`: Text descriptions for the music (one or more)
- `--audio_file`: Path to audio file for continuation mode
- `--duration`: Duration of generated audio in seconds (default: 10)
- `--model`: Model size to use (small, medium, large, or melody) (default: small)
- `--output`: Output file name (default: output.wav)

### Additional Parameters

For text-to-music generation, the following parameters are also available in the code:
- `use_sampling`: Whether to use sampling during generation (default: True)
- `top_k`: Number of top tokens to consider during sampling (default: 250)

## Model Sizes

- **small**: ~300M parameters, faster generation but lower quality (default)
- **medium**: ~1.5B parameters, balanced between quality and speed
- **large**: ~3.3B parameters, highest quality but slower generation
- **melody**: Specialized model for melody generation (~1.5B parameters)

## Notes

- The first time you run the script, it will download the model weights which may take some time depending on your internet connection.
- Larger models require more GPU memory.
- This implementation follows the examples shown in the official demo notebook.
- The script automatically normalizes audio output to ensure proper volume levels.

## Acknowledgements

This script uses the AudioCraft MusicGen model developed by Meta AI Research and is based on their official examples.