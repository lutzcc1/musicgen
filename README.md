# MusicGen API

A REST API service for generating music from text descriptions using Meta's AudioCraft MusicGen model.

## Features

- Generate music from text descriptions
- Continue music from an existing audio file
- Support for different model sizes (small, medium, large, melody)
- Configurable duration and generation parameters

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run locally:
```bash
uvicorn app.main:app --reload
```

3. Build and run with Docker:
```bash
docker build -t musicgen-api .
docker run -p 8000:8000 musicgen-api
```

## API Endpoints

### Generate Music
```
POST /api/v1/generate
Content-Type: application/json

{
    "descriptions": ["your text description here"],
    "model_size": "small",
    "duration": 10
}
```

### Continue Music
```
POST /api/v1/continue
Content-Type: multipart/form-data

audio_file: [WAV file]
descriptions: ["optional text description"]
duration: 10
```

## API Documentation

Once the server is running, visit:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## AWS Deployment

To deploy to AWS Elastic Beanstalk:

1. Install the AWS CLI and EB CLI
2. Initialize EB project:
```bash
eb init
```
3. Create and deploy:
```bash
eb create
```

## Notes

- The API uses temporary files for audio processing
- Generated audio is returned as WAV files
- Model files are downloaded on first use