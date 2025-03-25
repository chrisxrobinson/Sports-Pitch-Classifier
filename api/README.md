# Sports Pitch Classifier API

This API provides image classification for sports pitches and other aerial imagery using a PyTorch model.

## Local Development

### Running with Docker
```bash
# Build and run the API with Docker
docker build -t sports-pitch-api .
docker run -p 8000:8000 sports-pitch-api
```

### Running Locally without Docker
```bash
# Install dependencies
pip install -e .

# Run the FastAPI app
uvicorn app:app --reload
```

## API Interface

### Endpoints

- `GET /` - Welcome message and API status check
- `POST /predict` - Predict sports pitch classes from image

### Request Format
```json
{
  "image_data": "base64encodedimage",
  "s3_key": "path/to/image.jpg",
  "image_bucket": "sports-pitch-models",
  "model_key": "pitch_classifier_v1.pth",
  "model_bucket": "sports-pitch-models"
}
```

Notes:
- Provide either `image_data` (base64-encoded) or `s3_key` (referencing an S3 object)
- `image_bucket` and `model_bucket` are optional and will use the default bucket if not specified

### Response Format
```json
{
  "predictions": [
    {
      "class": "baseball_diamond",
      "class_idx": 2,
      "confidence": 0.95
    },
    {
      "class": "tennis_court",
      "class_idx": 41,
      "confidence": 0.03
    },
    {
      "class": "basketball_court",
      "class_idx": 3,
      "confidence": 0.01
    }
  ]
}
```

## Interactive Documentation

When the API is running, you can access the interactive API documentation at:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc
