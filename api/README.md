# Sports Pitch Classifier API

This API provides image classification for sports pitches and other aerial imagery using a PyTorch model.

## Local Development

### Running with Docker
```bash
# Build and run the API with Docker
docker build -t sports-pitch-api .
docker run -p 9000:8080 sports-pitch-api
```

### Running in AWS Lambda Local Environment
```bash
# Test Lambda function locally using the AWS Lambda Runtime Interface Emulator
docker run -p 9000:8080 sports-pitch-api

# Then call the function using curl
curl -XPOST "http://localhost:9000/2015-03-31/functions/function/invocations" -d '{
  "body": "{\"image_data\": \"base64encodedimage\", \"model_key\": \"pitch_classifier_v1.pth\"}"
}'
```

## API Interface

### Request Format
```json
{
  "image_data": "base64encodedimage",
  "model_key": "pitch_classifier_v1.pth",
  "model_bucket": "sports-pitch-models"
}
```

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

## Deployment

The API is designed to be deployed as an AWS Lambda function. See the deployment scripts in the `scripts` directory for automated deployment options.
