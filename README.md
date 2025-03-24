# Sports Pitch Classifier

A cost-effective machine learning pipeline for classifying sports pitches from satellite images, leveraging AWS, PyTorch, React, Docker, and GitHub Actions.

## Project Overview

This project consists of three main components:
- **Backend**: PyTorch-based machine learning models for classifying satellite imagery
- **API**: AWS Lambda-compatible API for inference
- **Frontend**: React web application for uploading images and viewing predictions

## Setup and Development

### Prerequisites

- Docker and Docker Compose
- Python 3.12
- Node.js v14+
- `uv` Python package manager
- AWS CLI (for deploying to AWS and local testing)

### Local Development

#### Initial Setup

Set up the Python environment:

```bash
# Install dependencies for the backend
make init
```

#### Running with Docker

To start the complete local development environment with frontend, API, and LocalStack (AWS services emulator):

```bash
# Start all services and set up LocalStack
make up
```

This will:
1. Start the containers for frontend, Lambda API, and LocalStack
2. Configure LocalStack with the required buckets and services
3. Make the services available at:
   - Frontend: http://localhost:3000
   - Lambda API: http://localhost:9090
   - LocalStack AWS services: http://localhost:4566

#### Individual Docker Commands

```bash
# Build Docker images separately
make docker-build-api
make docker-build-frontend
make docker-build  # Build all Docker images

# Start services
make up

# Stop services
make down
```

#### Model Management

Upload trained models to S3:

```bash
# For AWS deployment:
S3_BUCKET=your-aws-bucket-name make upload-models

# For local testing with LocalStack:
AWS_ENDPOINT_URL=http://localhost:4566 S3_BUCKET=sports-pitch-models make upload-models
```

Note: The S3_BUCKET environment variable is required.

### Development

```bash
# Run linting
make lint
```

#### Testing the Lambda Function Locally

After starting the local development environment with `make up`, you can test the Lambda function directly:

```bash
# Test the Lambda function with a sample request
curl -XPOST "http://localhost:9090/2015-03-31/functions/function/invocations" \
  -d '{"body": "{\"image_data\":\"base64encodedimagedatahere\",\"model_key\":\"pitch_classifier_v1.pth\"}"}'
```

For a more complete test with an actual image:

```bash
# Convert image to base64 and send to Lambda
IMAGE_FILE="path/to/your/image.jpg"
BASE64_IMAGE=$(base64 -i "$IMAGE_FILE" | tr -d '\n')
curl -XPOST "http://localhost:9090/2015-03-31/functions/function/invocations" \
  -d "{\"body\": \"{\\\"image_data\\\":\\\"$BASE64_IMAGE\\\",\\\"model_key\\\":\\\"pitch_classifier_v1.pth\\\"}\"}"
```

The Lambda function also supports fetching images from S3:

```bash
# Upload an image to S3 first
aws --endpoint-url=http://localhost:4566 s3 cp path/to/image.jpg s3://sports-pitch-models/images/test-image.jpg

# Then reference it in your API call
curl -XPOST "http://localhost:9090/2015-03-31/functions/function/invocations" \
  -d '{"body": "{\"s3_key\":\"images/test-image.jpg\",\"image_bucket\":\"sports-pitch-models\",\"model_key\":\"pitch_classifier_v1.pth\"}"}'
```

You can also interact with LocalStack directly to check S3 buckets or other AWS resources:

```bash
# List S3 buckets
aws --endpoint-url=http://localhost:4566 s3 ls

# List contents of the models bucket
aws --endpoint-url=http://localhost:4566 s3 ls s3://sports-pitch-models/model/
```

#### S3 Upload Approach

The frontend now uses an S3-based approach for image processing:
1. Images are resized in the browser
2. The resized image is uploaded to S3
3. Only the S3 key is sent to the Lambda function
4. Lambda retrieves the image from S3 for processing

This approach resolves issues with large image uploads and is more efficient for serverless processing.

#### Troubleshooting

If you encounter issues with large images being rejected by the API, the frontend has been updated to automatically resize images before sending them to the backend. If you're testing with the API directly, consider:

1. Resizing large images before encoding them
2. Using the S3-based approach for very large images
3. Checking the logs with `make show-lambda-logs` or `make show-frontend-logs`

## Model Training and Evaluation

See the backend README for detailed instructions on training and evaluating models.

## Deployment

For deployment instructions to AWS, refer to the deployment documentation in the respective directories:
- `frontend/README.md` for frontend deployment to S3
- `api/README.md` for Lambda function deployment
