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

### Local Development

#### Initial Setup

Set up the Python environment:

```bash
# Install dependencies for the backend
make init
```

#### Running with Docker

To start the complete local development environment with frontend, API, and MinIO (S3 replacement):

```bash
# Start all services and set up MinIO
make local-dev
```

This will:
1. Start the containers for frontend, Lambda API, and MinIO
2. Configure MinIO with the required buckets
3. Make the services available at:
   - Frontend: http://localhost:3000
   - Lambda API: http://localhost:9090
   - MinIO Console: http://localhost:9001 (login: minioadmin/minioadmin)
   - MinIO S3: http://localhost:9000

#### Individual Docker Commands

```bash
# Build Docker images separately
make docker-build-api
make docker-build-frontend
make docker-build  # Build all Docker images

# Start services
make docker-up

# Stop services
make docker-down

# Clean up Docker resources
make clean
```

### Development

```bash
# Run linting
make lint
```

## Model Training and Evaluation

See the backend README for detailed instructions on training and evaluating models.

## Deployment

For deployment instructions to AWS, refer to the deployment documentation in the respective directories:
- `frontend/README.md` for frontend deployment to S3
- `api/README.md` for Lambda function deployment
