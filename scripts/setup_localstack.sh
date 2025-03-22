#!/bin/bash
set -e

echo "Setting up LocalStack S3 bucket and models..."

# Configure AWS CLI to use LocalStack endpoint
export AWS_ACCESS_KEY_ID=test
export AWS_SECRET_ACCESS_KEY=test
export AWS_DEFAULT_REGION=us-east-1
export AWS_ENDPOINT_URL=http://localhost:4566

# Check if LocalStack container is running
LOCALSTACK_CONTAINER=$(docker ps | grep localstack | awk '{print $1}')
if [ -z "$LOCALSTACK_CONTAINER" ]; then
  echo "Error: LocalStack container is not running!"
  echo "Please check docker-compose logs for more details:"
  echo "docker-compose logs localstack"
  exit 1
fi

# Check LocalStack health endpoint
echo "Checking LocalStack health..."
HEALTH_OUTPUT=$(curl -s http://localhost:4566/_localstack/health)
if echo "$HEALTH_OUTPUT" | grep -q "\"s3\": \"running\"" || echo "$HEALTH_OUTPUT" | grep -q "\"s3\": \"available\""; then
  echo "LocalStack S3 service is running/available"
else
  echo "Error: LocalStack S3 service is not running or available"
  echo "LocalStack health status:"
  echo "$HEALTH_OUTPUT"
  echo ""
  echo "Check docker logs for more details:"
  echo "docker logs $LOCALSTACK_CONTAINER"
  exit 1
fi

echo "LocalStack health check passed. S3 service is ready."

# Wait for LocalStack to be fully operational
echo "Waiting for LocalStack S3 to be fully operational..."
MAX_RETRIES=20
count=0
until aws --endpoint-url=http://localhost:4566 s3 ls > /dev/null 2>&1 || [ $count -ge $MAX_RETRIES ]; do
  echo "LocalStack S3 not fully ready yet. Waiting... (Attempt $((count+1))/$MAX_RETRIES)"
  sleep 3
  count=$((count+1))
done

if [ $count -ge $MAX_RETRIES ]; then
  echo "Warning: LocalStack S3 did not become fully operational in time."
  echo "Will attempt to continue anyway..."
  echo "LocalStack logs:"
  docker logs $LOCALSTACK_CONTAINER --tail 30
fi

echo "Creating S3 bucket for models..."

# Create bucket if it doesn't exist (with retry logic)
BUCKET_NAME="sports-pitch-models"
echo "Creating $BUCKET_NAME bucket..."

MAX_ATTEMPTS=5
for attempt in $(seq 1 $MAX_ATTEMPTS); do
  if aws --endpoint-url=http://localhost:4566 s3 mb "s3://$BUCKET_NAME" 2>/dev/null; then
    echo "Bucket created successfully."
    break
  elif aws --endpoint-url=http://localhost:4566 s3 ls "s3://$BUCKET_NAME" 2>/dev/null; then
    echo "Bucket already exists, continuing..."
    break
  else
    echo "Attempt $attempt/$MAX_ATTEMPTS: Failed to create bucket. Retrying in 3 seconds..."
    sleep 3
    if [ $attempt -eq $MAX_ATTEMPTS ]; then
      echo "Warning: Could not create bucket after $MAX_ATTEMPTS attempts. Will try to continue..."
    fi
  fi
done

# Upload models from local directory
if [ -d "./backend/models" ]; then
  echo "Uploading models to LocalStack S3..."
  
  # Check if there are any .pth files in the directory
  PTH_FILES=$(find ./backend/models -name "*.pth" 2>/dev/null)
  if [ -z "$PTH_FILES" ]; then
    echo "No .pth model files found in ./backend/models/"
    echo "Creating a dummy model file for testing..."
    touch ./backend/models/dummy_model.pth
    echo "Dummy model file created."
  fi
  
  for model in ./backend/models/*.pth; do
    if [ -f "$model" ]; then
      echo "Uploading $(basename "$model") to s3://$BUCKET_NAME/model/"
      for attempt in $(seq 1 $MAX_ATTEMPTS); do
        if aws --endpoint-url=http://localhost:4566 s3 cp "$model" "s3://$BUCKET_NAME/model/$(basename "$model")" 2>/dev/null; then
          echo "Model uploaded successfully: $(basename "$model")"
          break
        else
          echo "Attempt $attempt/$MAX_ATTEMPTS: Failed to upload model. Retrying in 3 seconds..."
          sleep 3
          if [ $attempt -eq $MAX_ATTEMPTS ]; then
            echo "Warning: Could not upload model $(basename "$model") after $MAX_ATTEMPTS attempts."
          fi
        fi
      done
    fi
  done
else
  echo "No models directory found at ./backend/models/"
  echo "Creating the directory structure..."
  mkdir -p ./backend/models
  echo "Creating a dummy model file for testing..."
  touch ./backend/models/dummy_model.pth
  
  echo "Uploading dummy model to LocalStack S3..."
  for attempt in $(seq 1 $MAX_ATTEMPTS); do
    if aws --endpoint-url=http://localhost:4566 s3 cp "./backend/models/dummy_model.pth" "s3://$BUCKET_NAME/model/dummy_model.pth" 2>/dev/null; then
      echo "Dummy model uploaded successfully."
      break
    else
      echo "Attempt $attempt/$MAX_ATTEMPTS: Failed to upload dummy model. Retrying in 3 seconds..."
      sleep 3
      if [ $attempt -eq $MAX_ATTEMPTS ]; then
        echo "Warning: Could not upload dummy model after $MAX_ATTEMPTS attempts."
      fi
    fi
  done
fi

echo "LocalStack setup completed as best as possible!"

# List bucket contents to verify
echo "Trying to list bucket contents:"
aws --endpoint-url=http://localhost:4566 s3 ls "s3://$BUCKET_NAME/" --recursive || {
  echo "Could not list bucket contents. This might be expected until S3 is fully operational."
}

echo "You can access LocalStack services at: http://localhost:4566"
echo "S3 bucket has been configured: $BUCKET_NAME"
echo ""
echo "NOTE: Even if there were warnings, the Lambda function may still work correctly."
echo "Try accessing the frontend at http://localhost:3000"
