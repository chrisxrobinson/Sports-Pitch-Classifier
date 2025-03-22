#!/bin/bash
set -e

echo "Setting up MinIO buckets and models..."

# Use Docker to run MinIO client commands
MC="docker run --rm -it --network sports-pitch-classifier_default minio/mc"

# Configure mc to use our MinIO instance
echo "Configuring MinIO client..."
$MC alias set myminio http://minio:9000 minioadmin minioadmin

# Wait for MinIO to be ready
echo "Waiting for MinIO to start..."
MAX_RETRIES=10
count=0
until $MC ls myminio > /dev/null 2>&1 || [ $count -ge $MAX_RETRIES ]; do
  echo "MinIO not ready yet. Waiting... (Attempt $((count+1))/$MAX_RETRIES)"
  sleep 3
  count=$((count+1))
done

if [ $count -ge $MAX_RETRIES ]; then
  echo "Error: MinIO did not become ready in time."
  echo "Please check if the MinIO container is running with 'docker ps'."
  exit 1
fi

# Create bucket if it doesn't exist
BUCKET_NAME="sports-pitch-models"
echo "Creating $BUCKET_NAME bucket..."
$MC mb "myminio/$BUCKET_NAME" --ignore-existing

# Ensure the model directory exists in the bucket
$MC mb "myminio/$BUCKET_NAME/model" --ignore-existing 2>/dev/null || true

# Upload models from local directory
if [ -d "./backend/models" ]; then
  echo "Uploading models to MinIO..."
  
  # Create temporary container with models mounted
  docker run --rm -it \
    --network sports-pitch-classifier_default \
    -v "$(pwd)/backend/models:/models" \
    --entrypoint sh \
    minio/mc -c "
      mc alias set myminio http://minio:9000 minioadmin minioadmin && 
      for model in /models/*.pth; do
        if [ -f \"\$model\" ]; then
          echo \"Uploading \$(basename \"\$model\") to myminio/$BUCKET_NAME/model/\"
          mc cp \"\$model\" \"myminio/$BUCKET_NAME/model/\"
        fi
      done
    "
else
  echo "No models directory found at ./backend/models/"
  echo "Creating the directory structure..."
  mkdir -p ./backend/models
  echo "Please add your .pth model files to the ./backend/models/ directory"
fi

echo "MinIO setup completed successfully!"
echo "You can access the MinIO console at: http://localhost:9001"
echo "Username: minioadmin, Password: minioadmin"
