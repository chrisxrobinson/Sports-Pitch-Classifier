# Sports Pitch Classifier Frontend

This is the web interface for the Sports Pitch Classifier, allowing users to upload images and get classification results.

## Development

### Prerequisites
- Node.js (v14+)
- npm or yarn

### Local Development
```bash
# Install dependencies
npm install

# Start the development server
npm start
```

### Building for Production
```bash
# Create production build
npm run build
```

## Docker Development

### Building the Docker Image
```bash
docker build -t sports-pitch-frontend .
```

### Running the Container
```bash
docker run -p 3000:80 sports-pitch-frontend
```

## S3 Upload Flow

The frontend application uploads images directly to S3 using AWS SDK v3:
1. Images are selected by the user
2. Images are resized in the browser to reduce size
3. Resized images are uploaded directly to S3 using the AWS SDK
4. The S3 key is sent to the Lambda function for processing
5. The Lambda function loads the image from S3 and runs inference

## Environment Variables

The application can be configured using the following environment variables:

- `REACT_APP_API_URL`: URL of the Lambda function endpoint
- `REACT_APP_AVAILABLE_MODELS`: Comma-separated list of available model filenames
- `REACT_APP_MODEL_BUCKET`: S3 bucket name where models are stored
- `REACT_APP_AWS_ACCESS_KEY_ID`: AWS access key for S3 operations
- `REACT_APP_AWS_SECRET_ACCESS_KEY`: AWS secret key for S3 operations
- `REACT_APP_AWS_ENDPOINT_URL`: AWS endpoint URL (for LocalStack)
- `REACT_APP_AWS_REGION`: AWS region for S3 operations
