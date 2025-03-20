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

## Deployment to AWS S3

This application is designed to be deployed as a static website on AWS S3.

1. Create an S3 bucket with static website hosting enabled
2. Build the React application
3. Upload the contents of the `build` directory to the S3 bucket
4. Configure environment variables using S3 bucket website configuration

See the deployment script in the `scripts` directory for an automated deployment option.

## Environment Variables

The application can be configured using the following environment variables:

- `REACT_APP_API_URL`: URL of the Lambda function endpoint
- `REACT_APP_AVAILABLE_MODELS`: Comma-separated list of available model filenames
- `REACT_APP_MODEL_BUCKET`: S3 bucket name where models are stored
