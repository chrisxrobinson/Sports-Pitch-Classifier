import React, { useState, useEffect, useRef } from 'react';
import { Container, Typography, Button, Box, FormControl, InputLabel, Select, MenuItem, Paper, CircularProgress } from '@mui/material';
import { Bar } from 'react-chartjs-2';
import { Chart as ChartJS, CategoryScale, LinearScale, BarElement, Title, Tooltip, Legend } from 'chart.js';
import axios from 'axios';
import { S3Client, PutObjectCommand } from '@aws-sdk/client-s3';
import './App.css';

ChartJS.register(CategoryScale, LinearScale, BarElement, Title, Tooltip, Legend);

const s3Client = new S3Client({
  region: process.env.REACT_APP_AWS_REGION || 'us-east-1',
  endpoint: process.env.REACT_APP_AWS_ENDPOINT_URL || 'http://localhost:4566',
  credentials: {
    accessKeyId: process.env.REACT_APP_AWS_ACCESS_KEY_ID || 'test',
    secretAccessKey: process.env.REACT_APP_AWS_SECRET_ACCESS_KEY || 'test'
  },
  forcePathStyle: true,
  requestHandler: {
    retryMode: 'standard',
    maxAttempts: 3
  },
  customUserAgent: 'Sports-Pitch-Classifier',
});

function App() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [selectedModel, setSelectedModel] = useState('');
  const [availableModels, setAvailableModels] = useState([]);
  const [predictions, setPredictions] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);
  const [uploadProgress, setUploadProgress] = useState(0);
  
  const fileInputRef = useRef(null);

  const resizeImage = (file) => {
    return new Promise((resolve) => {
      const reader = new FileReader();
      reader.readAsDataURL(file);
      reader.onload = (event) => {
        const img = new Image();
        img.src = event.target.result;
        img.onload = () => {
          const MAX_WIDTH = 500;
          const MAX_HEIGHT = 500;
          
          let width = img.width;
          let height = img.height;
          
          if (width > height) {
            if (width > MAX_WIDTH) {
              height = Math.round((height * MAX_WIDTH) / width);
              width = MAX_WIDTH;
            }
          } else {
            if (height > MAX_HEIGHT) {
              width = Math.round((width * MAX_HEIGHT) / height);
              height = MAX_HEIGHT;
            }
          }
          
          // Create canvas and resize image
          const canvas = document.createElement('canvas');
          canvas.width = width;
          canvas.height = height;
          
          const ctx = canvas.getContext('2d');
          // Use white background for PNG transparency
          ctx.fillStyle = "white";
          ctx.fillRect(0, 0, width, height);
          ctx.drawImage(img, 0, 0, width, height);
          
          // Always convert to JPEG for API transmission (smaller file size)
          canvas.toBlob((blob) => {
            resolve(blob);
          }, 'image/jpeg', 0.4); // Lower quality for even smaller file size
        };
      };
    });
  };

  const uploadToS3 = async (file) => {
    try {
      const timestamp = Date.now();
      const randomString = Math.random().toString(36).substring(2, 10);
      const filename = `${timestamp}-${randomString}.jpg`;
      const s3Key = `images/${filename}`;
      
      const fileBuffer = await new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.onload = () => resolve(reader.result);
        reader.onerror = (error) => reject(error);
        reader.readAsArrayBuffer(file);
      });
      
      const command = new PutObjectCommand({
        Bucket: process.env.REACT_APP_MODEL_BUCKET || 'sports-pitch-models',
        Key: s3Key,
        Body: fileBuffer,
        ContentType: 'image/jpeg',
        ACL: 'public-read'
      });
      
      try {
        await s3Client.send(command);
        setUploadProgress(100);
        return s3Key;
      } catch (sendError) {
        if (sendError.name === 'TypeError' && sendError.message === 'Failed to fetch') {
          throw new Error('Failed to connect to S3. Please check if LocalStack is running and accessible.');
        }
        throw sendError;
      }
    } catch (error) {
      throw new Error(`Failed to upload image to S3: ${error.message}`);
    }
  };

  useEffect(() => {    
    const modelsString = process.env.REACT_APP_AVAILABLE_MODELS || 'pitch_classifier_v1.pth';
    const models = modelsString.split(',');
    setAvailableModels(models);
    setSelectedModel(models[0]);
  }, []);

  const handleFileChange = (event) => {
    const file = event.target.files[0];
    if (file) {
      setSelectedFile(file);
      
      const reader = new FileReader();
      reader.onloadend = () => {
        setPreview(reader.result);
      };
      reader.readAsDataURL(file);
      
      setPredictions(null);
      setError(null);
      setUploadProgress(0);
    }
  };

  const handleModelChange = (event) => {
    setSelectedModel(event.target.value);
    setPredictions(null);
    setError(null);
  };

  const handleSubmit = async () => {
    if (!selectedFile || !selectedModel) {
      setError('Please select both an image and a model');
      return;
    }

    setIsLoading(true);
    setError(null);
    setUploadProgress(0);

    try {
      const resizedImage = await resizeImage(selectedFile);
      
      setUploadProgress(10);
      
      const s3Key = await uploadToS3(resizedImage);
      
      const payload = {
        s3_key: s3Key,
        image_bucket: process.env.REACT_APP_MODEL_BUCKET || 'sports-pitch-models',
        model_key: selectedModel,
        model_bucket: process.env.REACT_APP_MODEL_BUCKET || 'sports-pitch-models'
      };

      const requestPayload = { body: JSON.stringify(payload) };

      try {
        const response = await axios.post(
          process.env.REACT_APP_API_URL || '/api',
          requestPayload,
          {
            headers: {
              'Content-Type': 'application/json'
            }
          }
        );

        if (response.data.predictions) {
          setPredictions(response.data.predictions);
        } else {
          setError('Invalid response format');
        }
      } catch (err) {
        setError(err.response?.data?.error || err.message || 'Error classifying image');
      } finally {
        setIsLoading(false);
      }
    } catch (err) {
      setError('Error processing image: ' + err.message);
      setIsLoading(false);
    }
  };

  const resetForm = () => {
    setSelectedFile(null);
    setPreview(null);
    setPredictions(null);
    setError(null);
    setUploadProgress(0);
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  // Prepare chart data if predictions are available
  const chartData = predictions ? {
    labels: predictions.map(p => p.class),
    datasets: [
      {
        label: 'Confidence (%)',
        data: predictions.map(p => (p.confidence * 100).toFixed(2)),
        backgroundColor: 'rgba(75, 192, 192, 0.6)',
        borderColor: 'rgba(75, 192, 192, 1)',
        borderWidth: 1
      }
    ]
  } : null;

  const chartOptions = {
    indexAxis: 'y',
    responsive: true,
    maintainAspectRatio: false,
    scales: {
      x: {
        beginAtZero: true,
        max: 100,
        title: {
          display: true,
          text: 'Confidence (%)'
        }
      }
    },
    plugins: {
      legend: {
        display: false
      },
      title: {
        display: true,
        text: 'Top Predictions'
      }
    }
  };

  return (
    <Container maxWidth="md" sx={{ py: 4 }}>
      <Typography variant="h3" component="h1" gutterBottom align="center">
        Sports Pitch Classifier
      </Typography>
      
      <Paper elevation={3} sx={{ p: 3, mb: 4 }}>
        <Box mb={3}>
          <Typography variant="h6" gutterBottom>
            Upload an image to classify
          </Typography>
          
          <Box display="flex" flexDirection="column" gap={2}>
            <input
              type="file"
              accept="image/*"
              onChange={handleFileChange}
              ref={fileInputRef}
              style={{ display: 'none' }}
              id="image-upload"
            />
            <label htmlFor="image-upload">
              <Button variant="contained" component="span">
                Select Image
              </Button>
            </label>
            
            {preview && (
              <Box mt={2} textAlign="center">
                <img 
                  src={preview} 
                  alt="Preview" 
                  style={{ maxWidth: '100%', maxHeight: '300px' }} 
                />
              </Box>
            )}
            
            <FormControl fullWidth margin="normal">
              <InputLabel id="model-select-label">Model</InputLabel>
              <Select
                labelId="model-select-label"
                id="model-select"
                value={selectedModel}
                label="Model"
                onChange={handleModelChange}
              >
                {availableModels.map((model) => (
                  <MenuItem key={model} value={model}>
                    {model}
                  </MenuItem>
                ))}
              </Select>
            </FormControl>
            
            {uploadProgress > 0 && uploadProgress < 100 && (
              <Box width="100%" mt={1}>
                <Typography variant="body2" gutterBottom>
                  Uploading: {uploadProgress}%
                </Typography>
                <div 
                  style={{ 
                    height: '4px', 
                    width: '100%', 
                    backgroundColor: '#e0e0e0',
                    borderRadius: '4px',
                    overflow: 'hidden'
                  }}
                >
                  <div 
                    style={{ 
                      height: '100%', 
                      width: `${uploadProgress}%`, 
                      backgroundColor: '#1976d2',
                      transition: 'width 0.3s ease'
                    }} 
                  />
                </div>
              </Box>
            )}
            
            <Box display="flex" gap={2} mt={2}>
              <Button 
                variant="contained" 
                color="primary" 
                onClick={handleSubmit}
                disabled={!selectedFile || !selectedModel || isLoading}
                fullWidth
              >
                {isLoading ? <CircularProgress size={24} color="inherit" /> : 'Classify'}
              </Button>
              
              <Button 
                variant="outlined" 
                onClick={resetForm}
                disabled={isLoading}
                fullWidth
              >
                Reset
              </Button>
            </Box>
          </Box>
        </Box>
        
        {error && (
          <Box mt={3} p={2} bgcolor="error.light" borderRadius={1}>
            <Typography color="error.dark">{error}</Typography>
          </Box>
        )}
        
        {predictions && (
          <Box mt={4}>
            <Typography variant="h6" gutterBottom>
              Classification Results
            </Typography>
            
            <Box display="flex" flexDirection={{ xs: 'column', md: 'row' }} gap={4}>
              {/* Image with top prediction */}
              <Box flex="1" textAlign="center">
                <Typography variant="subtitle1" gutterBottom>
                  Predicted: {predictions[0].class}
                </Typography>
                {preview && (
                  <img 
                    src={preview} 
                    alt={predictions[0].class} 
                    style={{ maxWidth: '100%', maxHeight: '250px' }} 
                  />
                )}
              </Box>
              
              {/* Bar chart with probabilities */}
              <Box flex="1" height="250px">
                {chartData && (
                  <Bar data={chartData} options={chartOptions} />
                )}
              </Box>
            </Box>
          </Box>
        )}
      </Paper>
      
      <Typography variant="body2" color="textSecondary" align="center" sx={{ mt: 3 }}>
        &copy; {new Date().getFullYear()} Sports Pitch Classifier
      </Typography>
    </Container>
  );
}

export default App;
