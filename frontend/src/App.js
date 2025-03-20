import React, { useState, useEffect, useRef } from 'react';
import { Container, Typography, Button, Box, FormControl, InputLabel, Select, MenuItem, Paper, CircularProgress } from '@mui/material';
import { Bar } from 'react-chartjs-2';
import { Chart as ChartJS, CategoryScale, LinearScale, BarElement, Title, Tooltip, Legend } from 'chart.js';
import axios from 'axios';
import './App.css';

// Register Chart.js components
ChartJS.register(CategoryScale, LinearScale, BarElement, Title, Tooltip, Legend);

function App() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [selectedModel, setSelectedModel] = useState('');
  const [availableModels, setAvailableModels] = useState([]);
  const [predictions, setPredictions] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);
  
  const fileInputRef = useRef(null);

  useEffect(() => {
    // Get available models from environment variable
    const modelsString = process.env.REACT_APP_AVAILABLE_MODELS || 'pitch_classifier_v1.pth';
    const models = modelsString.split(',');
    setAvailableModels(models);
    setSelectedModel(models[0]);
  }, []);

  const handleFileChange = (event) => {
    const file = event.target.files[0];
    if (file) {
      setSelectedFile(file);
      
      // Create a preview
      const reader = new FileReader();
      reader.onloadend = () => {
        setPreview(reader.result);
      };
      reader.readAsDataURL(file);
      
      // Reset predictions when a new file is selected
      setPredictions(null);
      setError(null);
    }
  };

  const handleModelChange = (event) => {
    setSelectedModel(event.target.value);
    // Reset predictions when a new model is selected
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

    try {
      // Convert image to base64
      const reader = new FileReader();
      reader.readAsDataURL(selectedFile);
      
      reader.onloadend = async () => {
        // Extract base64 data (remove "data:image/jpeg;base64," part)
        const base64Data = reader.result.split(',')[1];
        
        // Prepare request payload
        const payload = {
          image_data: base64Data,
          model_key: selectedModel,
          model_bucket: process.env.REACT_APP_MODEL_BUCKET || 'sports-pitch-models'
        };

        // Call Lambda function
        const response = await axios.post(
          process.env.REACT_APP_API_URL || '/api',
          payload,
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
        setIsLoading(false);
      };
    } catch (err) {
      console.error('Error classifying image:', err);
      setError(err.response?.data?.error || err.message || 'Error classifying image');
      setIsLoading(false);
    }
  };

  const resetForm = () => {
    setSelectedFile(null);
    setPreview(null);
    setPredictions(null);
    setError(null);
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
