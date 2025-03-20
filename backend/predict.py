import argparse
import torch
from PIL import Image
import torch.nn as nn
from torchvision import models, transforms
import matplotlib.pyplot as plt
import numpy as np
import os
from config import IDX_TO_LABEL

def load_model(model_path):
    # Create the model architecture
    model = models.resnet18(weights=None)
    
    # Load saved model weights
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
    
    # Get the number of classes from the saved model
    num_features = model.fc.in_features
    num_classes = len(checkpoint['class_to_idx'])
    model.fc = nn.Linear(num_features, num_classes)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    class_to_idx = checkpoint['class_to_idx']
    
    # Create a mapping from model's class indices to human-readable labels
    # Use IDX_TO_LABEL from config for better readability
    idx_to_name = {v: IDX_TO_LABEL[v] for v in class_to_idx.values()}
    
    return model, idx_to_name

def process_image(image_path):
    # Define the same image transformations used in validation
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Check if image exists
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")
    
    # Open the image
    img = Image.open(image_path)
    
    # Convert to RGB if image has alpha channel
    if img.mode == 'RGBA':
        print("Converting RGBA image to RGB")
        img = img.convert('RGB')
    elif img.mode != 'RGB':
        print(f"Converting {img.mode} image to RGB")
        img = img.convert('RGB')
    
    # Print image information for debugging
    print(f"Image size: {img.size}, Mode: {img.mode}")
    
    # Apply transformations
    img_tensor = preprocess(img)
    
    # Add batch dimension
    img_tensor = img_tensor.unsqueeze(0)
    
    return img_tensor, img

def predict(image_path, model_path, top_k=3, display=True):
    # Load the model
    model, idx_to_name = load_model(model_path)
    model.eval()
    
    # Process the image
    img_tensor, original_img = process_image(image_path)
    
    # Predict with the model
    with torch.no_grad():
        output = model(img_tensor)
        probabilities = torch.nn.functional.softmax(output[0], dim=0)
        
        # Get top K probabilities and indices
        top_probs, top_indices = torch.topk(probabilities, top_k)
        
    # Convert to lists
    top_probs = top_probs.numpy()
    top_indices = top_indices.numpy()
    
    # Map indices to class names
    top_classes = [idx_to_name[idx] for idx in top_indices]
    
    # Print results
    print(f"\nPredictions for {image_path}:")
    for i in range(top_k):
        print(f"{top_classes[i]}: {top_probs[i]*100:.2f}%")
    
    # Display the image with prediction if requested
    if display:
        plt.figure(figsize=(10, 5))
        
        # Display the image
        plt.subplot(1, 2, 1)
        plt.imshow(np.array(original_img))
        plt.title(f"Prediction: {top_classes[0]}")
        plt.axis('off')
        
        # Display the probabilities
        plt.subplot(1, 2, 2)
        y_pos = np.arange(top_k)
        plt.barh(y_pos, top_probs * 100)
        plt.yticks(y_pos, top_classes)
        plt.xlabel('Probability (%)')
        plt.title('Top Predictions')
        plt.tight_layout()
        plt.show()
    
    return top_classes, top_probs

def main():
    parser = argparse.ArgumentParser(description="Test Sports Pitch Classifier on a single image")
    parser.add_argument("--image_path", type=str, required=True, 
                        help="Path to the image to test")
    parser.add_argument("--model_path", type=str, default="models/pitch_classifier.pth",
                        help="Path to the saved model")
    parser.add_argument("--top_k", type=int, default=3,
                        help="Display top K predictions")
    parser.add_argument("--no_display", action="store_true",
                        help="Don't display the image with predictions")
    
    args = parser.parse_args()
    
    # Check if model exists
    if not os.path.exists(args.model_path):
        print(f"Model file not found: {args.model_path}")
        print(f"Working directory: {os.getcwd()}")
        print(f"Available files in models directory:")
        if os.path.exists("models"):
            print(os.listdir("models"))
        else:
            print("models directory not found")
        return
    
    # Make prediction
    predict(args.image_path, args.model_path, args.top_k, not args.no_display)

if __name__ == "__main__":
    main()
