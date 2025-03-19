import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models
from sklearn.metrics import classification_report, confusion_matrix
import argparse

from dataset import load_resisc45_dataset
from train import RESISC45Dataset, val_transforms

def load_model(model_path, num_classes):
    model = models.resnet18(weights=None)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, num_classes)
    
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['model_state_dict'])
    class_to_idx = checkpoint['class_to_idx']
    
    return model, class_to_idx

def evaluate_model(model, test_loader, device):
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    return all_preds, all_labels

def prepare_test_dataset(test_size=0.1, random_state=42):
    dataset = load_resisc45_dataset()
    
    # Split indices for train+val and test
    indices = list(range(len(dataset)))
    labels = [dataset[i]['label'] for i in indices]
    
    # Using stratified split to ensure class distribution is maintained
    from sklearn.model_selection import train_test_split
    _, test_indices = train_test_split(
        indices, test_size=test_size, random_state=random_state, stratify=labels
    )
    
    # Create test dataset with validation transforms
    test_dataset = RESISC45Dataset(dataset, test_indices, transform=val_transforms)
    test_loader = DataLoader(test_dataset, batch_size=32)
    
    return test_loader, test_dataset

def main():
    parser = argparse.ArgumentParser(description="Evaluate Sports Pitch Classifier")
    parser.add_argument("--model_path", type=str, default="models/pitch_classifier.pth",
                        help="Path to the saved model")
    parser.add_argument("--output_file", type=str, default="evaluation_results.txt",
                        help="Path to save evaluation results")
    args = parser.parse_args()
    
    # Set device
    device = torch.device("cuda:0" if torch.cuda.is_available() else 
                         "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Prepare test dataset
    test_loader, test_dataset = prepare_test_dataset()
    
    # Load model
    model, class_to_idx = load_model(args.model_path, len(test_dataset.classes))
    model.to(device)
    
    # Map indices to class names
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    
    # Convert numeric indices to string class names
    # This fixes the TypeError in classification_report
    class_names = [str(idx_to_class[i]) for i in range(len(idx_to_class))]
    
    # Evaluate model
    all_preds, all_labels = evaluate_model(model, test_loader, device)
    
    # Print metrics
    print("\nConfusion Matrix:")
    cm = confusion_matrix(all_labels, all_preds)
    print(cm)
    
    print("\nClassification Report:")
    report = classification_report(all_labels, all_preds, target_names=class_names)
    print(report)
    
    # Save results to file
    with open(args.output_file, 'w') as f:
        f.write("Confusion Matrix:\n")
        f.write(str(cm))
        f.write("\n\nClassification Report:\n")
        f.write(report)
    
    print(f"\nResults saved to {args.output_file}")

if __name__ == "__main__":
    main()
