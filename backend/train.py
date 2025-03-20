import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from sklearn.model_selection import train_test_split

from dataset import load_resisc45_dataset

# Custom dataset class to convert HF dataset to PyTorch format
class RESISC45Dataset(Dataset):
    def __init__(self, dataset, indices=None, transform=None):
        self.dataset = dataset
        self.indices = indices if indices is not None else range(len(dataset))
        self.transform = transform
        self.classes = sorted(set(self.dataset[i]['label'] for i in self.indices))
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}
        
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        data_idx = self.indices[idx]
        img = self.dataset[data_idx]['image']
        label = self.dataset[data_idx]['label']
        
        if self.transform:
            img = self.transform(img)
            
        # Convert original dataset label to our class index
        label_idx = self.class_to_idx[label]
        return img, label_idx

train_transforms = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(degrees=15),  # New: Random rotation
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # New: Color jitter
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], 
                         [0.229, 0.224, 0.225])
])
val_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], 
                         [0.229, 0.224, 0.225])
])

def prepare_dataloaders(batch_size=32, test_size=0.2, random_state=42):
    dataset = load_resisc45_dataset()
    
    # Split indices for train and validation
    indices = list(range(len(dataset)))
    train_indices, val_indices = train_test_split(
        indices, test_size=test_size, random_state=random_state, stratify=[dataset[i]['label'] for i in indices]
    )
    
    train_dataset = RESISC45Dataset(dataset, train_indices, transform=train_transforms)
    val_dataset = RESISC45Dataset(dataset, val_indices, transform=val_transforms)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    return train_loader, val_loader, train_dataset

# -------------------------------
# Fine-Tuning More Layers
# -------------------------------
def get_model(num_classes):
    # Load pre-trained ResNet18
    model = models.resnet18(weights='IMAGENET1K_V1')
    
    # Freeze all layers except 'layer4' and the final fully connected layer
    for name, param in model.named_parameters():
        if "layer4" not in name and "fc" not in name:
            param.requires_grad = False
    
    # Replace the final fully-connected layer
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, num_classes)
    
    return model

# -------------------------------
# Training Loop with LR Scheduler
# -------------------------------
def train_model(model, dataloaders, criterion, optimizer, num_epochs=10, patience=2):
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    model.to(device)
    
    best_model_wts = model.state_dict()
    best_acc = 0.0
    no_improve_epochs = 0

    # Initialize a learning rate scheduler (reduces LR by factor 0.1 every 5 epochs)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
    
    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)
        
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
                dataloader = dataloaders['train']
            else:
                model.eval()   # Set model to evaluation mode
                dataloader = dataloaders['val']
            
            running_loss = 0.0
            running_corrects = 0
            
            for inputs, labels in dataloader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                optimizer.zero_grad()
                
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            
            epoch_loss = running_loss / len(dataloader.dataset)
            epoch_acc = running_corrects.float() / len(dataloader.dataset)
            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
            
            if phase == 'val':
                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = model.state_dict().copy()
                    no_improve_epochs = 0
                else:
                    no_improve_epochs += 1
        
        # Step the learning rate scheduler at the end of each epoch
        scheduler.step()
        
        if no_improve_epochs >= patience:
            print(f'Early stopping after {epoch+1} epochs')
            break
        
        print()
    
    print(f'Best val Acc: {best_acc:.4f}')
    model.load_state_dict(best_model_wts)
    return model

def save_model(model, class_to_idx, save_dir='models'):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # Implement versioning to prevent overwriting models
    model_prefix = 'pitch_classifier_v'
    existing_models = [f for f in os.listdir(save_dir) if f.startswith(model_prefix) and f.endswith('.pth')]
    
    # Extract version numbers from existing model files
    existing_versions = [0]  # Default to 0 if no existing versions
    for model_file in existing_models:
        try:
            version = int(model_file[len(model_prefix):-4])  # Extract version number
            existing_versions.append(version)
        except ValueError:
            continue
    
    # Create new version by incrementing the highest version found
    new_version = max(existing_versions) + 1
    model_filename = f"{model_prefix}{new_version}.pth"
    
    torch.save({
        'model_state_dict': model.state_dict(),
        'class_to_idx': class_to_idx
    }, os.path.join(save_dir, model_filename))
    
    print(f"Model saved to {os.path.join(save_dir, model_filename)}")

def main():
    parser = argparse.ArgumentParser(description='Train Sports Pitch Classifier')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--patience', type=int, default=2, help='Early stopping patience')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    
    args = parser.parse_args()
    
    train_loader, val_loader, train_dataset = prepare_dataloaders(batch_size=args.batch_size)
    
    num_classes = len(train_dataset.classes)
    model = get_model(num_classes)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    
    dataloaders = {
        'train': train_loader,
        'val': val_loader
    }
    
    model = train_model(model, dataloaders, criterion, optimizer, num_epochs=args.epochs, patience=args.patience)
    
    save_model(model, train_dataset.class_to_idx)

    print("Model classes:", train_dataset.classes)

if __name__ == "__main__":
    main()
