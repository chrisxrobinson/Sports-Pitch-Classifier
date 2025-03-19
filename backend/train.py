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

# Data transformations
train_transforms = transforms.Compose([
    transforms.RandomResizedCrop(224),    # random crop to 224x224
    transforms.RandomHorizontalFlip(),    # random horizontal flip for augmentation
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], 
                         [0.229, 0.224, 0.225])  # normalise using ImageNet means
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
    
    # Create datasets
    train_dataset = RESISC45Dataset(dataset, train_indices, transform=train_transforms)
    val_dataset = RESISC45Dataset(dataset, val_indices, transform=val_transforms)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    return train_loader, val_loader, train_dataset

def get_model(num_classes):
    # Load pre-trained ResNet18
    model = models.resnet18(weights='IMAGENET1K_V1')
    
    # Freeze all parameters
    for param in model.parameters():
        param.requires_grad = False
    
    # Replace the final fully-connected layer
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, num_classes)
    
    return model

def train_model(model, dataloaders, criterion, optimizer, num_epochs=10, patience=2):
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    model.to(device)
    
    best_model_wts = model.state_dict()
    best_acc = 0.0
    no_improve_epochs = 0
    
    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)
        
        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
                dataloader = dataloaders['train']
            else:
                model.eval()   # Set model to evaluate mode
                dataloader = dataloaders['val']
            
            running_loss = 0.0
            running_corrects = 0
            
            # Iterate over data
            for inputs, labels in dataloader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                # Zero the parameter gradients
                optimizer.zero_grad()
                
                # Forward pass
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    
                    # Backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                
                # Statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            
            epoch_loss = running_loss / len(dataloader.dataset)
            epoch_acc = running_corrects.float() / len(dataloader.dataset)
            
            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
            
            # Save the best model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = model.state_dict().copy()
                no_improve_epochs = 0
            elif phase == 'val':
                no_improve_epochs += 1
        
        # Early stopping
        if no_improve_epochs >= patience:
            print(f'Early stopping after {epoch+1} epochs')
            break
        
        print()
    
    print(f'Best val Acc: {best_acc:.4f}')
    
    # Load best model weights
    model.load_state_dict(best_model_wts)
    return model

def save_model(model, class_to_idx, save_dir='models'):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    torch.save({
        'model_state_dict': model.state_dict(),
        'class_to_idx': class_to_idx
    }, os.path.join(save_dir, 'pitch_classifier.pth'))
    
    print(f"Model saved to {os.path.join(save_dir, 'pitch_classifier.pth')}")

def main():
    # Set up command line arguments
    parser = argparse.ArgumentParser(description='Train Sports Pitch Classifier')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--patience', type=int, default=2, help='Early stopping patience')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    
    args = parser.parse_args()
    
    # Create dataloaders
    train_loader, val_loader, train_dataset = prepare_dataloaders(batch_size=args.batch_size)
    
    # Create model
    num_classes = len(train_dataset.classes)
    model = get_model(num_classes)
    
    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=args.lr)
    
    dataloaders = {
        'train': train_loader,
        'val': val_loader
    }
    
    model = train_model(model, dataloaders, criterion, optimizer, 
                        num_epochs=args.epochs, patience=args.patience)
    
    save_model(model, train_dataset.class_to_idx)

if __name__ == "__main__":
    main()