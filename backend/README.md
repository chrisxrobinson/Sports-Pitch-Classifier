# Sports Pitch Classifier

This project is a deep learning classifier for identifying different types of sports pitches, land features, and other aerial imagery using the RESISC45 dataset.

## Setup

1. Install the required packages:
```bash
uv env
source .venv/bin/activate
uv sync 
```

## Training the Model

Train the model with default settings:
```bash
uv run python train.py
```

Or customize training with arguments:
```bash
uv run python train.py --epochs 15 --lr 0.001 --patience 3 --batch_size 64
```

Available arguments:
- `--epochs`: Number of training epochs (default: 10)
- `--lr`: Learning rate (default: 0.001)
- `--patience`: Early stopping patience (default: 2)
- `--batch_size`: Batch size for training (default: 32)

## Evaluating the Model

Evaluate the model on the test set:
```bash
uv run python evaluate.py --model_path models/pitch_classifier_v1.pth
```

You can specify a custom output file for the evaluation results:
```bash
uv run python evaluate.py --model_path models/pitch_classifier_v1.pth --output_file evaluation_results/my_evaluation.txt
```

## Making Predictions

Make predictions on a single image:
```bash
uv run python predict.py --image_path path/to/your/image.jpg --model_path models/pitch_classifier_v1.pth
```

Additional prediction options:
- `--top_k`: Number of top predictions to display (default: 3)
- `--no_display`: Don't show the image with predictions (useful for scripting)

## Dataset

This project uses the RESISC45 dataset, which contains 45 classes of remote sensing images:
- Sports-related classes: baseball_diamond, basketball_court, golf_course, ground_track_field, tennis_court, stadium
- Other classes include natural features (forest, lake, mountain), infrastructure (airport, bridge, railway), and more

## Model Architecture

The classifier uses a ResNet18 architecture pretrained on ImageNet, with the following modifications:
- Fine-tuning of layer4 and fully-connected layer
- Trained with data augmentation (random crops, flips, rotations, color jitter)
- Early stopping to prevent overfitting
