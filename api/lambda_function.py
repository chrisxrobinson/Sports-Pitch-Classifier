import json
import base64
import io
import os
import tempfile
import boto3

from PIL import Image
import torch
from torchvision import transforms, models

s3_client = boto3.client('s3')

# Default bucket where models are stored
DEFAULT_BUCKET = os.environ.get('MODEL_BUCKET', 'sports-pitch-models')

def download_model_from_s3(bucket, key):
    try:
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_path = tmp_file.name
            
        s3_client.download_file(bucket, key, tmp_path)
        return tmp_path
    except Exception as e:
        raise Exception(f"Failed to download model from S3: {str(e)}")

def load_model(model_path):
    model = models.resnet18(weights=None)
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
    num_features = model.fc.in_features
    num_classes = len(checkpoint['class_to_idx'])
    model.fc = torch.nn.Linear(num_features, num_classes)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model, checkpoint['class_to_idx']

preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def lambda_handler(event, context):
    try:
        if event.get("isBase64Encoded", False):
            body_data = json.loads(base64.b64decode(event["body"]).decode('utf-8'))
        else:
            body_data = json.loads(event["body"]) if isinstance(event["body"], str) else event["body"]
        
        image_data = base64.b64decode(body_data.get("image_data"))
        model_name = body_data.get("model_key", "pitch_classifier.pth")
        model_bucket = body_data.get("model_bucket", DEFAULT_BUCKET)
        
        model_path = download_model_from_s3(model_bucket, f"model/{model_name}")
        
        model, class_to_idx = load_model(model_path)
        
        idx_to_class = {v: k for k, v in class_to_idx.items()}
        
        image = Image.open(io.BytesIO(image_data))
        if image.mode != "RGB":
            image = image.convert("RGB")

        input_tensor = preprocess(image).unsqueeze(0)

        with torch.no_grad():
            output = model(input_tensor)
            probabilities = torch.nn.functional.softmax(output[0], dim=0)
            
            top_probs, top_idxs = torch.topk(probabilities, 3)
            
            predictions = []
            for i in range(3):
                class_idx = top_idxs[i].item()
                predictions.append({
                    "class": idx_to_class.get(class_idx, str(class_idx)),
                    "class_idx": class_idx,
                    "confidence": top_probs[i].item()
                })

        if os.path.exists(model_path):
            os.remove(model_path)
            
        result = {
            "predictions": predictions
        }
        
        return {
            "statusCode": 200,
            "body": json.dumps(result)
        }
    except Exception as e:
        return {
            "statusCode": 500,
            "body": json.dumps({"error": str(e)})
        }