import os
import io
import tempfile
import base64
from typing import Optional, Dict, List
from pydantic import BaseModel
import boto3
import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from torchvision import transforms, models

app = FastAPI(title="Sports Pitch Classifier API")

LABEL_MAPPING = {
    '0': 'airplane',
    '1': 'airport',
    '2': 'baseball_diamond',
    '3': 'basketball_court',
    '4': 'beach',
    '5': 'bridge',
    '6': 'chaparral',
    '7': 'church',
    '8': 'circular_farmland',
    '9': 'cloud',
    '10': 'commercial_area',
    '11': 'dense_residential',
    '12': 'desert',
    '13': 'forest',
    '14': 'freeway',
    '15': 'golf_course',
    '16': 'ground_track_field',
    '17': 'harbor',
    '18': 'industrial_area',
    '19': 'intersection',
    '20': 'island',
    '21': 'lake',
    '22': 'meadow',
    '23': 'medium_residential',
    '24': 'mobile_home_park',
    '25': 'mountain',
    '26': 'overpass',
    '27': 'palace',
    '28': 'parking_lot',
    '29': 'railway',
    '30': 'railway_station',
    '31': 'rectangular_farmland',
    '32': 'river',
    '33': 'roundabout',
    '34': 'runway',
    '35': 'sea_ice',
    '36': 'ship',
    '37': 'snowberg',
    '38': 'sparse_residential',
    '39': 'stadium',
    '40': 'storage_tank',
    '41': 'tennis_court',
    '42': 'terrace',
    '43': 'thermal_power_station',
    '44': 'wetland'
}

SPORTS_CLASSES = [
    'baseball_diamond',
    'basketball_court',
    'golf_course',
    'ground_track_field',
    'tennis_court',
    'stadium'
]

IDX_TO_LABEL = {i: label for i, label in enumerate(LABEL_MAPPING.values())}

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DEFAULT_BUCKET = os.environ.get('MODEL_BUCKET', 'sports-pitch-models')
AWS_REGION = os.environ.get('AWS_REGION', 'us-east-1')
AWS_ENDPOINT_URL = os.environ.get('AWS_ENDPOINT_URL', None)

s3_client = boto3.client(
    's3',
    region_name=AWS_REGION,
    endpoint_url=AWS_ENDPOINT_URL,
    aws_access_key_id=os.environ.get('AWS_ACCESS_KEY_ID', 'test'),
    aws_secret_access_key=os.environ.get('AWS_SECRET_ACCESS_KEY', 'test')
)

preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

class ImageRequest(BaseModel):
    image_data: Optional[str] = None
    s3_key: Optional[str] = None
    image_bucket: Optional[str] = None
    model_key: str = "pitch_classifier.pth"
    model_bucket: Optional[str] = None

def download_model_from_s3(bucket, key):
    try:
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_path = tmp_file.name
        
        s3_client.download_file(bucket, key, tmp_path)
        return tmp_path
    except Exception as e:
        try:
            alternative_key = key.replace('model/', '')
            s3_client.download_file(bucket, alternative_key, tmp_path)
            return tmp_path
        except Exception as nested_e:
            raise HTTPException(
                status_code=500, 
                detail=f"Failed to download model from S3: {str(e)}, Alternative attempt error: {str(nested_e)}"
            )

def get_image_from_s3(bucket, key):
    try:
        response = s3_client.get_object(Bucket=bucket, Key=key)
        return response['Body'].read()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to download image from S3: {str(e)}")

def load_model(model_path):
    model = models.resnet18(weights=None)
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
    num_features = model.fc.in_features
    num_classes = len(checkpoint['class_to_idx'])
    model.fc = torch.nn.Linear(num_features, num_classes)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model, checkpoint['class_to_idx']

@app.get("/")
async def root():
    return {"message": "Welcome to the Sports Pitch Classifier API"}

@app.get("/classes")
async def get_classes():
    """Return a list of classifiable categories"""
    return {
        "all_classes": LABEL_MAPPING,
        "sports_classes": SPORTS_CLASSES
    }

@app.post("/classify")
async def predict(request: ImageRequest):
    try:
        image_data = None
        if request.image_data:
            image_data = base64.b64decode(request.image_data)
        elif request.s3_key:
            image_bucket = request.image_bucket or DEFAULT_BUCKET
            image_data = get_image_from_s3(image_bucket, request.s3_key)
        else:
            raise HTTPException(
                status_code=400, 
                detail="Missing image_data or s3_key in request"
            )
        
        model_name = request.model_key
        model_bucket = request.model_bucket or DEFAULT_BUCKET
        
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
                class_name = idx_to_class.get(class_idx, str(class_idx))
                
                readable_label = None
                
                if str(class_name).isdigit():
                    readable_label = LABEL_MAPPING.get(str(class_name))
                
                if readable_label is None:
                    readable_label = LABEL_MAPPING.get(class_name)
                
                if readable_label is None:
                    readable_label = IDX_TO_LABEL.get(class_idx)
                
                if readable_label is None:
                    readable_label = class_name
                
                if not isinstance(readable_label, str):
                    readable_label = str(readable_label)
                
                is_sport = readable_label in SPORTS_CLASSES
                
                predictions.append({
                    "class": readable_label,
                    "class_idx": int(class_idx),
                    "confidence": float(top_probs[i].item()),
                    "is_sport": bool(is_sport)
                })
                
        if os.path.exists(model_path):
            os.remove(model_path)
            
        return {
            "predictions": predictions,
            "sports_classes": SPORTS_CLASSES
        }
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
