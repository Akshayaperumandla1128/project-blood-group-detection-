import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import joblib
import numpy as np
import os

class CNNFeatureExtractor:
    def __init__(self):
        model = models.resnet18(pretrained=True)
        self.feature_layer = nn.Sequential(*list(model.children())[:-1])
        self.feature_layer.eval()
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def get_vector(self, image_path):
        img = Image.open(image_path).convert('RGB')
        img_t = self.transform(img).unsqueeze(0)
        with torch.no_grad():
            features = self.feature_layer(img_t)
        return features.flatten().numpy()

extractor = CNNFeatureExtractor()

def get_cnn_features(image_path):
    return extractor.get_vector(image_path)

# Inside model_utils.py
import numpy as np
import joblib

def predict_blood_group(image_path, minutiae_vector):
    try:
        # 1. Extract CNN Features (Macro)
        cnn_vec = get_cnn_features(image_path)
        
        # 2. NORMALIZATION: Bring Minutiae (0-100) into same range as CNN (0-1)
        # This prevents the "B Negative" or "A+" stuck loop
        minutiae_scaled = minutiae_vector / 100.0
        
        # 3. Fusion (512 + 2 = 514 features)
        combined = np.hstack((cnn_vec, minutiae_scaled)).reshape(1, -1)
        
        # 4. Predict using the RBF-trained SVM
        svm = joblib.load('models/hybrid_svm_blood_group.pkl')
        le = joblib.load('models/label_encoder.pkl')
        
        idx = svm.predict(combined)[0]
        group = le.inverse_transform([idx])[0]
        
        # 5. Get Probability (Confidence)
        probs = svm.predict_proba(combined)[0]
        confidence = f"{int(np.max(probs) * 100)}%"
        
        return {
            "blood_group": group,
            "confidence": confidence,
            "status": "Success"
        }
    except Exception as e:
        return {"status": "Error", "message": str(e)}