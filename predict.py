"""
ไฟล์นี้ส่งให้คนที่ 3 (Flask API) ใช้คู่กับ model.pt และ class_names.json

ตัวอย่าง:
    from predict import predict
    result = predict("leaf.jpg")   # หรือส่ง PIL.Image ก็ได้
    # {"label": "Tomato_Early_blight", "confidence": 0.97, "probabilities": {...}}
"""
import json

import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms

_device = "cuda" if torch.cuda.is_available() else "cpu"

with open("class_names.json", encoding="utf-8") as f:
    _meta = json.load(f)
CLASSES = _meta["classes"]
IMG_SIZE = _meta["img_size"]

_model = models.resnet18(weights=None)
_model.fc = nn.Linear(_model.fc.in_features, len(CLASSES))
_model.load_state_dict(torch.load("model.pt", map_location=_device))
_model.to(_device).eval()

_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def predict(image):
    """image: path (str) หรือ PIL.Image -> dict ผลทำนาย"""
    if isinstance(image, str):
        image = Image.open(image)
    x = _tf(image.convert("RGB")).unsqueeze(0).to(_device)
    with torch.no_grad():
        probs = torch.softmax(_model(x), dim=1)[0].cpu().tolist()
    best = max(range(len(probs)), key=probs.__getitem__)
    return {
        "label": CLASSES[best],
        "confidence": round(probs[best], 4),
        "probabilities": {c: round(p, 4) for c, p in zip(CLASSES, probs)},
    }