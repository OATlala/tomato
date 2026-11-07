"""
Tomato leaf disease classifier (ResNet18 transfer learning)
รันบน Google Colab (Runtime > Change runtime type > GPU)

ติดตั้งก่อน:  !pip install -q kagglehub scikit-learn
แล้วรัน:      !python train.py
"""
import json
import random
from pathlib import Path

import kagglehub
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

# ---------------- Config (แก้ตรงนี้ให้ตรงกับที่ทีมตกลงกัน) ----------------
SEED = 42
CLASSES = ["Tomato_Early_blight", "Tomato_Late_blight", "Tomato_healthy"]
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 8
LR = 1e-4
SPLIT = (0.70, 0.15, 0.15)  # train / val / test
# ---------------------------------------------------------------------------

random.seed(SEED)
torch.manual_seed(SEED)
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device)

# 1) โหลด dataset (PlantVillage จาก Kaggle)
root = Path(kagglehub.dataset_download("arjuntejaswi/plant-village"))
print("Dataset at:", root)


def find_class_dir(name):
    for p in root.rglob(name):
        if p.is_dir():
            return p
    raise FileNotFoundError(f"ไม่เจอโฟลเดอร์ {name} ใน {root}")


# 2) แบ่ง train/val/test แบบ stratified (แยกตามคลาส ไม่ให้ภาพซ้ำข้ามชุด)
splits = {"train": [], "val": [], "test": []}
for idx, cls in enumerate(CLASSES):
    files = sorted(
        f for f in find_class_dir(cls).iterdir()
        if f.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    random.shuffle(files)
    n = len(files)
    n_tr, n_va = int(n * SPLIT[0]), int(n * SPLIT[1])
    splits["train"] += [(f, idx) for f in files[:n_tr]]
    splits["val"] += [(f, idx) for f in files[n_tr:n_tr + n_va]]
    splits["test"] += [(f, idx) for f in files[n_tr + n_va:]]
    print(f"{cls}: {n} images")

print({k: len(v) for k, v in splits.items()})

# 3) Transforms (augmentation เฉพาะ train)
norm = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
train_tf = transforms.Compose([
    transforms.RandomResizedCrop(IMG_SIZE, scale=(0.7, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(25),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
    transforms.ToTensor(),
    norm,
])
eval_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    norm,
])


class LeafDataset(Dataset):
    def __init__(self, items, tf):
        self.items, self.tf = items, tf

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        path, label = self.items[i]
        return self.tf(Image.open(path).convert("RGB")), label


loaders = {
    "train": DataLoader(LeafDataset(splits["train"], train_tf), BATCH_SIZE, shuffle=True, num_workers=2),
    "val": DataLoader(LeafDataset(splits["val"], eval_tf), BATCH_SIZE, num_workers=2),
    "test": DataLoader(LeafDataset(splits["test"], eval_tf), BATCH_SIZE, num_workers=2),
}

# 4) Model: ResNet18 pretrained (ImageNet) + เปลี่ยนชั้นสุดท้าย
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
model.fc = nn.Linear(model.fc.in_features, len(CLASSES))
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)


def evaluate(loader):
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for x, y in loader:
            out = model(x.to(device))
            preds += out.argmax(1).cpu().tolist()
            labels += y.tolist()
    return labels, preds


# 5) Train loop (เก็บโมเดลที่ val accuracy ดีที่สุด)
best_acc = 0.0
for epoch in range(1, EPOCHS + 1):
    model.train()
    total_loss = 0.0
    for x, y in loaders["train"]:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * x.size(0)

    labels, preds = evaluate(loaders["val"])
    val_acc = sum(a == b for a, b in zip(labels, preds)) / len(labels)
    print(f"Epoch {epoch}/{EPOCHS}  loss={total_loss / len(splits['train']):.4f}  val_acc={val_acc:.4f}")

    if val_acc > best_acc:
        best_acc = val_acc
        torch.save(model.state_dict(), "model.pt")

# 6) ประเมินผลบน test set ด้วยโมเดลที่ดีที่สุด
model.load_state_dict(torch.load("model.pt", map_location=device))
labels, preds = evaluate(loaders["test"])
print("\n=== Test results ===")
print(classification_report(labels, preds, target_names=CLASSES, digits=4))
print("Confusion matrix:\n", confusion_matrix(labels, preds))

# 7) บันทึกไฟล์ที่ต้องส่งให้คนที่ 3
with open("class_names.json", "w", encoding="utf-8") as f:
    json.dump({"classes": CLASSES, "img_size": IMG_SIZE}, f, ensure_ascii=False, indent=2)
print("\nSaved: model.pt, class_names.json")