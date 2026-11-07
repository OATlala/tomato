import sys
from pathlib import Path

from predict import predict

folder = Path(sys.argv[1] if len(sys.argv) > 1 else "early_1")
exts = {".jpg", ".jpeg", ".png"}

files = sorted(f for f in folder.rglob("*") if f.suffix.lower() in exts and not f.name.startswith("._"))
if not files:
    print("ไม่เจอรูปในโฟลเดอร์", folder)

for f in files:
    r = predict(str(f))
    print(f"{f.name:30s} -> {r['label']} ({r['confidence'] * 100:.1f}%)")
    for name, p in r["probabilities"].items():
        print(f"      {name:22s} {p * 100:5.1f}%")