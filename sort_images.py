"""
แยกรูปใบมะเขือเทศที่ปนกันในโฟลเดอร์เดียว ตามผลที่โมเดลทาย

วิธีใช้:  python sort_images.py mixed_test
ผลลัพธ์:  โฟลเดอร์ sorted_results/ แยกเป็น
            Tomato_healthy/       <- ใบปกติ (ดี)
            Tomato_Early_blight/  <- ใบเสีย
            Tomato_Late_blight/   <- ใบเสีย
            uncertain/            <- โมเดลไม่แน่ใจ (ความมั่นใจต่ำกว่า 70%)
          และไฟล์ results.csv

เคล็ดลับตรวจความถูกต้อง: ตั้งชื่อรูปให้ขึ้นต้นด้วย early / late / healthy
เช่น early_1.jpg แล้วสคริปต์จะคำนวณ % ที่ทายถูกให้
"""
import csv
import shutil
import sys
from pathlib import Path

from predict import predict

THRESHOLD = 0.70  # ต่ำกว่านี้ถือว่า "ไม่แน่ใจ"

src = Path(sys.argv[1] if len(sys.argv) > 1 else "mixed_test")
out = Path("sorted_results")
exts = {".jpg", ".jpeg", ".png"}

files = sorted(f for f in src.rglob("*") if f.suffix.lower() in exts and not f.name.startswith("._"))
if not files:
    print("ไม่เจอรูปในโฟลเดอร์", src)
    sys.exit()

if out.exists():
    shutil.rmtree(out)

hints = {
    "early": "Tomato_Early_blight",
    "late": "Tomato_Late_blight",
    "healthy": "Tomato_healthy",
}

rows = []
checked = correct = 0
for f in files:
    r = predict(str(f))
    label = r["label"] if r["confidence"] >= THRESHOLD else "uncertain"
    dest = out / label
    dest.mkdir(parents=True, exist_ok=True)
    shutil.copy(f, dest / f.name)
    rows.append([f.name, r["label"], r["confidence"]])

    mark = ""
    for key, cls in hints.items():
        if f.name.lower().startswith(key):
            checked += 1
            ok = r["label"] == cls
            correct += ok
            mark = "  ถูก" if ok else "  ผิด (เฉลย: " + cls + ")"
    print(f"{f.name:30s} -> {label} ({r['confidence'] * 100:.1f}%){mark}")

with open("results.csv", "w", newline="", encoding="utf-8") as fh:
    w = csv.writer(fh)
    w.writerow(["file", "predicted", "confidence"])
    w.writerows(rows)

print(f"\nรวม {len(files)} รูป -> ดูผลแยกโฟลเดอร์ที่ {out}/ และไฟล์ results.csv")
if checked:
    print(f"ทายถูก {correct}/{checked} รูป ({correct / checked * 100:.1f}%)")