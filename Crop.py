# detect_and_crop.py
import os
from pathlib import Path
from ultralytics import YOLO
import cv2
import pandas as pd
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='runs/detect/train/weights/best.pt', help='YOLO model path')
parser.add_argument('--src', type=str, default='dataset/valid/images', help='source images folder')
parser.add_argument('--out', type=str, default='dataset/crops', help='output crops folder')
parser.add_argument('--conf', type=float, default=0.3, help='min confidence')
args = parser.parse_args()

Path(args.out).mkdir(parents=True, exist_ok=True)
meta = []  # rows: crop_filename, orig_image, x1,y1,x2,y2,score

device = 'cpu'  # ultralytics will pick GPU if available unless you forced CPU earlier
model = YOLO(args.model)

img_paths = sorted(Path(args.src).glob('*.*'))
crop_idx = 0
for img_path in img_paths:
    img = cv2.imread(str(img_path))
    res = model.predict(source=str(img_path), conf=args.conf, device=device, save=False, verbose=False)[0]
    # res.boxes holds boxes (xyxy)
    if len(res.boxes) == 0:
        continue
    for box in res.boxes:
        xyxy = box.xyxy[0].cpu().numpy()  # [x1,y1,x2,y2]
        score = float(box.conf[0].cpu().numpy())
        x1,y1,x2,y2 = map(int, xyxy)
        # Clamp to image
        h,w = img.shape[:2]
        x1,x2 = max(0,x1), min(w-1,x2)
        y1,y2 = max(0,y1), min(h-1,y2)
        crop = img[y1:y2, x1:x2]
        if crop.size == 0:
            continue
        crop_name = f'crop_{crop_idx:05d}.jpg'
        cv2.imwrite(str(Path(args.out)/crop_name), crop)
        meta.append({'crop_filename': crop_name,
                     'orig_image': img_path.name,
                     'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
                     'score': score})
        crop_idx += 1

df = pd.DataFrame(meta)
df.to_csv(Path(args.out)/'crops_meta.csv', index=False)
print('Saved crops:', len(df), '->', args.out)
