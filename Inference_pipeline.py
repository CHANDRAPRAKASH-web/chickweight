# inference_pipeline.py
import argparse
import os
from pathlib import Path
import json
import math

import torch
import torchvision
from torchvision import transforms
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import cv2
from ultralytics import YOLO

# ---- MLP model definition (must match train_mlp.MLP) ----
import torch.nn as nn
class MLP(nn.Module):
    def __init__(self, in_dim=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.ReLU(),
            nn.Linear(256,128),
            nn.ReLU(),
            nn.Linear(128,1)
        )
    def forward(self,x):
        return self.net(x).squeeze(1)

# ---- helper functions ----
def xyxy_to_xywh(box):
    x1,y1,x2,y2 = box
    w = x2-x1
    h = y2-y1
    cx = x1 + w/2
    cy = y1 + h/2
    return (cx, cy, w, h)

def sort_boxes(boxes):
    """
    boxes: list of [x1,y1,x2,y2]
    returns indices sorted left->right then top->bottom (stable)
    """
    # compute center x,y
    centers = [( (b[0]+b[2])/2.0, (b[1]+b[3])/2.0 ) for b in boxes]
    # sort by x then y (but to avoid mixing rows, use row threshold based on box height)
    # simpler approach: sort by x then y
    idxs = list(range(len(boxes)))
    idxs.sort(key=lambda i: (centers[i][0], centers[i][1]))
    return idxs

def draw_annotations(image_pil, boxes, scores, labels, weights_pred, save_path):
    """
    image_pil: PIL Image
    boxes: list of [x1,y1,x2,y2]
    scores: list of floats
    labels: list of str (e.g., 'chick_1')
    weights_pred: list of floats
    """
    draw = ImageDraw.Draw(image_pil)
    try:
        font = ImageFont.truetype("arial.ttf", 18)
    except:
        font = ImageFont.load_default()
    for b, s, lab, w in zip(boxes, scores, labels, weights_pred):
        x1,y1,x2,y2 = map(int, b)
        # box
        draw.rectangle([x1,y1,x2,y2], outline="lime", width=2)
        txt = f"{lab}: {w:.0f} g ({s:.2f})"
        text_size = draw.textsize(txt, font=font)
        # draw filled rectangle for text background
        draw.rectangle([x1, y1 - text_size[1] - 4, x1 + text_size[0] + 6, y1], fill="lime")
        draw.text((x1+2, y1 - text_size[1] - 2), txt, fill="black", font=font)
    image_pil.save(save_path)

# ---- main pipeline ----
def run_inference(
    input_path,
    yolo_model_path="runs/detect/train/weights/best.pt",
    mlp_path="models/mlp.pth",
    resnet_pretrained=True,
    out_dir="runs/inference",
    conf_thres=0.25,
    device=None
):
    device = torch.device(device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu"))
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    # --- load YOLO detector ---
    print("Loading YOLO model:", yolo_model_path)
    yolo = YOLO(yolo_model_path)

    # --- load feature extractor (resnet18 without fc) ---
    print("Loading ResNet18 backbone (for feature extraction).")
    resnet = torchvision.models.resnet18(pretrained=resnet_pretrained)
    modules = list(resnet.children())[:-1]
    backbone = torch.nn.Sequential(*modules).to(device)
    backbone.eval()

    tf = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225])
    ])

    # --- load MLP regressor ---
    print("Loading MLP regressor:", mlp_path)
    # need to know in_dim; assuming 512 from resnet18
    model_mlp = MLP(in_dim=512).to(device)
    model_mlp.load_state_dict(torch.load(mlp_path, map_location=device))
    model_mlp.eval()

    # ---- handle single image or folder ----
    input_path = Path(input_path)
    images = []
    if input_path.is_dir():
        for ext in ("*.jpg","*.jpeg","*.png","*.bmp"):
            images.extend(sorted(input_path.glob(ext)))
    else:
        images = [input_path]

    results_all = []

    for img_path in images:
        print("Processing:", img_path)
        # run yolo predict (returns list of results)
        # we request boxes, scores
        res = yolo.predict(str(img_path), conf=conf_thres, device=device.type if device.type!='cpu' else 'cpu', imgsz=640)
        # ultralytics returns results list (1 per image) - use first
        if len(res) == 0:
            print("No result returned by YOLO for", img_path)
            continue
        r = res[0]
        # r.boxes: each has xyxy, conf, id
        boxes = []
        scores = []
        for box in r.boxes:
            xyxy = box.xyxy.cpu().numpy().tolist()[0]  # [x1,y1,x2,y2]
            conf = float(box.conf.cpu().numpy())
            boxes.append(xyxy)
            scores.append(conf)

        if len(boxes) == 0:
            print("No detections in", img_path)
            results_all.append({
                "image": str(img_path),
                "detections": [],
                "min_weight": None,
                "max_weight": None,
                "avg_weight": None,
                "annotated_image": None
            })
            continue

        # crop each detected box and compute feature
        crops = []
        boxes_int = []
        for b in boxes:
            x1,y1,x2,y2 = map(int, b)
            boxes_int.append([x1,y1,x2,y2])
            pil = Image.open(img_path).convert("RGB")
            w = pil.width; h = pil.height
            # clamp coordinates
            x1c = max(0, min(x1, w-1))
            y1c = max(0, min(y1, h-1))
            x2c = max(0, min(x2, w-1))
            y2c = max(0, min(y2, h-1))
            if x2c<=x1c or y2c<=y1c:
                # skip degenerate
                crops.append(None)
                continue
            crop = pil.crop((x1c,y1c,x2c,y2c))
            crops.append(crop)

        # prepare features
        feats = []
        valid_idx = []
        with torch.no_grad():
            for i,c in enumerate(crops):
                if c is None:
                    feats.append(None); continue
                x = tf(c).unsqueeze(0).to(device)
                f = backbone(x)  # 1x512x1x1
                f = f.squeeze().cpu().numpy().reshape(-1)
                feats.append(f)
                valid_idx.append(i)

        # predict weights with mlp
        preds = [None] * len(feats)
        if len(valid_idx) > 0:
            batch_X = torch.tensor(np.vstack([feats[i] for i in valid_idx]), dtype=torch.float32).to(device)
            with torch.no_grad():
                out = model_mlp(batch_X).cpu().numpy()
            # place back
            for k, i in enumerate(valid_idx):
                preds[i] = float(out[k])

        # Sort boxes to stable ordering (left->right then top->bottom)
        order = sort_boxes(boxes_int)
        labels = []
        preds_ordered = []
        scores_ordered = []
        boxes_ordered = []
        for idx, ord_i in enumerate(order, start=1):
            labels.append(f"chick_{idx}")
            preds_ordered.append(preds[ord_i] if preds[ord_i] is not None else float("nan"))
            scores_ordered.append(scores[ord_i])
            boxes_ordered.append(boxes_int[ord_i])

        # compute min/max/avg ignoring nan
        weights_valid = [p for p in preds_ordered if (p is not None and not math.isnan(p))]
        min_w = float(min(weights_valid)) if weights_valid else None
        max_w = float(max(weights_valid)) if weights_valid else None
        avg_w = float(sum(weights_valid)/len(weights_valid)) if weights_valid else None

        # save annotated image
        pil_img = Image.open(img_path).convert("RGB")
        out_annot = Path(out_dir) / f"{img_path.stem}_annotated{img_path.suffix}"
        draw_annotations(pil_img, boxes_ordered, scores_ordered, labels, preds_ordered, str(out_annot))

        # build result dict
        detections = []
        for lab, box, sc, pr in zip(labels, boxes_ordered, scores_ordered, preds_ordered):
            detections.append({
                "label": lab,
                "box": [int(x) for x in box],
                "conf": float(sc),
                "pred_weight_g": (None if pr is None or math.isnan(pr) else float(pr))
            })

        result = {
            "image": str(img_path),
            "annotated_image": str(out_annot),
            "detections": detections,
            "min_weight_g": min_w,
            "max_weight_g": max_w,
            "avg_weight_g": avg_w
        }

        results_all.append(result)
        # print quick summary
        print("Saved annotated image ->", out_annot)
        print(f"Detected {len(detections)} chicks. avg={avg_w}g min={min_w}g max={max_w}g")

    return results_all

# ---- CLI ----
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, help="input image file or folder")
    p.add_argument("--yolo", default="runs/detect/train/weights/best.pt", help="yolo model path")
    p.add_argument("--mlp", default="models/mlp.pth", help="mlp model path")
    p.add_argument("--out", default="runs/inference", help="output dir")
    p.add_argument("--conf", type=float, default=0.25, help="yolo confidence threshold")
    p.add_argument("--device", default=None, help="cpu or cuda (auto if None)")
    args = p.parse_args()

    results = run_inference(
        args.input,
        yolo_model_path=args.yolo,
        mlp_path=args.mlp,
        resnet_pretrained=True,
        out_dir=args.out,
        conf_thres=args.conf,
        device=args.device
    )
    print(json.dumps(results, indent=2))
