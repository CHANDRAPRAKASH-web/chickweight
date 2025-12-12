# extract_features.py
import torch, torchvision
from torchvision import transforms
from PIL import Image
import numpy as np
from pathlib import Path
import pandas as pd
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--crops', type=str, default='dataset/crops', help='crops folder')
parser.add_argument('--out', type=str, default='dataset/features', help='output folder')
args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

Path(args.out).mkdir(parents=True, exist_ok=True)

# load pretrained resnet18
resnet = torchvision.models.resnet18(pretrained=True)
# remove final fc
modules = list(resnet.children())[:-1]
backbone = torch.nn.Sequential(*modules)
backbone.to(device)
backbone.eval()

tf = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225])
])

crops = sorted(Path(args.crops).glob('crop_*.jpg'))
features = []
meta = []
with torch.no_grad():
    for p in crops:
        img = Image.open(p).convert('RGB')
        x = tf(img).unsqueeze(0).to(device)
        feat = backbone(x)  # shape 1x512x1x1
        feat = feat.squeeze().cpu().numpy().reshape(-1)  # 512
        features.append(feat)
        meta.append({'crop_filename': p.name})
features = np.vstack(features)  # Nx512
np.save(Path(args.out)/'features.npy', features)
pd.DataFrame(meta).to_csv(Path(args.out)/'crops_features_meta.csv', index=False)
print('Saved features:', features.shape, '->', args.out)
