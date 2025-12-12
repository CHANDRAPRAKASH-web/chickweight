# quick_predict.py
import torch, numpy as np
from train_mlp import MLP   # or re-define the MLP class exactly the same

feat = np.load('dataset/features/features.npy')
print('features shape:', feat.shape)
model = MLP(in_dim=feat.shape[1])
model.load_state_dict(torch.load('models/mlp.pth', map_location='cpu'))
model.eval()

x = torch.tensor(feat[0:1], dtype=torch.float32)
pred = model(x).item()
print('pred grams =', pred)
