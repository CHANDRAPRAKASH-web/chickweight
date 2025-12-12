# train_mlp.py
import numpy as np, pandas as pd
import torch, torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
import argparse
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('--features', default='dataset/features/features.npy')
parser.add_argument('--meta', default='dataset/crops/crops_meta.csv')
parser.add_argument('--weights', default='weights.csv')   # path to your weights.csv (orig image -> weight)
parser.add_argument('--out', default='models', help='save dir')
parser.add_argument('--epochs', type=int, default=50)
parser.add_argument('--batch', type=int, default=32)
parser.add_argument('--lr', type=float, default=1e-3)
args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

X = np.load(args.features)    # N x 512
crops_meta = pd.read_csv(args.meta)   # has crop_filename, orig_image, ...
weights_df = pd.read_csv(args.weights) # filename, weight_g

# create mapping orig_image -> weight
weights_df['filename'] = weights_df['filename'].astype(str)
wmap = dict(zip(weights_df['filename'], weights_df['weight_g']))

# for each crop, get weight via orig_image
y = []
missing = 0
for _, r in crops_meta.iterrows():
    orig = r['orig_image']
    if orig in wmap:
        y.append(wmap[orig])
    else:
        y.append(np.nan)
        missing += 1

X = X[~np.isnan(y)]
y = np.array(y)[~np.isnan(y)].astype(np.float32)
print('Dropped', missing, 'crops with missing weight, final samples', len(y))

# split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# prepare loaders
train_ds = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32))
val_ds = TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.float32))
train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False)

# Small MLP
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

model = MLP(in_dim=X.shape[1]).to(device)
opt = torch.optim.Adam(model.parameters(), lr=args.lr)
loss_fn = nn.MSELoss()

best_val = 1e9
Path(args.out).mkdir(parents=True, exist_ok=True)
for epoch in range(1, args.epochs+1):
    model.train()
    train_loss = 0
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        pred = model(xb)
        loss = loss_fn(pred, yb)
        opt.zero_grad(); loss.backward(); opt.step()
        train_loss += loss.item() * xb.size(0)
    train_loss /= len(train_loader.dataset)
    # val
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for xb, yb in val_loader:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb)
            val_loss += loss_fn(pred, yb).item() * xb.size(0)
    val_loss /= len(val_loader.dataset)
    print(f'Epoch {epoch} train_loss={train_loss:.2f} val_loss={val_loss:.2f}')
    if val_loss < best_val:
        best_val = val_loss
        torch.save(model.state_dict(), Path(args.out)/'mlp.pth')
        print('Saved best model', Path(args.out)/'mlp.pth')

print('Done. Best val MSE:', best_val)
