import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import pandas as pd
import numpy as np

class BreakHisDataset(Dataset):
    def __init__(self, root_dir, csv_file=None, transform=None):
        self.root = root_dir
        self.transform = transform or transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
        ])
        self.samples = []
        if csv_file and os.path.exists(csv_file):
            df = pd.read_csv(csv_file)
            for _, r in df.iterrows():
                self.samples.append((r['path'], int(r['label'])))
        else:
            # expects root_dir/benign and root_dir/malignant or similar labels
            for label, cls in enumerate(['benign','malignant']):
                folder = os.path.join(root_dir, cls)
                if not os.path.exists(folder):
                    continue
                for fn in os.listdir(folder):
                    if fn.lower().endswith(('.png','.jpg','.jpeg')):
                        self.samples.append((os.path.join(folder,fn), label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert('RGB')
        img = self.transform(img)
        return img, label

class WDBC_Dataset(torch.utils.data.Dataset):
    def __init__(self, csv_file, features=None, target='target'):
        df = pd.read_csv(csv_file)
        if features is None:
            features = [c for c in df.columns if c not in ['id', target]]
        self.X = df[features].values.astype(np.float32)
        self.y = df[target].apply(lambda x: 1 if x in ['M',1,'malignant'] else 0).values.astype(np.int64)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return torch.tensor(self.X[idx]), int(self.y[idx])
