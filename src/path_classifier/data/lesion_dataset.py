from pathlib import Path

# import pandas as pd
from tqdm import tqdm
from PIL import Image
import numpy as np
import albumentations as A

from torch.utils.data import Dataset
from torchvision.io import read_image


class LesionImageDataset(Dataset):
    
    positive_label = 'Malignant'
    negative_label = 'Benign'
    accepted_exts = ['jpg', 'JPG', 'png', 'PNG']
    
    def __init__(
        self, 
        data_root: Path, 
        transform=None, 
    ):
        self.data_list = self.enumerate_dataset(data_root)
        self.transform = transform
        
    def __len__(self):
        return len(self.data_list)

    def __getitem__(
        self, 
        idx: int,
    ):
        img_path, label = self.data_list[idx]

        # keep either Albumentations or torchvision happy
        if isinstance(self.transform, A.core.composition.Compose):
            img = np.array(Image.open(img_path))
        else:
            img = read_image(str(img_path))
        
        if self.transform is not None:
            if isinstance(self.transform, A.core.composition.Compose):
                img = self.transform(image=img)['image']
            else:
                img = self.transform(img)
            
        return {
            'image': img, 
            'label': label,
            'filename': str(img_path)
        }
        

    def enumerate_dataset(
        self, 
        root_dir: Path,
    ):
        pos_ex_dir = root_dir / LesionImageDataset.positive_label
        neg_ex_dir = root_dir / LesionImageDataset.negative_label
        
        data_list = []
        for label, subdir in [(1, pos_ex_dir), (0, neg_ex_dir)]:
            for ext in LesionImageDataset.accepted_exts:
                for filename in subdir.glob(f'*.{ext}'):
                    data_list.append((filename, label))
        
        return data_list
        
        