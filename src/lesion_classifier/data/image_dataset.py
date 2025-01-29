from pathlib import Path

# import pandas as pd
from tqdm import tqdm
from PIL import Image
import numpy as np
import pandas as pd
import albumentations as A

from torch.utils.data import Dataset
from torchvision.io import read_image


class ImageDataset(Dataset):
    
    def __init__(
        self, 
        metadata_path: Path, 
        transform=None, 
    ):
        self.data_list = pd.read_csv(metadata_path)
        self.transform = transform
        
    def __len__(self):
        return len(self.data_list)

    def __getitem__(
        self, 
        idx: int,
    ):
        row = self.data_list.iloc[idx]
        img_path, label = row['filename'], row['label']

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
        