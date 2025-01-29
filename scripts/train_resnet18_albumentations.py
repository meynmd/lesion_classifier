from pathlib import Path
import datetime
import argparse

import transformers

from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import torch.nn as nn

import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim
from torch.utils.data import Dataset, DataLoader

from data.image_dataset import ImageDataset
from engine.training_engine import TrainingEngine
import models


def get_args():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument(
        '--data-root',
        '-d',
        type=Path,
        required=True   
    )
    arg_parser.add_argument(
        '--output-root', 
        '-o',
        type=Path,
        required=True
    )
    arg_parser.add_argument(
        '--n-epochs', 
        type=int,
        default=10
    )
    arg_parser.add_argument(
        '--init-lr', 
        type=float,
        default=1e-4
    )
    arg_parser.add_argument(
        '--run-name', 
        type=str,
        default='resnet18'
    )
    arg_parser.add_argument(
        '--eval-freq', 
        type=int,
        default=2
    )
    arg_parser.add_argument(
        '--save-freq', 
        type=int,
        default=2
    )
    arg_parser.add_argument(
        '--cuda', 
        action='store_true',
    )
    arg_parser.add_argument(
        '--batch-size',
        type=int,
        default=256
    )
    
    return arg_parser.parse_args()

    
def main():

    # get console args
    args = get_args()
    data_root = args.data_root
    output_root = args.output_root
    n_epochs = args.n_epochs
    run_name = args.run_name
    eval_freq = args.eval_freq
    save_freq = args.save_freq
    init_lr = args.init_lr
    use_cuda = args.cuda
    batch_size = args.batch_size

    # init model and get its preferred image transforms
    model, _ = models.resnet.build_resnet18(n_categories=1)

    transforms_train = A.Compose(
        [
            A.SmallestMaxSize(max_size=256),
            # A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=180, p=0.5),
            A.Affine(
                scale=(0.9, 1.1), 
                translate_percent=(-0.05, 0.05), 
                rotate=(-180., 180.), 
                shear=(-15, 15), 
                interpolation=1, 
                mask_interpolation=0, 
                fit_output=False, 
                keep_ratio=True, 
                p=0.33
            ),
            A.RandomCrop(height=224, width=224),
            A.RGBShift(r_shift_limit=16, g_shift_limit=16, b_shift_limit=16, p=0.1),
            A.RandomBrightnessContrast(
                brightness_limit=(-0.1, 0.1), 
                contrast_limit=(-0.1, 0.1),
                ensure_safe_range=True,
                p=0.1
            ),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ]
    )
    transforms_eval = A.Compose(
        [
            A.SmallestMaxSize(max_size=256),
            A.CenterCrop(height=224, width=224),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ]
    )

    # init datasets and loaders
    train_set = ImageDataset(
        data_root / 'train.csv', 
        transform=transforms_train
    )
    val_set = ImageDataset(
        data_root / 'validation.csv',
        transform=transforms_eval
    )
    test_set = ImageDataset(
        data_root / 'test.csv',
        transform=transforms_eval
    )

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=64, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=64, shuffle=False)

    # determine how many steps to train for and warmup period
    epoch_length = len(train_loader)
    n_train_steps = n_epochs * epoch_length
    n_warmup_steps = 0.1 * n_train_steps

    # set up optimizer, scheduler and loss criterion
    optimizer = optim.Adam(model.parameters(), lr=init_lr)
    scheduler = transformers.get_scheduler(
        'cosine', 
        optimizer, 
        num_warmup_steps=n_warmup_steps, 
        num_training_steps=n_train_steps
    )
    criterion = nn.BCEWithLogitsLoss()

    # where to output parameters and logging
    save_dir = output_root / f'{run_name}'
    time_now = datetime.datetime.now()
    save_subdir = save_dir / time_now.strftime(f"%Y_%h_%d_%H_%M")

    engine = TrainingEngine(
        model=model,
        train_loader=train_loader,
        loss_func=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        val_loader=val_loader,
        test_loader=test_loader,
        save_dir=save_subdir,
        use_cuda=use_cuda,
        eval_freq=eval_freq,
        ckpt_freq=save_freq
    )

    engine.train(n_epochs, n_train_steps)


if __name__ == '__main__':
    main()
