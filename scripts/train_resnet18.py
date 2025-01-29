from pathlib import Path
import datetime
import argparse

import transformers

from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import torch.nn as nn

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

    # init model and get its preferred image transforms
    model, model_transforms = models.resnet.build_resnet18(n_categories=1)

    # init datasets and loaders
    train_set = ImageDataset(
        data_root / 'train.csv', 
        transform=model_transforms
    )
    val_set = ImageDataset(
        data_root / 'validation.csv',
        transform=model_transforms
    )
    test_set = ImageDataset(
        data_root / 'test.csv',
        transform=model_transforms
    )

    train_loader = DataLoader(train_set, batch_size=256, shuffle=True)
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
        use_cuda=True,
        eval_freq=eval_freq,
        ckpt_freq=save_freq
    )

    engine.train(n_epochs, n_train_steps)


if __name__ == '__main__':
    main()
