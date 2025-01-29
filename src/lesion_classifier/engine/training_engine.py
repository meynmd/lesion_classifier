from pathlib import Path
import logging
from turtle import pd
from typing import Callable, Iterable

# import pandas as pd
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
from torch.nn.functional import sigmoid
from torcheval.metrics import (
    Mean, 
    BinaryAccuracy,
    BinaryPrecision,
    BinaryRecall,
    BinaryF1Score,
    BinaryPrecisionRecallCurve,
    BinaryAUPRC
)
from torch.utils.tensorboard import SummaryWriter


class TrainingEngine:
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        loss_func: Callable,
        optimizer: optim.Optimizer,
        scheduler: optim.lr_scheduler.LRScheduler,
        val_loader: DataLoader = None,
        test_loader: DataLoader = None,
        save_dir: Path = None,
        eval_freq: int = 1,
        ckpt_freq: int = 1,
        use_cuda: bool = False
    ):
        logging.basicConfig(level=logging.INFO)

        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.loss_func = loss_func
        self.optim = optimizer
        self.sched = scheduler
        self.epoch = 0
        self.eval_freq = eval_freq
        self.ckpt_freq = ckpt_freq
        self.train_step = 0
        self.use_cuda = use_cuda

        if self.use_cuda:
            self.model = self.model.to('cuda')
            self.device = 'cuda'
        else:
            self.device = 'cpu'

        if save_dir is None:
            assert ckpt_freq is None, \
                'must specify save_dir if checkpoints to be saved'
            self.writer = None
        else:
            self.save_dir = save_dir
            if not save_dir.exists():
                save_dir.mkdir(parents=True)
            log_dir = save_dir / 'logging'
            if not log_dir.exists():
                log_dir.mkdir()            
            self.writer = SummaryWriter(log_dir=log_dir)

        self._initialize_metrics()

    def _initialize_metrics(self):

        self.validation_metrics = {
            ('accuracy', 'scalar'): BinaryAccuracy(device=self.device),
            ('precision', 'scalar'): BinaryPrecision(device=self.device),
            ('recall', 'scalar'): BinaryRecall(device=self.device),
            ('F1', 'scalar'): BinaryF1Score(device=self.device),
            ('AUPRC', 'scalar'): BinaryAUPRC(device=self.device),
        }
        self.evaluation_metrics = {
            ('accuracy', 'scalar'): BinaryAccuracy(device=self.device),
            ('precision', 'scalar'): BinaryPrecision(device=self.device),
            ('recall', 'scalar'): BinaryRecall(device=self.device),
            ('F1', 'scalar'): BinaryF1Score(device=self.device),
            ('AUPRC', 'scalar'): BinaryAUPRC(device=self.device),
            ('PRC', 'vector'): BinaryPrecisionRecallCurve(device=self.device), 
        }

    def train_epoch(
        self
    ):
        self.model.train()
        self.epoch += 1
        train_loss_mean = Mean(device=self.device)
        for batch in tqdm(
            self.train_loader, 
            desc=f'Epoch {self.epoch}'
        ):
            self.train_step += 1
            lrs = [pg['lr'] for pg in self.optim.param_groups]
            for i, lr in enumerate(lrs):
                self.writer.add_scalar(
                    f'Train/lr_group_{i}', 
                    lr, 
                    self.train_step
                )
            x, labels = batch['image'], batch['label']
            if self.use_cuda:
                x = x.to('cuda')
                labels = labels.to('cuda')
            
            loss = self.train_forward_backward(x, labels)
            train_loss_mean.update(loss)
            if self.writer is not None:
                self.writer.add_scalar(
                    f'Train/loss', 
                    loss, 
                    self.train_step
                )
            self.sched.step()

        return train_loss_mean.compute()
            
    def train_forward_backward(
        self,
        inputs,
        labels
    ):
        self.optim.zero_grad()
        loss = self.forward(inputs, labels)
        loss.backward()
        self.optim.step()
        
        return loss
        
    def forward(
        self,
        inputs,
        labels
    ):
        outputs = self.model(inputs)
        loss = self.loss_func(
            outputs.squeeze(1), 
            labels.to(torch.float)
        )
        return loss

    def forward_inference(
        self,
        inputs
    ):
        logits = self.model(inputs)
        probs = sigmoid(logits)
        
        return probs

    def evaluate(
        self,
        mode: str = 'validation'
    ):
        assert mode in ('validation', 'evaluation'), \
            'supported modes are ("validation", "evaluation")'

        self._initialize_metrics()

        if mode == 'validation':
            metrics = self.validation_metrics
            assert self.val_loader is not None, \
                'no validation loader was specified'
            loader = self.val_loader
        else:
            metrics = self.evaluation_metrics
            assert self.test_loader is not None, \
                'no test loader was specified'
            loader = self.test_loader

        self.model.eval()
        loss_mean = Mean(device=self.device)
        
        for batch_idx, batch in enumerate(tqdm(
            loader, 
            desc=f'Evaluating'
        )):
            x, labels = batch['image'], batch['label']
            if self.use_cuda:
                x = x.to('cuda')
                labels = labels.to('cuda')
            
            with torch.no_grad():
                outputs = self.forward_inference(x)
            
            loss = self.loss_func(
                outputs.squeeze(1), 
                labels.to(torch.float)
            )
            loss_mean.update(loss)
            for key in metrics:
                metrics[key].update(
                    outputs.squeeze(1), 
                    labels
                )

        results = {('val_loss_mean', 'scalar'): loss_mean.compute()}

        for (name, met_type), metric in metrics.items():
            results[(name, met_type)] = metric.compute()
        
        return results

    def run_metrics(self, mode: str = 'validation'):
        metrics = self.evaluate(mode=mode)
        for (name, met_type), v in metrics.items():
            if met_type == 'scalar':
                logging.info(f'\t{name} = {v:.3f}')
                if self.writer is not None:
                    self.writer.add_scalar(
                        f'Validation/{name}', 
                        v, 
                        self.train_step
                    )

        return metrics

    def train(
        self,
        n_epochs: int,
        n_train_steps_max: int = None
    ):
        for epoch in range(n_epochs):
            loss_mean = self.train_epoch()
            logging.info(f'mean training loss: {loss_mean:.3f}')

            if (self.eval_freq is not None) and (self.epoch % self.eval_freq == 0):
                self.run_metrics()

            if (self.ckpt_freq is not None) and (self.epoch % self.ckpt_freq == 0):
                filename = self.save_dir / f'checkpoint_epoch_{self.epoch:03d}.pth'
                torch.save(self.model.state_dict(), filename)

            if self.train_step >= n_train_steps_max:
                break

        logging.info('final eval')
        metrics = self.run_metrics(mode='evaluation')

        if self.save_dir is not None:
            filename = self.save_dir / f'final_weights_epoch_{self.epoch:03d}.pth'
            logging.info(f'saving final model state to {filename}')
            torch.save(self.model.state_dict(), filename)
