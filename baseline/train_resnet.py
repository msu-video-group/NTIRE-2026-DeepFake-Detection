from torch.utils.data import Dataset, DataLoader
import os
import pandas as pd
from typing import List, Dict, Callable, Tuple
from PIL import Image
import numpy as np
import torch
import torch.nn as nn
from torchvision.models import resnet50
from torchvision import transforms
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.utilities.rank_zero import rank_zero_only
from torchmetrics.classification import BinaryAUROC, Accuracy
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pathlib import Path
from aug_utils_train import distort_images
import csv

class AIGenDetDataset(Dataset):
    def __init__(self, root_dir, subset_dirs, transform=None):
        """
        Initialize the dataset from one or more subset directories.

        Args:
            root_dir (str): Root path containing subset directories.
            subset_dirs (List[str]): Names of subset directories to include
                (each must contain a `labels.csv` and `images/` folder).
            transform (Callable, optional): Optional augmentation/transform
                applied to the tensor image (expects a callable returning a
                tuple where the first element is the transformed image).
        """
        self.convert_to_tensor = transforms.Compose([
            transforms.Resize(1024),
            transforms.CenterCrop(1024),
            transforms.ToTensor()
        ])
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.transform = transform
        
        self.root_dir = root_dir
        self.subset_dirs = [os.path.join(root_dir, x) for x in subset_dirs]
        self.subset_dirs = [x for x in self.subset_dirs if os.path.isdir(x)]
        
        label_dfs = [pd.read_csv(os.path.join(x, 'labels.csv'), index_col=0) for x in self.subset_dirs]
        self.label_df = pd.DataFrame(columns=['image_name', 'label', 'subset_dir'])
        for idx, ldf in enumerate(label_dfs):
            ldf['subset_dir'] = Path(self.subset_dirs[idx]).name
            self.label_df = pd.concat([self.label_df, ldf], ignore_index=True)
        print(f'Found {len(self.subset_dirs)} directories, {len(self.label_df)} images in total.')

    def __len__(self):
        return len(self.label_df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        label = self.label_df.loc[idx, 'label']
        image_name = self.label_df.loc[idx, 'image_name']

        if image_name.split(".")[-1] not in {"jpg", "jpeg", "png"}:
            image_name += ".jpg"
        
        img_path = os.path.join(self.root_dir, self.label_df.loc[idx, 'subset_dir'], 'images', image_name)
        
        image = Image.open(img_path)
        image = self.convert_to_tensor(image)
        if self.transform is not None:
            image = self.transform(image)[0]
        image = self.normalize(image)
        
        sample = {'image': image, 'label': label, "image_name": image_name}
        return sample

    @staticmethod
    def read_from_shards(shard_dir, shard_nums=None, transform=None):
        """
        Convenience constructor that builds the dataset from shard folders.

        Args:
            shard_dir (str): Root directory containing shard folders
                named like `shard_0`, `shard_1`, etc.
            shard_nums (List[int], optional): Specific shard indices to use.
                If None, defaults to shards 0–5.
            transform (Callable, optional): Optional transform to apply.

        Returns:
            AIGenDetDataset: Dataset instance covering the selected shards.
        """
        if shard_nums is None:
            shard_dirs = [f'shard_{i}' for i in range(0,6)]
        else:
            shard_dirs = [f'shard_{i}' for i in shard_nums]
        return AIGenDetDataset(root_dir=shard_dir, subset_dirs=shard_dirs, transform=transform)

def collate(batch):
    images = torch.stack([item['image'] for item in batch])
    labels = torch.Tensor([item['label'] for item in batch])
    image_names = [item['image_name'] for item in batch]
    
    return {'image': images, 'label': labels, 'image_name': image_names}


def make_dataloaders(shard_dir, val_dataset_dir, batch_size=32, num_workers=16):
    """
    Create train/val/test DataLoaders for the AI-generated image detection task.

    Args:
        shard_dir (str): Path to training shards directory.
        val_dataset_dir (str): Path to validation/test dataset directory.
        batch_size (int, optional): Batch size for all loaders.
        num_workers (int, optional): Number of DataLoader workers.

    Returns:
        Tuple[DataLoader, DataLoader, DataLoader, DataLoader]:
            (train_dataloader, val_dataloader, test_clear, test_distorted)
    """
    train_dataset = AIGenDetDataset.read_from_shards(shard_dir, shard_nums=[1], transform=distort_images)
    val_dataset = AIGenDetDataset.read_from_shards(shard_dir, shard_nums=[5])
    test_clear = AIGenDetDataset(val_dataset_dir, subset_dirs=["clear"])
    test_distorted = AIGenDetDataset(val_dataset_dir, subset_dirs=["distorted"])

    common_params = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "prefetch_factor": 16,
        "collate_fn": collate,
        "pin_memory": True,
        "persistent_workers": True
    }

    train_dataloader = DataLoader(dataset=train_dataset, shuffle=True, drop_last=True, **common_params)
    val_dataloader = DataLoader(dataset=val_dataset, shuffle=False, drop_last=False, **common_params)
    test_clear = DataLoader(dataset=test_clear, shuffle=False, drop_last=False, **common_params)
    test_distorted = DataLoader(dataset=test_distorted, shuffle=False, drop_last=False, **common_params)

    return train_dataloader, val_dataloader, test_clear, test_distorted


class BaselineDetector(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = resnet50(weights=None)
        self.lin_1 = nn.Linear(self.backbone.fc.out_features, 128)
        self.relu = nn.ReLU()
        self.lin_2 = nn.Linear(128, 2)
        
    def forward(self, x):
        x = self.backbone(x)
        x = self.lin_1(x)
        x = self.relu(x)
        logits = self.lin_2(x)
        
        return logits


class TrainingModule(pl.LightningModule):
    def __init__(
        self,
        model: nn.Module,
        class_weights = None,
        lr = 1e-3,
        min_lr = 1e-6,
        submission_file: str = None
    ):
        """
        LightningModule wrapper for training/evaluation.

        Args:
            model (nn.Module): Model to train.
            class_weights (List[float] or None): Class weights for the
                CrossEntropy loss.
            lr (float): Initial learning rate.
            min_lr (float): Minimum learning rate for scheduler.
            submission_file (str, optional): Path to CSV for writing test
                predictions (image_name, score).
        """
        super().__init__()
        self.model = model
        self.loss_fn = nn.CrossEntropyLoss(weight=torch.Tensor(class_weights))
        self.monitor_key = "val_loss"

        self.lr = lr
        self.min_lr = min_lr

        self.rocauc = BinaryAUROC()
        self.accuracy = Accuracy(task="binary")

        self.submission_file = submission_file
        self.init_submissions()
        
    
    @rank_zero_only
    def init_submissions(self):
        if self.submission_file is not None:
            self.submission_file = Path(self.submission_file)
            self.fieldnames = ["image_name", "score"]
            with self.submission_file.open("a", newline="", encoding="utf-8") as csv_file:
                writer = csv.DictWriter(csv_file, fieldnames=self.fieldnames)
                writer.writeheader()
        

    def _shared_step(self, batch, stage: str):
        x = batch["image"]
        y = batch["label"].long()
        image_names = batch["image_name"]
        logits = self.model(x)
        total_loss = self.loss_fn(logits, y)
        acc = self.accuracy(logits.argmax(dim=-1), y)
        
        self.log(
            f"{stage}_loss",
            total_loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True
        )
        self.log(
            "lr",
            self.optimizer.param_groups[0]['lr'],
            on_step=False,
            on_epoch=True,
            prog_bar=True
        )
        self.log(
            f"{stage}_acc",
            acc,
            on_step=False,
            on_epoch=True,
            prog_bar=True
        )

        if stage == "train":
            return total_loss
        else:
            predictions = F.softmax(logits, dim=-1)[:, 1]
            self.rocauc.update(predictions, y)
            if stage == "test":
                for image_name, score in zip(image_names, predictions):
                    self.test_results.append({"image_name": image_name, "score": score.item()})

            return acc

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, stage="train")

    def validation_step(self, batch, batch_idx):
        return self._shared_step(batch, stage="val")

    def test_step(self, batch, batch_idx):
       return self._shared_step(batch, stage="test")

    def _log_rocauc(self):
        score = self.rocauc.compute()
        self.log(
            "rocauc",
            score,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True
        )
        
    def on_validation_epoch_start(self):
        self.rocauc.reset()

    def on_test_epoch_start(self):
        self.rocauc.reset()
        self.test_results = []

    def on_test_epoch_end(self):
        self._log_rocauc()
        if self.submission_file is not None:
            with self.submission_file.open("a", newline="", encoding="utf-8") as csv_file:
                writer = csv.DictWriter(csv_file, fieldnames=self.fieldnames)
                writer.writerows(self.test_results)

    def on_validation_epoch_end(self):
        self._log_rocauc()
    
    def on_after_backward(self):
        total_norm = 0.0
        for p in self.model.parameters():
            if p.grad is not None:
                param_norm = p.grad.detach().norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5
        self.log("grad", total_norm, prog_bar=True, on_step=True, on_epoch=False)

    def configure_optimizers(self):
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, patience=5, factor=0.5, min_lr=self.min_lr)
        
        return {
            "optimizer": self.optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": self.monitor_key,
                "interval": "epoch",
                "frequency": 1,
            },
        }
        
    def configure_callbacks(self):
        callbacks = [
            ModelCheckpoint(
                filename="{epoch:02d}-{" + self.monitor_key + ":.4f}",
                monitor=self.monitor_key,
                mode="min",
                save_top_k=3,
                save_last=True,
                verbose=True,
                enable_version_counter=False,
            ),
            EarlyStopping(
                monitor=self.monitor_key, 
                mode="min", 
                patience=10, 
                verbose=True
            )
        ]
        return callbacks


def load_from_chekpoint(training_module, ckpt_path):
    """
    Load a LightningModule state from a checkpoint file.

    Args:
        training_module (pl.LightningModule): Module instance to load into.
        ckpt_path (str): Path to checkpoint file.

    Returns:
        pl.LightningModule: The module with loaded state.
    """
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    state_dict = checkpoint.get("state_dict")
    training_module.load_state_dict(state_dict, strict=True)
    return training_module


def main():
    os.environ['MASTER_PORT'] = '29500'

    gpu_ids = [0]
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, gpu_ids))
    
    checkpoint_dir = "./checkpoints/resnet"
    shard_dir = "/root/users/deepfake_bench/data/pixelprose/NTIRE_train/public_train_shards"
    val_dataset_dir = "/root/users/deepfake_bench/data/pixelprose/NTIRE_val"
    submission_file = "./submission.csv"
    
    train_dataloader, val_dataloader, test_clear, test_distorted = \
        make_dataloaders(shard_dir=shard_dir, val_dataset_dir=val_dataset_dir, batch_size=16, num_workers=16)
    model = BaselineDetector()

    training_module = TrainingModule(
        model = model,
        class_weights = [1.7, 1],
        lr = 1e-3,
        min_lr = 1e-6,
        submission_file = submission_file
    )

    trainer = pl.Trainer(
        accelerator="gpu",
        devices=len(gpu_ids),
        strategy="ddp_spawn" if len(gpu_ids) > 1 else "auto",
        min_epochs=50,
        max_epochs=96,
        logger=False,
        gradient_clip_val=5.0,
    )
    
    ckpt_path = os.path.join(checkpoint_dir, "epoch=59-val_loss=0.1979.ckpt")
    if not os.path.isfile(ckpt_path):
        ckpt_path = None

    trainer.fit(
        training_module,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
        ckpt_path=ckpt_path,
    )

    rocauc_clear = trainer.test(training_module, test_clear)[0]["rocauc"]
    rocauc_distorted = trainer.test(training_module, test_distorted)[0]["rocauc"]
    print(f"rocauc on clear {rocauc_clear}")
    print(f"rocauc on distorted {rocauc_distorted}")

    
if __name__ == "__main__":
    main()
