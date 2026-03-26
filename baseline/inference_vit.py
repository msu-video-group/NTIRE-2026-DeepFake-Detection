import os
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
import timm
from torchvision import transforms
import torch.nn.functional as F
from pathlib import Path
from tqdm.auto import tqdm


class BaselineDetector(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = timm.create_model("vit_medium_patch16_gap_384.sw_in12k_ft_in1k", pretrained=False)
        
        self.backbone.head_drop = nn.Dropout(p=0.2)
        
        in_features = self.backbone.head.in_features
        self.backbone.head = nn.Linear(in_features, 128)
        self.relu = nn.ReLU()
        self.lin = nn.Linear(128, 2)
        
    def forward(self, x):
        x = self.backbone(x)
        x = self.relu(x)
        logits = self.lin(x)
        
        return logits

def create_transform():
    return transforms.Compose([
        transforms.Resize(384),
        transforms.CenterCrop(384),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


def load_from_chekpoint(model, ckpt_path):
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    state_dict = checkpoint.get("state_dict")
    model.load_state_dict(state_dict, strict=True)
    return model


def main():
    checkpoint_path = "./checkpoints/vit_best_val_loss=0.4451.pt"
    val_dataset_dir = Path("/root/users/deepfake_bench/data/pixelprose/NTIRE_val/v3/validation_set/val_images")
    submission_file = "./submission.csv"
    
    model = BaselineDetector()
    model = load_from_chekpoint(model, checkpoint_path)

    model.eval()

    transform = create_transform()

    image_names = []
    scores = []

    for file in tqdm(os.listdir(val_dataset_dir)):
        img = Image.open(val_dataset_dir / file).convert("RGB")
        img = transform(img)
        logits = model(img.unsqueeze(0))
        score = F.softmax(logits, dim=-1)[:, 1]

        image_names.append(file)
        scores.append(score.item())

    df = pd.DataFrame({"image_name": image_names, "score": scores})
    df.to_csv(submission_file)


if __name__ == "__main__":
    main()
