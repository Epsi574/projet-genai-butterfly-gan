"""
Dataset for loading butterfly images from CSV.
"""
import pandas as pd
import numpy as np
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


class ButterflyDataset(Dataset):
    """CSV-driven dataset (filename -> label)."""
    def __init__(self, df, img_root, class_to_idx, transform=None):
        self.img_root = Path(img_root)
        self.transform = transform
        self.class_to_idx = class_to_idx
        self.samples = [
            (row['filename'], class_to_idx[row['label']])
            for _, row in df.iterrows()
            if row['label'] in class_to_idx
        ]
        print(f'   Loaded {len(self.samples)} images')

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        filename, label = self.samples[idx]
        img = Image.open(self.img_root / filename).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, label


def get_transforms(img_size=64, mode='train'):
    """Retourne les transformations appropriées."""
    if mode == 'train':
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])


def load_dataset(csv_path, img_root, top_k=None, img_size=64, batch_size=64):
    """
    Load dataset from CSV and create DataLoader.
    
    Args:
        csv_path: Path to CSV (columns: filename, label)
        img_root: Root directory for images
        top_k: Number of species to keep (None = all)
        img_size: Image size
        batch_size: Batch size
        
    Returns:
        dataloader, n_classes, class_to_idx, idx_to_class, df_stats
    """
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip().str.lower()
    
    print(f'CSV loaded: {len(df)} images')
    print(f'Columns: {df.columns.tolist()}')
    
    all_classes = sorted(df['label'].dropna().unique().tolist())
    print(f'{len(all_classes)} species found')
    
    df_stats = (df.groupby('label')
                .size()
                .reset_index(name='nb_images')
                .rename(columns={'label': 'espece'})
                .sort_values('nb_images', ascending=False)
                .reset_index(drop=True))
    
    if top_k:
        selected_classes = df_stats.head(top_k)['espece'].tolist()
        print(f'{top_k} most represented species selected')
    else:
        selected_classes = df_stats['espece'].tolist()
        print(f'All {len(selected_classes)} species selected')
    
    n_classes = len(selected_classes)
    class_to_idx = {cls: i for i, cls in enumerate(selected_classes)}
    idx_to_class = {i: cls for cls, i in class_to_idx.items()}
    
    transform = get_transforms(img_size, mode='train')
    dataset = ButterflyDataset(df, img_root, class_to_idx, transform)
    
    # pin_memory uniquement si GPU disponible
    import torch
    use_pin_memory = torch.cuda.is_available()
    
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=4, 
        pin_memory=use_pin_memory, 
        drop_last=True
    )
    
    print(f'DataLoader ready - {len(dataloader)} batches/epoch')
    print(f'   Total images      : {df_stats.nb_images.sum()}')
    print(f'   Min images/class  : {df_stats.nb_images.min()}')
    print(f'   Max images/class  : {df_stats.nb_images.max()}')
    print(f'   Average           : {df_stats.nb_images.mean():.0f}')
    
    return dataloader, n_classes, class_to_idx, idx_to_class, df_stats
