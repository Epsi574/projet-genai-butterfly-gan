"""
Evaluation metrics: FID, Intra-class FID, Precision & Recall.
"""
import torch
import torch.nn as nn
import numpy as np
import shutil
from pathlib import Path
from PIL import Image
from tqdm.auto import tqdm
from torchvision import transforms
from torchvision.transforms.functional import to_pil_image
import torchvision.models as tv_models
from cleanfid import fid as cleanfid
from sklearn.neighbors import NearestNeighbors


def generate_images_for_fid(G, arch_name, n_classes, idx_to_class, latent_dim,
                            device, n_per_class=500, batch_size=64, output_dir='./fid_gen'):
    """
    Generate n_per_class images per species and save them.
    Structure: output_dir/arch_name/class_name/img.png
    """
    out_dir = Path(output_dir) / arch_name
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)

    G.eval()
    with torch.no_grad():
        for cls_idx in tqdm(range(n_classes), desc=f'{arch_name} - generation'):
            cls_dir = out_dir / idx_to_class[cls_idx]
            cls_dir.mkdir()
            generated = 0
            while generated < n_per_class:
                bs = min(batch_size, n_per_class - generated)
                z = torch.randn(bs, latent_dim, device=device)
                labels = torch.full((bs,), cls_idx, dtype=torch.long, device=device)
                imgs = G(z, labels).cpu()
                imgs = (imgs * 0.5 + 0.5).clamp(0, 1)
                for i, img in enumerate(imgs):
                    to_pil_image(img).save(cls_dir / f'{generated+i:05d}.png')
                generated += bs
    G.train()
    print(f'{arch_name}: {n_classes * n_per_class} images generated -> {out_dir}')
    return str(out_dir)


def save_real_images_for_fid(dataset, idx_to_class, n_per_class=500, output_dir='./fid_real'):
    """Save real images (same format as generated ones)."""
    out_dir = Path(output_dir)
    if out_dir.exists():
        return str(out_dir)
    out_dir.mkdir(parents=True)
    
    n_classes = len(idx_to_class)
    by_class = {i: [] for i in range(n_classes)}
    
    for img, label in tqdm(dataset, desc='Saving real images'):
        if len(by_class[label]) < n_per_class:
            by_class[label].append(img)
    
    for cls_idx, imgs in by_class.items():
        cls_dir = out_dir / idx_to_class[cls_idx]
        cls_dir.mkdir(exist_ok=True)
        for i, img in enumerate(imgs):
            pil = to_pil_image((img * 0.5 + 0.5).clamp(0, 1))
            pil.save(cls_dir / f'{i:05d}.png')
    
    print(f'Real images saved -> {out_dir}')
    return str(out_dir)


def compute_global_fid(real_dir, gen_dir, arch_name):
    """Compute global FID between all real and generated images."""
    import sys
    print(f'{arch_name}: images generated')
    print(f'Computing global FID - {arch_name}...')
    sys.stdout.flush()
    
    flat_real = f'./fid_flat/real'
    flat_gen = f'./fid_flat/{arch_name}'
    Path(flat_real).mkdir(parents=True, exist_ok=True)
    Path(flat_gen).mkdir(parents=True, exist_ok=True)
    
    print(f'   Copying files...')
    sys.stdout.flush()
    for p in Path(real_dir).glob('*/*.png'):
        dst = Path(flat_real) / f'{p.parent.name}_{p.name}'
        if not dst.exists():
            shutil.copy(p, dst)
    for p in Path(gen_dir).glob('*/*.png'):
        shutil.copy(p, Path(flat_gen) / f'{p.parent.name}_{p.name}')
    
    print(f'   Computing FID with InceptionV3...')
    sys.stdout.flush()
    score = cleanfid.compute_fid(flat_real, flat_gen, mode='clean',
                                num_workers=0, batch_size=32, verbose=True, device='cpu')
    print(f'Global FID {arch_name}: {score:.2f}')
    sys.stdout.flush()
    return score


def compute_intraclass_fid(real_dir, gen_dir, arch_name, selected_classes):
    """Compute FID for each species separately."""
    per_class_fid = {}
    for cls_name in tqdm(selected_classes[:min(len(selected_classes), 10)],
                        desc=f'Intra-class FID - {arch_name}'):
        r = str(Path(real_dir) / cls_name)
        g = str(Path(gen_dir) / cls_name)
        if not Path(r).exists() or not Path(g).exists():
            continue
        n_real = len(list(Path(r).glob('*.png')))
        n_gen = len(list(Path(g).glob('*.png')))
        if n_real < 50 or n_gen < 50:
            continue
        try:
            score = cleanfid.compute_fid(r, g, mode='clean',
                                        num_workers=0, batch_size=32, verbose=True,
                                        device='cpu')
            per_class_fid[cls_name] = score
        except Exception as e:
            print(f'   Warning {cls_name}: {e}')
    avg = np.mean(list(per_class_fid.values())) if per_class_fid else float('nan')
    print(f'   Average Intra-class FID ({arch_name}): {avg:.2f}')
    return per_class_fid, avg


def get_inception_features(img_dir, device, batch_size=64, max_imgs=2000):
    """Extrait les features InceptionV3 pour un dossier d'images."""
    import torch
    device = torch.device('cpu')  # Forcer CPU
    inception = tv_models.inception_v3(pretrained=True, transform_input=False)
    inception.fc = nn.Identity()
    inception.eval().to(device)

    transform = transforms.Compose([
        transforms.Resize(299),
        transforms.CenterCrop(299),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])

    paths = np.array(list(Path(img_dir).glob('**/*.png')))
    paths = paths[np.random.permutation(len(paths))[:max_imgs]]
    print(f'   Extracting features from {len(paths)} images...')
    features = []
    
    for i in tqdm(range(0, len(paths), batch_size), desc='   Features Inception'):
        batch_paths = paths[i:i+batch_size]
        imgs = torch.stack([transform(Image.open(p).convert('RGB'))
                           for p in batch_paths]).to(device)
        with torch.no_grad():
            feat = inception(imgs)
        features.append(feat.cpu().numpy())
    
    return np.concatenate(features, axis=0)


def precision_recall(real_features, fake_features, k=3):
    """
    Compute Precision & Recall (Kynkäänniemi et al., 2019).
    Precision = proportion of generated images in real distribution.
    Recall = proportion of real images covered by generated ones.
    """
    # Normalisation L2
    real = real_features / (np.linalg.norm(real_features, axis=1, keepdims=True) + 1e-8)
    fake = fake_features / (np.linalg.norm(fake_features, axis=1, keepdims=True) + 1e-8)

    knn_real = NearestNeighbors(n_neighbors=k+1).fit(real)
    knn_fake = NearestNeighbors(n_neighbors=k+1).fit(fake)

    # Precision
    dist_real, _ = knn_real.kneighbors(real)
    rad_real = dist_real[:, -1]
    dist_fake_to_real, idx = knn_real.kneighbors(fake)
    precision = (dist_fake_to_real[:, 0] <= rad_real[idx[:, 0]]).mean()

    # Recall
    dist_fake, _ = knn_fake.kneighbors(fake)
    rad_fake = dist_fake[:, -1]
    dist_real_to_fake, idx = knn_fake.kneighbors(real)
    recall = (dist_real_to_fake[:, 0] <= rad_fake[idx[:, 0]]).mean()

    return float(precision), float(recall)


def evaluate_all(G, arch_name, real_dir, n_classes, idx_to_class, latent_dim, 
                device, selected_classes, n_per_class=300):
    """
    Complete evaluation: FID, Intra-class FID, Precision & Recall.
    
    Returns:
        dict with all metrics
    """
    # Génération des images
    gen_dir = generate_images_for_fid(G, arch_name, n_classes, idx_to_class,
                                     latent_dim, device, n_per_class=n_per_class)
    
    # FID global
    fid_global = compute_global_fid(real_dir, gen_dir, arch_name)
    
    # Intra-class FID
    ic_fid_dict, ic_fid_avg = compute_intraclass_fid(real_dir, gen_dir, 
                                                     arch_name, selected_classes)
    
    # Precision & Recall
    print(f'Extraction features InceptionV3 - {arch_name}...')
    feats_real = get_inception_features(real_dir, device)
    feats_gen = get_inception_features(gen_dir, device)
    precision, recall = precision_recall(feats_real, feats_gen)
    
    print(f'{arch_name} - Precision: {precision:.3f}, Recall: {recall:.3f}')
    
    return {
        'fid': fid_global,
        'ic_fid': ic_fid_avg,
        'ic_fid_per_class': ic_fid_dict,
        'precision': precision,
        'recall': recall,
        'gen_dir': gen_dir
    }
