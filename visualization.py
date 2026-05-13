"""
Fonctions de visualisation et génération d'images.
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
from torchvision import utils


def generate_fixed_grid(G, arch_name, epoch, n_classes, idx_to_class, latent_dim,
                       device, output_dir, n_per_class=4):
    """Génère une grille d'images fixes pour suivre la progression."""
    G.eval()
    with torch.no_grad():
        all_imgs = []
        for cls_idx in range(min(n_classes, 8)):
            z = torch.randn(n_per_class, latent_dim, device=device)
            labels = torch.full((n_per_class,), cls_idx, dtype=torch.long, device=device)
            imgs = G(z, labels).cpu()
            all_imgs.append(imgs)
        grid_imgs = torch.cat(all_imgs, dim=0)
        grid = utils.make_grid(grid_imgs * 0.5 + 0.5, nrow=n_per_class, padding=2)
        
        fig, ax = plt.subplots(figsize=(10, min(n_classes, 8) * 1.4))
        ax.imshow(grid.permute(1, 2, 0).numpy())
        ax.set_title(f'{arch_name} - Epoch {epoch}', fontsize=11)
        ax.axis('off')
        
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        path = f'{output_dir}/{arch_name}_epoch{epoch:04d}.png'
        plt.savefig(path, dpi=120, bbox_inches='tight')
        plt.close()
    G.train()


def plot_losses(history, arch_name, output_dir):
    """Affiche les courbes de loss G et D."""
    epochs = [h['epoch'] for h in history]
    g_loss = [h['g_loss'] for h in history]
    d_loss = [h['d_loss'] for h in history]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 3))
    ax1.plot(epochs, g_loss, '#7F77DD', lw=1.5)
    ax1.set_title(f'{arch_name} - Generator Loss')
    ax1.set_xlabel('Epoch')
    ax1.spines[['top', 'right']].set_visible(False)
    
    ax2.plot(epochs, d_loss, '#D85A30', lw=1.5)
    ax2.set_title(f'{arch_name} - Loss Discriminateur')
    ax2.set_xlabel('Epoch')
    ax2.spines[['top', 'right']].set_visible(False)
    
    plt.tight_layout()
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    plt.savefig(f'{output_dir}/{arch_name}_losses.png', dpi=120, bbox_inches='tight')
    plt.close()


def plot_dataset_distribution(df_stats, output_dir, top_n=30):
    """Visualise la distribution des images par espèce."""
    fig, ax = plt.subplots(figsize=(14, 4))
    top = df_stats.head(top_n)
    bars = ax.bar(top['espece'], top['nb_images'], color='#7F77DD', alpha=0.85, width=0.7)
    ax.axhline(df_stats['nb_images'].mean(), color='#D85A30', ls='--', lw=1.2,
              label=f'Moyenne ({df_stats["nb_images"].mean():.0f})')
    ax.set_title(f'Distribution des images par espèce (top {top_n})', fontsize=12, pad=10)
    ax.set_ylabel('Nombre d\'images')
    ax.tick_params(axis='x', rotation=75, labelsize=7)
    ax.spines[['top', 'right']].set_visible(False)
    ax.legend()
    plt.tight_layout()
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    plt.savefig(f'{output_dir}/dataset_distribution.png', dpi=150, bbox_inches='tight')
    plt.close()


def comparative_grid(real_dir, gen_dirs, arch_names, species_list, n_imgs=4, output_dir='./results'):
    """
    Grille comparative : Réel vs 3 Architectures.
    Pour chaque espèce: ligne réelle + lignes générées.
    """
    n_species = len(species_list)
    n_rows = 1 + len(arch_names)

    fig, axes = plt.subplots(
        n_species * n_rows, n_imgs,
        figsize=(n_imgs * 2.0 + 1.5, n_species * n_rows * 2.0),
        gridspec_kw={'hspace': 0.05, 'wspace': 0.05}
    )

    for si, species in enumerate(species_list):
        row_base = si * n_rows

        # Ligne 0: images réelles
        real_paths = list(Path(real_dir, species).glob('*.png'))[:n_imgs]
        for j, p in enumerate(real_paths):
            ax = axes[row_base, j]
            ax.imshow(Image.open(p))
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            if j == 0:
                ax.set_ylabel(
                    f'{species.replace("_", " ")}\nRéel',
                    fontsize=9, rotation=0, ha='right', va='center',
                    labelpad=10, fontweight='bold'
                )

        # Lignes 1..N: images générées
        for ai, (gen_dir, aname) in enumerate(zip(gen_dirs, arch_names)):
            gen_paths = list(Path(gen_dir, species).glob('*.png'))[:n_imgs]
            for j, p in enumerate(gen_paths):
                ax = axes[row_base + 1 + ai, j]
                ax.imshow(Image.open(p))
                ax.set_xticks([])
                ax.set_yticks([])
                for spine in ax.spines.values():
                    spine.set_visible(False)
                if j == 0:
                    ax.set_ylabel(
                        aname, fontsize=9, rotation=0,
                        ha='right', va='center', labelpad=10
                    )

    plt.suptitle('Comparaison qualitative : Réel vs Généré (par espèce)',
                fontsize=12, y=0.995)
    plt.tight_layout()
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    out_path = f'{output_dir}/comparative_grid.png'
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Grille sauvegardée -> {out_path}')


def plot_intraclass_fid(ic_fid_dcgan, ic_fid_wgan, ic_fid_sngan, output_dir):
    """Visualise l'Intra-class FID par espèce."""
    common_classes = sorted(
        set(ic_fid_dcgan.keys()) & set(ic_fid_wgan.keys()) & set(ic_fid_sngan.keys())
    )
    if not common_classes:
        print('Warning: Not enough common species for visualization')
        return
    
    x = np.arange(len(common_classes))
    w = 0.25
    fig, ax = plt.subplots(figsize=(max(10, len(common_classes)*1.2), 4))
    ax.bar(x - w, [ic_fid_dcgan[c] for c in common_classes],
          w, label='CDCGAN', color='#7F77DD', alpha=0.85)
    ax.bar(x, [ic_fid_wgan[c] for c in common_classes],
          w, label='CWGAN-GP', color='#1D9E75', alpha=0.85)
    ax.bar(x + w, [ic_fid_sngan[c] for c in common_classes],
          w, label='CSNGAN', color='#D85A30', alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels([c.replace('_', ' ')[:18] for c in common_classes],
                       rotation=40, ha='right', fontsize=8)
    ax.set_ylabel('Intra-class FID')
    ax.set_title('Intra-class FID par espèce')
    ax.legend()
    ax.spines[['top', 'right']].set_visible(False)
    plt.tight_layout()
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    plt.savefig(f'{output_dir}/intraclass_fid.png', dpi=150, bbox_inches='tight')
    plt.close()


def plot_comparative_losses(hist_dcgan, hist_wgan, hist_sngan, output_dir):
    """Courbes de loss comparatives."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    configs = [
        ('CDCGAN', hist_dcgan, '#7F77DD'),
        ('CWGAN-GP', hist_wgan, '#1D9E75'),
        ('CSNGAN', hist_sngan, '#D85A30'),
    ]
    for name, hist, color in configs:
        if hist:
            epochs = [h['epoch'] for h in hist]
            g_loss = [h['g_loss'] for h in hist]
            d_loss = [h['d_loss'] for h in hist]
            axes[0].plot(epochs, g_loss, color=color, lw=1.5, label=name)
            axes[1].plot(epochs, d_loss, color=color, lw=1.5, label=name)

    axes[0].set_title('Loss Générateur (toutes architectures)')
    axes[0].set_xlabel('Epoch')
    axes[0].legend()
    axes[0].spines[['top', 'right']].set_visible(False)
    
    axes[1].set_title('Loss Discriminateur / Critique')
    axes[1].set_xlabel('Epoch')
    axes[1].legend()
    axes[1].spines[['top', 'right']].set_visible(False)
    
    plt.tight_layout()
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    plt.savefig(f'{output_dir}/comparative_losses.png', dpi=150, bbox_inches='tight')
    plt.close()


def radar_chart(df_results, output_dir):
    """Radar chart normalisé pour comparer les 4 métriques."""
    metrics = ['FID Global', 'IC-FID', 'Precision ', 'Recall']
    arches = df_results['Architecture'].tolist()
    colors = ['#7F77DD', '#1D9E75', '#D85A30']

    # Normalisation
    raw = np.array([
        df_results['FID Global :'].values,
        df_results['IC-FID :'].values,
        df_results['Precision '].values,
        df_results['Recall'].values,
    ], dtype=float)
    normed = np.zeros_like(raw)
    for i, ascending in enumerate([True, True, False, False]):
        mi, ma = raw[i].min(), raw[i].max()
        if ma == mi:
            normed[i] = 0.5
        elif ascending:
            normed[i] = 1 - (raw[i] - mi) / (ma - mi)
        else:
            normed[i] = (raw[i] - mi) / (ma - mi)

    N = len(metrics)
    angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
    for ai, (arch, color) in enumerate(zip(arches, colors)):
        values = normed[:, ai].tolist()
        values += values[:1]
        ax.plot(angles, values, color=color, lw=2, label=arch)
        ax.fill(angles, values, color=color, alpha=0.12)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics, fontsize=10)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels([])
    ax.set_title('Performance comparative normalisée\n(1 = meilleur)', pad=20, fontsize=11)
    ax.legend(loc='upper right', bbox_to_anchor=(1.35, 1.15))
    ax.spines['polar'].set_visible(False)
    plt.tight_layout()
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    plt.savefig(f'{output_dir}/radar_chart.png', dpi=150, bbox_inches='tight')
    plt.close()
