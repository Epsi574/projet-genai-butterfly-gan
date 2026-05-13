"""
Main script for training and evaluating conditional GANs.
Benchmark of three architectures: CDCGAN, CWGAN-GP, CSNGAN.
"""
import torch
import torch.optim as optim
import pandas as pd
import json
import shutil
from pathlib import Path

import config
from dataset import load_dataset, ButterflyDataset, get_transforms
from models import ConditionalGenerator, ConditionalDiscriminator
from train import train_gan
from evaluation import save_real_images_for_fid, evaluate_all
from visualization import (generate_fixed_grid, plot_losses, plot_dataset_distribution,
                          comparative_grid, plot_intraclass_fid, plot_comparative_losses, 
                          radar_chart)


def main():
    """Point d'entrée principal du benchmark."""
    
    print(f'\n{"="*70}')
    print(f'  Conditional GANs Benchmark - Butterfly Classification')
    print(f'  Architectures: CDCGAN, CWGAN-GP, CSNGAN')
    print(f'  Device: {config.DEVICE}')
    if config.DEVICE.type == 'cuda':
        print(f'  GPU: {torch.cuda.get_device_name(0)}')
    print(f'{"="*70}\n')
    
    # Créer les dossiers de sortie
    Path(config.CHECKPOINT_DIR).mkdir(parents=True, exist_ok=True)
    Path(config.GENERATED_DIR).mkdir(parents=True, exist_ok=True)
    Path(config.RESULTS_DIR).mkdir(parents=True, exist_ok=True)
    
    # Load dataset
    print('\nLoading dataset...')
    csv_path = f'{config.DATA_DIR}/Training_set.csv'
    img_root = f'{config.DATA_DIR}/train'
    
    dataloader, n_classes, class_to_idx, idx_to_class, df_stats = load_dataset(
        csv_path=csv_path,
        img_root=img_root,
        top_k=config.TOP_K,
        img_size=config.IMG_SIZE,
        batch_size=config.BATCH_SIZE
    )
    
    # Visualisation distribution
    plot_dataset_distribution(df_stats, config.RESULTS_DIR)
    
    selected_classes = list(class_to_idx.keys())
    
    # Configuration pour checkpoint
    config_dict = {
        'latent_dim': config.LATENT_DIM,
        'embed_dim': config.EMBED_DIM,
        'img_size': config.IMG_SIZE,
    }
    
    # ==================== CDCGAN ====================
    print('\n' + '='*70)
    print('  ARCHITECTURE 1: CDCGAN (Conditional DCGAN)')
    print('='*70)
    
    G_dcgan = ConditionalGenerator(config.LATENT_DIM, n_classes, config.EMBED_DIM, config.NGF)
    D_dcgan = ConditionalDiscriminator(n_classes, config.EMBED_DIM, config.NDF, 
                                      use_sn=False, use_bn=True)
    
    opt_G_dcgan = optim.Adam(G_dcgan.parameters(), lr=config.LR_G, betas=(0.5, 0.999))
    opt_D_dcgan = optim.Adam(D_dcgan.parameters(), lr=config.LR_D, betas=(0.5, 0.999))
    
    G_dcgan, D_dcgan, hist_dcgan = train_gan(
        arch_name='CDCGAN',
        G=G_dcgan,
        D=D_dcgan,
        opt_G=opt_G_dcgan,
        opt_D=opt_D_dcgan,
        dataloader=dataloader,
        n_epochs=config.N_EPOCHS,
        device=config.DEVICE,
        n_classes=n_classes,
        latent_dim=config.LATENT_DIM,
        class_to_idx=class_to_idx,
        config_dict=config_dict,
        mode='dcgan',
        checkpoint_dir=config.CHECKPOINT_DIR,
        save_every=config.SAVE_EVERY,
        resume=True
    )
    
    generate_fixed_grid(G_dcgan, 'CDCGAN', config.N_EPOCHS, n_classes, 
                       idx_to_class, config.LATENT_DIM, config.DEVICE, 
                       config.GENERATED_DIR)
    plot_losses(hist_dcgan, 'CDCGAN', config.RESULTS_DIR)
    
    # ==================== CWGAN-GP ====================
    print('\n' + '='*70)
    print('  ARCHITECTURE 2: CWGAN-GP (Conditional Wasserstein GAN + GP)')
    print('='*70)
    
    G_wgan = ConditionalGenerator(config.LATENT_DIM, n_classes, config.EMBED_DIM, config.NGF)
    D_wgan = ConditionalDiscriminator(n_classes, config.EMBED_DIM, config.NDF,
                                     use_sn=False, use_bn=False)
    
    opt_G_wgan = optim.Adam(G_wgan.parameters(), lr=config.LR_G, betas=(0.0, 0.9))
    opt_D_wgan = optim.Adam(D_wgan.parameters(), lr=config.LR_D, betas=(0.0, 0.9))
    
    G_wgan, D_wgan, hist_wgan = train_gan(
        arch_name='CWGAN-GP',
        G=G_wgan,
        D=D_wgan,
        opt_G=opt_G_wgan,
        opt_D=opt_D_wgan,
        dataloader=dataloader,
        n_epochs=config.N_EPOCHS,
        device=config.DEVICE,
        n_classes=n_classes,
        latent_dim=config.LATENT_DIM,
        class_to_idx=class_to_idx,
        config_dict=config_dict,
        mode='wgan-gp',
        n_critic=config.N_CRITIC,
        gp_lambda=config.GP_LAMBDA,
        checkpoint_dir=config.CHECKPOINT_DIR,
        save_every=config.SAVE_EVERY,
        resume=True
    )
    
    generate_fixed_grid(G_wgan, 'CWGAN-GP', config.N_EPOCHS, n_classes,
                       idx_to_class, config.LATENT_DIM, config.DEVICE,
                       config.GENERATED_DIR)
    plot_losses(hist_wgan, 'CWGAN-GP', config.RESULTS_DIR)
    
    # ==================== CSNGAN ====================
    print('\n' + '='*70)
    print('  ARCHITECTURE 3: CSNGAN (Conditional Spectral Normalization GAN)')
    print('='*70)
    
    G_sngan = ConditionalGenerator(config.LATENT_DIM, n_classes, config.EMBED_DIM, config.NGF)
    D_sngan = ConditionalDiscriminator(n_classes, config.EMBED_DIM, config.NDF,
                                      use_sn=True, use_bn=False)
    
    opt_G_sngan = optim.Adam(G_sngan.parameters(), lr=2e-4, betas=(0.0, 0.9))
    opt_D_sngan = optim.Adam(D_sngan.parameters(), lr=1e-4, betas=(0.0, 0.9))
    
    G_sngan, D_sngan, hist_sngan = train_gan(
        arch_name='CSNGAN',
        G=G_sngan,
        D=D_sngan,
        opt_G=opt_G_sngan,
        opt_D=opt_D_sngan,
        dataloader=dataloader,
        n_epochs=config.N_EPOCHS,
        device=config.DEVICE,
        n_classes=n_classes,
        latent_dim=config.LATENT_DIM,
        class_to_idx=class_to_idx,
        config_dict=config_dict,
        mode='sngan',
        checkpoint_dir=config.CHECKPOINT_DIR,
        save_every=config.SAVE_EVERY,
        resume=True
    )
    
    generate_fixed_grid(G_sngan, 'CSNGAN', config.N_EPOCHS, n_classes,
                       idx_to_class, config.LATENT_DIM, config.DEVICE,
                       config.GENERATED_DIR)
    plot_losses(hist_sngan, 'CSNGAN', config.RESULTS_DIR)
    
    # ==================== ÉVALUATION ====================
    print('\n' + '='*70)
    print('  ÉVALUATION QUANTITATIVE')
    print('='*70)
    
    # Prepare real images
    print('\nPreparing real images for FID...')
    transform = get_transforms(config.IMG_SIZE, mode='eval')
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip().str.lower()
    real_dataset = ButterflyDataset(df, img_root, class_to_idx, transform)
    real_dir = save_real_images_for_fid(real_dataset, idx_to_class, 
                                       n_per_class=config.N_EVAL_IMG)
    
    # Evaluate CDCGAN
    print('\nEvaluating CDCGAN...')
    results_dcgan = evaluate_all(G_dcgan, 'CDCGAN', real_dir, n_classes, 
                                 idx_to_class, config.LATENT_DIM, config.DEVICE,
                                 selected_classes, n_per_class=config.N_EVAL_IMG)
    
    # Cleanup generated images only (keep real images for next architectures)
    print('Cleaning up CDCGAN temporary files...')
    shutil.rmtree('./fid_gen/CDCGAN', ignore_errors=True)
    shutil.rmtree('./fid_flat/CDCGAN', ignore_errors=True)
    
    # Evaluate CWGAN-GP
    print('\nEvaluating CWGAN-GP...')
    results_wgan = evaluate_all(G_wgan, 'CWGAN-GP', real_dir, n_classes,
                               idx_to_class, config.LATENT_DIM, config.DEVICE,
                               selected_classes, n_per_class=config.N_EVAL_IMG)
    
    # Cleanup generated images only
    print('Cleaning up CWGAN-GP temporary files...')
    shutil.rmtree('./fid_gen/CWGAN-GP', ignore_errors=True)
    shutil.rmtree('./fid_flat/CWGAN-GP', ignore_errors=True)
    
    # Evaluate CSNGAN
    print('\nEvaluating CSNGAN...')
    results_sngan = evaluate_all(G_sngan, 'CSNGAN', real_dir, n_classes,
                                idx_to_class, config.LATENT_DIM, config.DEVICE,
                                selected_classes, n_per_class=config.N_EVAL_IMG)
    
    # Final cleanup of all temporary files
    print('Final cleanup of all temporary FID files...')
    shutil.rmtree('./fid_gen', ignore_errors=True)
    shutil.rmtree('./fid_real', ignore_errors=True)
    shutil.rmtree('./fid_flat', ignore_errors=True)
    
    # ==================== VISUALISATIONS ====================
    print('\n' + '='*70)
    print('  GÉNÉRATION DES VISUALISATIONS')
    print('='*70)
    
    # Grille comparative
    showcase_species = ['MOURNING CLOAK', 'BROWN SIPROETA', 'BANDED ORANGE HELICONIAN']
    showcase_species = [s for s in showcase_species if s in selected_classes]
    if showcase_species:
        comparative_grid(
            real_dir=real_dir,
            gen_dirs=[results_dcgan['gen_dir'], results_wgan['gen_dir'], 
                     results_sngan['gen_dir']],
            arch_names=['CDCGAN', 'CWGAN-GP', 'CSNGAN'],
            species_list=showcase_species[:3],
            n_imgs=4,
            output_dir=config.RESULTS_DIR
        )
    
    # Intra-class FID
    plot_intraclass_fid(results_dcgan['ic_fid_per_class'],
                       results_wgan['ic_fid_per_class'],
                       results_sngan['ic_fid_per_class'],
                       config.RESULTS_DIR)
    
    # Courbes comparatives
    plot_comparative_losses(hist_dcgan, hist_wgan, hist_sngan, config.RESULTS_DIR)
    
    # ==================== TABLEAU RÉCAPITULATIF ====================
    print('\n' + '='*70)
    print('  TABLEAU RÉCAPITULATIF')
    print('='*70)
    
    results = {
        'Architecture': ['CDCGAN', 'CWGAN-GP', 'CSNGAN'],
        'Loss': ['BCE', 'Wasserstein + GP', 'Hinge'],
        'Régularisation': ['BatchNorm', 'Gradient Penalty', 'Spectral Norm'],
        'FID Global :': [
            round(results_dcgan['fid'], 2),
            round(results_wgan['fid'], 2),
            round(results_sngan['fid'], 2)
        ],
        'IC-FID :': [
            round(results_dcgan['ic_fid'], 2),
            round(results_wgan['ic_fid'], 2),
            round(results_sngan['ic_fid'], 2)
        ],
        'Precision ': [
            round(results_dcgan['precision'], 3),
            round(results_wgan['precision'], 3),
            round(results_sngan['precision'], 3)
        ],
        'Recall': [
            round(results_dcgan['recall'], 3),
            round(results_wgan['recall'], 3),
            round(results_sngan['recall'], 3)
        ],
    }
    
    df_results = pd.DataFrame(results)
    print('\n' + df_results.to_string(index=False))
    
    df_results.to_csv(f'{config.RESULTS_DIR}/benchmark_results.csv', index=False)
    
    radar_chart(df_results, config.RESULTS_DIR)
    
    # Meilleurs résultats
    best_global_fid = df_results.loc[df_results['FID Global :'].idxmin(), 'Architecture']
    best_ic_fid = df_results.loc[df_results['IC-FID :'].idxmin(), 'Architecture']
    best_precision = df_results.loc[df_results['Precision '].idxmax(), 'Architecture']
    best_recall = df_results.loc[df_results['Recall'].idxmax(), 'Architecture']
    
    # Rapport JSON
    report = {
        'dataset': 'butterfly-image-classification (Kaggle)',
        'reference': 'StudioGAN: Kang et al., 2023',
        'config': {
            'img_size': config.IMG_SIZE,
            'latent_dim': config.LATENT_DIM,
            'n_classes': n_classes,
            'n_epochs': config.N_EPOCHS,
            'batch_size': config.BATCH_SIZE,
            'n_eval_imgs_per_class': config.N_EVAL_IMG,
            'selected_classes': selected_classes,
        },
        'results': {
            'CDCGAN': results_dcgan,
            'CWGAN-GP': results_wgan,
            'CSNGAN': results_sngan,
        },
        'best': {
            'fid_global': best_global_fid,
            'ic_fid': best_ic_fid,
            'precision': best_precision,
            'recall': best_recall,
        }
    }
    
    # Nettoyer les clés non sérialisables
    for arch in ['CDCGAN', 'CWGAN-GP', 'CSNGAN']:
        report['results'][arch].pop('gen_dir', None)
    
    with open(f'{config.RESULTS_DIR}/benchmark_report.json', 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f'\n{"="*70}')
    print(f'  BENCHMARK COMPLETED')
    print(f'  Results saved in: {config.RESULTS_DIR}/')
    print(f'  - benchmark_results.csv : metrics table')
    print(f'  - benchmark_report.json : full report')
    print(f'  - *.png : visualizations')
    print(f'{"="*70}\n')


if __name__ == '__main__':
    main()
