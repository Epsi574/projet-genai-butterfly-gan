"""
Generic training loop and utilities for GANs.
"""
import torch
import time
import shutil
from pathlib import Path
from tqdm.auto import tqdm
from torchvision import utils

from losses import dcgan_loss_D, dcgan_loss_G, wgan_loss_D, wgan_loss_G
from losses import gradient_penalty, hinge_loss_D, hinge_loss_G


def save_checkpoint(arch_name, epoch, G, D, opt_G, opt_D, history, 
                   n_classes, class_to_idx, config, checkpoint_dir):
    """Save complete checkpoint."""
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
    ckpt_path = f'{checkpoint_dir}/{arch_name}_epoch{epoch:04d}.pt'
    torch.save({
        'epoch': epoch,
        'arch': arch_name,
        'G_state': G.state_dict(),
        'D_state': D.state_dict(),
        'opt_G_state': opt_G.state_dict(),
        'opt_D_state': opt_D.state_dict(),
        'history': history,
        'n_classes': n_classes,
        'class_to_idx': class_to_idx,
        'config': config,
    }, ckpt_path)
    shutil.copy(ckpt_path, f'{checkpoint_dir}/{arch_name}_latest.pt')
    return ckpt_path


def load_checkpoint(arch_name, G, D, opt_G, opt_D, checkpoint_dir, device):
    """Load latest checkpoint if available."""
    path = f'{checkpoint_dir}/{arch_name}_latest.pt'
    if not Path(path).exists():
        print(f'   No checkpoint found for {arch_name}')
        return 0, []
    ckpt = torch.load(path, map_location=device)
    G.load_state_dict(ckpt['G_state'])
    D.load_state_dict(ckpt['D_state'])
    opt_G.load_state_dict(ckpt['opt_G_state'])
    opt_D.load_state_dict(ckpt['opt_D_state'])
    print(f'   Checkpoint loaded: epoch {ckpt["epoch"]}')
    return ckpt['epoch'], ckpt.get('history', [])


def train_gan(arch_name, G, D, opt_G, opt_D, dataloader, n_epochs, device,
              n_classes, latent_dim, class_to_idx, config_dict,
              mode='dcgan', n_critic=1, gp_lambda=10, 
              checkpoint_dir='./checkpoints', save_every=20, resume=True):
    """
    Training loop for all three architectures.
    
    Args:
        mode: 'dcgan' | 'wgan-gp' | 'sngan'
        n_critic: discriminator steps per generator step (WGAN)
        gp_lambda: gradient penalty weight (WGAN-GP)
        resume: resume from latest checkpoint
    """
    G.to(device)
    D.to(device)
    G.train()
    D.train()

    start_epoch = 0
    history = []
    if resume:
        start_epoch, history = load_checkpoint(arch_name, G, D, opt_G, opt_D, 
                                               checkpoint_dir, device)

    sep = '=' * 60
    print(f'\n{sep}')
    print(f'  Training: {arch_name} ({mode.upper()})')
    print(f'  Epochs  : {start_epoch+1} -> {n_epochs}')
    print(f'  Batches/ep   : {len(dataloader)}')
    print(f'{sep}')

    for epoch in range(start_epoch + 1, n_epochs + 1):
        epoch_g_loss = 0.0
        epoch_d_loss = 0.0
        t0 = time.time()

        pbar = tqdm(dataloader, desc=f'Ep {epoch}/{n_epochs}', leave=False)

        for real_imgs, real_labels in pbar:
            B = real_imgs.size(0)
            real_imgs = real_imgs.to(device)
            real_labels = real_labels.to(device)

            n_d_steps = n_critic if mode == 'wgan-gp' else 1

            for _ in range(n_d_steps):
                opt_D.zero_grad()
                z = torch.randn(B, latent_dim, device=device)
                fake_labels = torch.randint(0, n_classes, (B,), device=device)
                with torch.no_grad():
                    fake_imgs = G(z, fake_labels)

                real_logits = D(real_imgs, real_labels)
                fake_logits = D(fake_imgs.detach(), fake_labels)

                if mode == 'dcgan':
                    d_loss = dcgan_loss_D(real_logits, fake_logits)
                elif mode == 'wgan-gp':
                    d_loss = wgan_loss_D(real_logits, fake_logits)
                    gp = gradient_penalty(D, real_imgs, fake_imgs.detach(),
                                        real_labels, device)
                    d_loss = d_loss + gp_lambda * gp
                elif mode == 'sngan':
                    d_loss = hinge_loss_D(real_logits, fake_logits)

                d_loss.backward()
                opt_D.step()

            epoch_d_loss += d_loss.item()

            opt_G.zero_grad()
            z = torch.randn(B, latent_dim, device=device)
            fake_labels = torch.randint(0, n_classes, (B,), device=device)
            fake_imgs = G(z, fake_labels)
            fake_logits = D(fake_imgs, fake_labels)

            if mode == 'dcgan':
                g_loss = dcgan_loss_G(fake_logits)
            elif mode == 'wgan-gp':
                g_loss = wgan_loss_G(fake_logits)
            elif mode == 'sngan':
                g_loss = hinge_loss_G(fake_logits)

            g_loss.backward()
            opt_G.step()
            epoch_g_loss += g_loss.item()

            pbar.set_postfix({'D': f'{d_loss.item():.3f}', 'G': f'{g_loss.item():.3f}'})

        # Fin d'epoch
        avg_g = epoch_g_loss / len(dataloader)
        avg_d = epoch_d_loss / len(dataloader)
        elapsed = time.time() - t0
        history.append({'epoch': epoch, 'g_loss': avg_g, 'd_loss': avg_d, 'time': elapsed})

        print(f'Ep {epoch:4d}/{n_epochs} | G: {avg_g:7.4f} | D: {avg_d:7.4f} | {elapsed:.1f}s')

        if epoch % save_every == 0 or epoch == n_epochs:
            path = save_checkpoint(arch_name, epoch, G, D, opt_G, opt_D, history,
                                 n_classes, class_to_idx, config_dict, checkpoint_dir)
            print(f'   Checkpoint -> {path}')

    return G, D, history
