"""
Loss functions for the three GAN architectures.
"""
import torch
import torch.nn.functional as F


def dcgan_loss_D(real_logits, fake_logits):
    """Standard BCE discriminator loss (DCGAN)."""
    real_loss = F.binary_cross_entropy_with_logits(real_logits, torch.ones_like(real_logits))
    fake_loss = F.binary_cross_entropy_with_logits(fake_logits, torch.zeros_like(fake_logits))
    return (real_loss + fake_loss) / 2


def dcgan_loss_G(fake_logits):
    """Non-saturating BCE generator loss (DCGAN)."""
    return F.binary_cross_entropy_with_logits(fake_logits, torch.ones_like(fake_logits))


def wgan_loss_D(real_scores, fake_scores):
    """Wasserstein critic loss: maximize E[D(real)] - E[D(fake)]."""
    return fake_scores.mean() - real_scores.mean()


def wgan_loss_G(fake_scores):
    """Wasserstein generator loss: minimize -E[D(fake)]."""
    return -fake_scores.mean()


def gradient_penalty(critic, real_imgs, fake_imgs, real_labels, device):
    """
    Gradient Penalty (Gulrajani et al., 2017).
    Penalizes critic gradient interpolated between real and fake.
    """
    B = real_imgs.size(0)
    alpha = torch.rand(B, 1, 1, 1, device=device)
    interpolated = alpha * real_imgs + (1 - alpha) * fake_imgs
    labels = real_labels
    interpolated.requires_grad_(True)

    d_interp = critic(interpolated, labels)
    grads = torch.autograd.grad(
        outputs=d_interp,
        inputs=interpolated,
        grad_outputs=torch.ones_like(d_interp),
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]
    grads = grads.view(B, -1)
    gp = ((grads.norm(2, dim=1) - 1) ** 2).mean()
    return gp


def hinge_loss_D(real_logits, fake_logits):
    """Hinge discriminator loss (SNGAN)."""
    real_loss = F.relu(1.0 - real_logits).mean()
    fake_loss = F.relu(1.0 + fake_logits).mean()
    return real_loss + fake_loss


def hinge_loss_G(fake_logits):
    """Hinge generator loss (SNGAN)."""
    return -fake_logits.mean()
