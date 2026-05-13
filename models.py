"""
Conditional generator and discriminator architectures.
Shared backbone for CDCGAN, CWGAN-GP and CSNGAN.
"""
import torch
import torch.nn as nn


class TransposedConvBlock(nn.Module):
    """Transpose convolution block with BN + ReLU (generator)."""
    def __init__(self, in_ch, out_ch, last=False):
        super().__init__()
        layers = [nn.ConvTranspose2d(in_ch, out_ch, 4, 2, 1, bias=False)]
        if not last:
            layers += [nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True)]
        else:
            layers.append(nn.Tanh())  # sortie dans [-1, 1]
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class ConvBlock(nn.Module):
    """Bloc de convolution pour discriminateur."""
    def __init__(self, in_ch, out_ch, stride=2, use_bn=True, use_sn=False,
                 use_ln=False, ln_shape=None):
        super().__init__()
        conv = nn.Conv2d(in_ch, out_ch, 4, stride, 1, bias=not (use_bn or use_ln))
        if use_sn:
            conv = nn.utils.spectral_norm(conv)
        layers = [conv]
        if use_bn:
            layers.append(nn.BatchNorm2d(out_ch))
        elif use_ln and ln_shape is not None:
            layers.append(nn.LayerNorm(ln_shape))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class ConditionalGenerator(nn.Module):
    """
    Conditional DCGAN generator.
    Architecture: z + embed(label) -> Linear -> reshape -> 4 TransposedConvBlocks -> 64×64 RGB
    """
    def __init__(self, latent_dim, n_classes, embed_dim, ngf=64):
        super().__init__()
        self.label_embed = nn.Embedding(n_classes, embed_dim)
        input_dim = latent_dim + embed_dim

        # Projection vers feature map 4×4
        self.proj = nn.Sequential(
            nn.Linear(input_dim, ngf * 8 * 4 * 4, bias=False),
        )
        self.ngf = ngf

        # 4×4 -> 8×8 -> 16×16 -> 32×32 -> 64×64
        self.deconv = nn.Sequential(
            TransposedConvBlock(ngf * 8, ngf * 4),
            TransposedConvBlock(ngf * 4, ngf * 2),
            TransposedConvBlock(ngf * 2, ngf),
            TransposedConvBlock(ngf, 3, last=True),
        )
        self._init_weights()

    def _init_weights(self):
        """Initialisation récursive DCGAN."""
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.normal_(m.weight.data, 0.0, 0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias.data, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.normal_(m.weight.data, 1.0, 0.02)
                nn.init.constant_(m.bias.data, 0)

    def forward(self, z, labels):
        emb = self.label_embed(labels)
        x = torch.cat([z, emb], dim=1)
        x = self.proj(x)
        x = x.view(-1, self.ngf * 8, 4, 4)
        return self.deconv(x)

    def count_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class ConditionalDiscriminator(nn.Module):
    """
    Discriminateur conditionnel avec projection conditionnelle.
    Supporte BatchNorm, Spectral Norm et LayerNorm.
    """
    def __init__(self, n_classes, embed_dim, ndf=64, use_sn=False, use_bn=True):
        super().__init__()
        self.use_sn = use_sn
        use_ln = (not use_bn) and (not use_sn)

        def _conv(in_ch, out_ch, spatial, first=False):
            return ConvBlock(
                in_ch, out_ch, stride=2,
                use_bn=use_bn and not first,
                use_sn=use_sn,
                use_ln=use_ln and not first,
                ln_shape=[out_ch, spatial, spatial] if not first else None,
            )

        self.features = nn.Sequential(
            _conv(3, ndf, 32, first=True),
            _conv(ndf, ndf * 2, 16),
            _conv(ndf * 2, ndf * 4, 8),
            _conv(ndf * 4, ndf * 8, 4),
        )
        feat_dim = ndf * 8 * 4 * 4
        linear = nn.Linear(feat_dim, 1)
        if use_sn:
            linear = nn.utils.spectral_norm(linear)
        self.linear = linear

        self.embed = nn.Embedding(n_classes, feat_dim)
        if use_sn:
            self.embed = nn.utils.spectral_norm(self.embed)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.normal_(m.weight.data, 0.0, 0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias.data, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.normal_(m.weight.data, 1.0, 0.02)
                nn.init.constant_(m.bias.data, 0)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight.data, 0.0, 0.02)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight.data, 0.0, 0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias.data, 0)

    def forward(self, x, labels):
        phi = self.features(x)
        phi = phi.view(phi.size(0), -1)
        out = self.linear(phi)
        emb = self.embed(labels)
        proj = (phi * emb).sum(dim=1, keepdim=True)
        return (out + proj).squeeze(1)

    def count_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
