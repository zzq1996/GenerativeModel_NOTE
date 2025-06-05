import torch
import torchvision
import matplotlib.pyplot as plt


def linear_beta_schedule(timesteps, start=0.0001, end=0.02):
    """Creates a linear noise schedule"""
    return torch.linspace(start, end, timesteps)


import torch.nn.functional as F

# Number of time steps in the diffusion process
T = 300

# Create the betas
betas = linear_beta_schedule(timesteps=T)

# Pre-calculations
alphas = 1. - betas
alphas_cumprod = torch.cumprod(alphas, axis=0)
sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)


def get_val_from_t(vals, t, x_shape):
    """
    Helper to get specific t's of a passed list of values 'vals' and returning them with the same dimensions as 'x_shape'
    """
    batch_size = t.shape[0]
    out = vals.gather(-1, t.cpu())

    # reshape and add correct dimension and move to same device as 't'
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)


def forward_diffusion(x_0, t, device="cpu"):
    """
    Takes an initial image 'x_0' and a time step 't' and returns the
    noisy version of the image at time 't' and the noise applied
    """
    noise = torch.randn_like(x_0).to(device)

    sqrt_alphas_cumprod_t = get_val_from_t(sqrt_alphas_cumprod, t, x_0.shape).to(device)
    sqrt_one_minus_alphas_cumprod_t = get_val_from_t(sqrt_one_minus_alphas_cumprod, t, x_0.shape).to(device)

    return (
        sqrt_alphas_cumprod_t * x_0 + sqrt_one_minus_alphas_cumprod_t * noise,
        noise
    )


import torchvision

dataset = torchvision.datasets.StanfordCars(root='/mnt/sda/zhangzq/dataset/DDPM',
                                            split='train',
                                            download=True)

from torchvision import transforms

# resolution to use
IMG_SIZE = 32

dataset_transformed = torchvision.datasets.StanfordCars(root='/mnt/sda/zhangzq/dataset/DDPM',
                                                        transform=transforms.Compose([
                                                            transforms.Resize((IMG_SIZE, IMG_SIZE)),
                                                            transforms.RandomHorizontalFlip(),
                                                            transforms.ToTensor(),
                                                            # convert values to be between [-1.0,  1.0]
                                                            transforms.Lambda(lambda x: (x * 2) - 1)
                                                        ]))

import numpy as np


def tensor_to_pil(image):
    """Reverse the transformation"""
    reverse_transforms = transforms.Compose([
        # back to [0, 1] for the values
        transforms.Lambda(lambda t: (t + 1) / 2),
        # CHW to HWC
        transforms.Lambda(lambda t: t.permute(1, 2, 0)),
        # convert to range [0, 255]
        transforms.Lambda(lambda t: t * 255.),
        transforms.Lambda(lambda t: t.numpy().astype(np.uint8)),
        transforms.ToPILImage(),
    ])

    return reverse_transforms(image)


# take an image
image_0 = dataset_transformed[0][0]

# since plotting 'T' number of images is a bit much we decrease it
num_images = 10
stepsize = T // (num_images - 1)

# plot the image for different t's
# fig, ax = plt.subplots(1, num_images, figsize=(20, 20), constrained_layout=True)
for ind in range(0, num_images):
    t = stepsize * ind
    tt = torch.Tensor([t]).type(torch.int64)

    image_t, _ = forward_diffusion(image_0, tt)
    # ax[ind].axis("off")
    # ax[ind].set_title(f"t = {t}")
    # ax[ind].imshow(tensor_to_pil(image_t))
    std, mean = torch.std_mean(image_t)
    print(f"Mean: {mean:.2f} Std: {std:.2f}")

# pre-calculate some variables
alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
posterior_variance = (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod) * betas
sqrt_one_over_alphas = torch.sqrt(1.0 / alphas)


@torch.no_grad()
def sample_timestep(x, t, model, device):
    """
    Samples images from 'model' given image 'x' and time 't'
    """

    betas_t = get_val_from_t(betas, t, x.shape)
    sqrt_one_minus_alphas_cumprod_t = get_val_from_t(sqrt_one_minus_alphas_cumprod, t, x.shape)
    sqrt_one_over_alphas_t = get_val_from_t(sqrt_one_over_alphas, t, x.shape)

    posterior_mean_t = sqrt_one_over_alphas_t * (x - betas_t / sqrt_one_minus_alphas_cumprod_t * model(x, t))
    posterior_variance_t = get_val_from_t(posterior_variance, t, x.shape)

    # do not add noise if we are at t = 0
    noise = torch.randn_like(x) if t > 0 else 0.0

    return posterior_mean_t + torch.sqrt(posterior_variance_t) * noise


@torch.no_grad()
def sample_images(model, device, num_samples=5, images_per_sample=10):
    """
    Helper function to sample 'num_samples' images from the model
    and 'images_per_sample' images for each sample spread out between t = [0, T]
    """

    # tensor to store the results
    images = torch.empty((num_samples, num_images, 1, 3, IMG_SIZE, IMG_SIZE))

    stepsize = int(T / num_images)
    for sample in range(num_samples):

        # sample noise
        img = torch.randn((1, 3, IMG_SIZE, IMG_SIZE), device=device)

        # loop backwards
        for i in range(0, T)[::-1]:

            t = torch.full((1,), i, device=device, dtype=torch.long)
            img = sample_timestep(img, t, model, device=device)

            if i % stepsize == 0:
                # store sample image
                col = num_images - i // stepsize - 1
                images[sample, col] = img.detach().cpu()

    return images


import torch.nn.functional as F


def get_loss(model, x_0, t, device):
    # get noisy image and the noise from the forward diffusion
    x_t, noise = forward_diffusion(x_0, t, device)

    # predict the noise given noisy image and t
    noise_pred = model(x_t, t)

    return F.mse_loss(noise, noise_pred)


from functools import partial
import math
from tokenize import group

import torch
from torch import nn
from einops import reduce, rearrange


def l2norm(t):
    return F.normalize(t, dim=-1)


class UNet(nn.Module):
    def __init__(self, img_channels: int, init_dim: int, time_emb_dim: int, num_res: int = 4):
        """Creates a UNet

        Args:
            in_channels (int): number of images channels
            init_dim (int): number of output channels in the first layer
            time_emb_dim (int): time dimension size
            num_res (int, optional): Number of resolutions
        """
        super().__init__()

        # initial conv
        self.init_conv = nn.Conv2d(img_channels, init_dim, kernel_size=7, padding=3)

        # create list of the different dimensions
        dims = [init_dim, *map(lambda m: init_dim * m, [2 ** res for res in range(0, num_res)])]

        # create convenient list of tuples with input and output channels for each resolution
        in_out_dims = list(zip(dims[:-1], dims[1:]))

        # time embedding block
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(init_dim),
            nn.Linear(init_dim, time_emb_dim),
            nn.GELU(),
            nn.Linear(time_emb_dim, time_emb_dim)
        )

        # downsample
        self.down_layers = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(in_out_dims):
            is_last = ind >= num_res - 1

            self.down_layers.append(nn.ModuleList([
                ResNetBlock(dim_in, dim_in, time_emb_dim=time_emb_dim),
                ResNetBlock(dim_in, dim_in, time_emb_dim=time_emb_dim),
                Residual(PreNorm(dim_in, LinearAttention(dim_in))),
                Downsample(dim_in, dim_out) if not is_last else nn.Conv2d(dim_in, dim_out, 3, padding=1)
            ]))

        # middle block
        mid_dim = dims[-1]
        self.mid_block1 = ResNetBlock(mid_dim, mid_dim, time_emb_dim=time_emb_dim)
        self.mid_attn = Residual(PreNorm(mid_dim, Attention(mid_dim)))
        self.mid_block2 = ResNetBlock(mid_dim, mid_dim, time_emb_dim=time_emb_dim)

        # upsample
        self.up_layers = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(reversed(in_out_dims)):
            is_last = ind == num_res - 1

            self.up_layers.append(nn.ModuleList([
                ResNetBlock(dim_in + dim_out, dim_out, time_emb_dim=time_emb_dim),
                ResNetBlock(dim_in + dim_out, dim_out, time_emb_dim=time_emb_dim),
                Residual(PreNorm(dim_out, LinearAttention(dim_out))),
                Upsample(dim_out, dim_in) if not is_last else nn.Conv2d(dim_out, dim_in, 3, padding=1)
            ]))

        self.final_res_block = ResNetBlock(init_dim * 2, init_dim, time_emb_dim=time_emb_dim)
        self.final_conv = nn.Conv2d(init_dim, img_channels, 1)

    def forward(self, x, time):

        x = self.init_conv(x)

        r = x.clone()

        t = self.time_mlp(time)

        h = []

        for block1, block2, attn, downsample in self.down_layers:
            x = block1(x, t)
            h.append(x)

            x = block2(x, t)
            x = attn(x)
            h.append(x)

            x = downsample(x)

        x = self.mid_block1(x, t)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t)

        for block1, block2, attn, upsample in self.up_layers:
            x = torch.cat((x, h.pop()), dim=1)
            x = block1(x, t)

            x = torch.cat((x, h.pop()), dim=1)
            x = block2(x, t)
            x = attn(x)

            x = upsample(x)

        x = torch.cat((x, r), dim=1)

        x = self.final_res_block(x, t)
        return self.final_conv(x)


class WeightStandardizedConv2d(nn.Conv2d):
    """
    https://arxiv.org/abs/1903.10520
    weight standardization purportedly works synergistically with group normalization
    """

    def forward(self, x):
        eps = 1e-5 if x.dtype == torch.float32 else 1e-3

        weight = self.weight
        mean = reduce(weight, 'o ... -> o 1 1 1', 'mean')
        var = reduce(weight, 'o ... -> o 1 1 1', partial(torch.var, unbiased=False))
        normalized_weight = (weight - mean) * (var + eps).rsqrt()

        return F.conv2d(x, normalized_weight, self.bias, self.stride, self.padding, self.dilation, self.groups)


class ResNetBlock(nn.Module):
    def __init__(self, dim_in, dim_out, time_emb_dim, groups=8):
        super().__init__()

        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, dim_out * 2)
        )

        self.block1 = Block(dim_in, dim_out, groups=groups)
        self.block2 = Block(dim_out, dim_out, groups=groups)
        self.res_conv = nn.Conv2d(dim_in, dim_out, 1) if dim_in != dim_out else nn.Identity()

    def forward(self, x, t, time_emb=None):
        scale_shift = None
        if self.mlp is not None and time_emb is not None:
            time_emb = self.mlp(time_emb)
            time_emb = rearrange(time_emb, 'b c -> b c 1 1')
            scale_shift = time_emb.chunk(2, dim=1)

        h = self.block1(x, scale_shift=scale_shift)
        h = self.block2(h)
        return h + self.res_conv(x)


class Block(nn.Module):
    def __init__(self, dim, dim_out, groups=8):
        super().__init__()
        self.proj = WeightStandardizedConv2d(dim, dim_out, 3, padding=1)
        self.norm = nn.GroupNorm(groups, dim_out)
        self.act = nn.SiLU()

    def forward(self, x, scale_shift=None):
        x = self.proj(x)
        x = self.norm(x)

        if scale_shift is not None:
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        x = self.act(x)
        return x


def Upsample(dim_in, dim_out):
    return nn.Sequential(
        nn.Upsample(scale_factor=2, mode='nearest'),
        nn.Conv2d(dim_in, dim_out, 3, padding=1)
    )


def Downsample(dim_in, dim_out=None):
    return nn.Conv2d(dim_in, dim_out, 4, 2, 1)


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x


class LayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))

    def forward(self, x):
        eps = 1e-5 if x.dtype == torch.float32 else 1e-3
        var = torch.var(x, dim=1, unbiased=False, keepdim=True)
        mean = torch.mean(x, dim=1, keepdim=True)
        return (x - mean) * (var + eps).rsqrt() * self.g


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = LayerNorm(dim)

    def forward(self, x):
        x = self.norm(x)
        return self.fn(x)


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class LinearAttention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)

        self.to_out = nn.Sequential(
            nn.Conv2d(hidden_dim, dim, 1),
            LayerNorm(dim)
        )

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(lambda t: rearrange(t, 'b (h c) x y -> b h c (x y)', h=self.heads), qkv)

        q = q.softmax(dim=-2)
        k = k.softmax(dim=-1)

        q = q * self.scale
        v = v / (h * w)

        context = torch.einsum('b h d n, b h e n -> b h d e', k, v)

        out = torch.einsum('b h d e, b h d n -> b h e n', context, q)
        out = rearrange(out, 'b h c (x y) -> b (h c) x y', h=self.heads, x=h, y=w)
        return self.to_out(out)


class Attention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32, scale=10):
        super().__init__()
        self.scale = scale
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(lambda t: rearrange(t, 'b (h c) x y -> b h c (x y)', h=self.heads), qkv)

        q, k = map(l2norm, (q, k))

        sim = torch.einsum('b h d i, b h d j -> b h i j', q, k) * self.scale
        attn = sim.softmax(dim=-1)
        out = torch.einsum('b h i j, b h d j -> b h i d', attn, v)
        out = rearrange(out, 'b h (x y) d -> b (h d) x y', x=h, y=w)
        return self.to_out(out)


model = UNet(img_channels=3, init_dim=64, time_emb_dim=32)
print(f"Num params: {sum(p.numel() for p in model.parameters()):,}")

import pytorch_lightning as pl
import torch
import torchvision


class LightningModel(pl.LightningModule):
    def __init__(self, model):
        super().__init__()

        # just wrap the model
        self.model = model

    def forward(self, x, t):
        return self.model(x, t)

    def training_step(self, batch, batch_idx):
        """Here we do the optimization"""

        # batch is in a list
        batch = batch[0]

        batch_size = batch.shape[0]

        # sample timesteps
        t = torch.randint(0, T, (batch_size,), device=self.device).long()

        # calc the loss
        loss = get_loss(self.model, batch, t, device=self.device)

        # log loss to tensorboard
        self.log("Loss", loss, on_step=True, on_epoch=True)

        return loss

    def on_train_epoch_end(self) -> None:
        """Function that runs when an epoch has been trained"""

        # every 10 epoch we sample images and log to tensorboard
        if self.trainer.current_epoch % 10 == 0:
            images = sample_images(model=self.model, device=self.device)
            img_h, img_w = images.shape[-2:]
            image_grid = torchvision.utils.make_grid(images.reshape(-1, 3, img_h, img_w), nrow=images.shape[1],
                                                     normalize=True)
            self.logger.experiment.add_image('Model samples', image_grid, self.trainer.current_epoch)

        # return super().training_epoch_end(image_grid)

    def configure_optimizers(self):
        """Define our optimizer"""
        return torch.optim.Adam(model.parameters(), lr=0.0002)


from torch.utils.data import DataLoader, Subset

# Select a subset of the data in order to converge faster
dataset_transformed = Subset(dataset_transformed, range(0, 3000))

dataloader = DataLoader(
    dataset_transformed,
    batch_size=128,
    shuffle=True,
    drop_last=True,
    num_workers=24
)

trainer = pl.Trainer(
    accelerator="gpu",
    devices=1,
    precision=32,
    max_epochs=1000,
    logger=True,
    log_every_n_steps=1
)
model = LightningModel(model)
trainer.fit(model, dataloader)
