import torch
import torch.nn as nn
import yaml
from pathlib import PosixPath

from .modules import Encoder, Decoder
from .distributions import DiagonalGaussianDistribution
from dmodels import get_cfg



class AutoencoderKL(nn.Module):
    def __init__(self, ddconfig, embed_dim):
        super().__init__()
        self.encoder = Encoder(**ddconfig)
        self.decoder = Decoder(**ddconfig)
        assert ddconfig["double_z"]
        self.quant_conv = torch.nn.Conv2d(2*ddconfig["z_channels"], 2*embed_dim, 1)
        self.post_quant_conv = torch.nn.Conv2d(embed_dim, ddconfig["z_channels"], 1)
        self.embed_dim = embed_dim
   
    @classmethod
    def from_config(cls, config=None):
        if config is None:
            config = get_cfg("kl_f8.yaml")

        if isinstance(config, (str, PosixPath)):
            with open(config, "r") as f:
                cfg = yaml.load(f, Loader=yaml.FullLoader)
        elif isinstance(config, dict):
            cfg = config
        else:
            raise ValueError(f"Invalid config type: {type(config)}")
        return cls(**cfg)

    def encode(self, x):
        h = self.encoder(x)
        moments = self.quant_conv(h)
        posterior = DiagonalGaussianDistribution(moments)
        return posterior

    def decode(self, z):
        z = self.post_quant_conv(z)
        dec = self.decoder(z)
        return dec

    def forward(self, input, sample_posterior=True):
        posterior = self.encode(input)
        if sample_posterior:
            z = posterior.sample()
        else:
            z = posterior.mode()
        dec = self.decode(z)
        return dec, posterior

    def get_input(self, batch, k):
        x = batch[k]
        if len(x.shape) == 3:
            x = x[..., None]
        x = x.permute(0, 3, 1, 2).to(memory_format=torch.contiguous_format).float()
        return x

    def get_last_layer(self):
        return self.decoder.conv_out.weight