import torch
from dmodels.autoencoderkl import AutoencoderKL
from dmodels import get_cfg


if __name__ == '__main__':
    config = get_cfg("kl_f8.yaml")
    print(config)
    config['ddconfig']['z_channels'] = 8
    # config['ddconfig']['resolution'] = 1024
    #"""
    model = AutoencoderKL.from_config(config)
    model.eval()

    image = torch.randn(1, 3, 512, 512)
    with torch.no_grad():
        output, posterior = model(image)
        print(output.shape)
    #"""