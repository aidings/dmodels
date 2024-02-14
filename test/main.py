import torch
from dmodels.autoencoderkl import AutoencoderKL
from dmodels.unet import SdxlUNet2DConditionModel
from dmodels import get_cfg



def vae():
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

def unet():
    cfg = get_cfg("sdxl_v1.yaml")
    batch_size = 1
    x = torch.randn(batch_size, 4, 128, 128)  # 1024x1024
    t = torch.randint(low=0, high=10, size=(batch_size,))
    ctx = torch.randn(batch_size, 77, 2048)
    y = torch.randn(batch_size, cfg["adm_in_channels"])
    model = SdxlUNet2DConditionModel.from_config()
    model.eval()
    with torch.no_grad():
        output = model(x, t, ctx, y)
        print(output.shape) 
        
if __name__ == '__main__':
    unet()