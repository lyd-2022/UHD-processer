import torch
# import pytorch_lightning as pl # type: ignore
import torch.nn.functional as F
from contextlib import contextmanager
import torch.nn as nn
import sys
sys.path.append("/code/UHD-allinone")

# from taming.modules.vqvae.quantize import VectorQuantizer2 as VectorQuantizer

# from basicsr.archs.encoder_3 import Encoder, Decoder
from basicsr.utils.distributions.distributions import DiagonalGaussianDistribution

from basicsr.utils.vae_util import instantiate_from_config
from basicsr.utils.registry import ARCH_REGISTRY
from basicsr.archs.MAB import MAB
from basicsr.archs.Fourier_Upsampling import freup_Areadinterpolation,freup_AreadinterpolationV2,freup_Cornerdinterpolation,freup_Periodicpadding
from basicsr.archs.encoder_3 import nonlinearity, Normalize, ResnetBlock, Downsample, Upsample,fresadd,AttnBlock,LinAttnBlock
# from basicsr.archs.encoder import make_attn
import numpy as np


from basicsr.utils.modules.attention import LinearAttention
from basicsr.archs.fourmer import ProcessBlock
from basicsr.archs.restormer import TransformerBlock

from basicsr.archs.Resblock.Res_four import Res_four,Res_four2,Res_four3,Res_four4,Res_four5,Res_four6,Res_four7,Res_four8,Res_four9,Res_four10,Res_four11,Res_four12
   

# from basicsr.utils.registry import ARCH_REGISTRY



class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(DepthwiseSeparableConv, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, stride=stride, padding=padding, groups=in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        
    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x
    

    

def make_attn(in_channels, attn_type="vanilla"):
    assert attn_type in ["vanilla", "linear","Fourmer","restormer", "none"], f'attn_type {attn_type} unknown'
    print(f"making attention of type '{attn_type}' with {in_channels} in_channels")
    if attn_type == "vanilla":
        return AttnBlock(in_channels)
    elif attn_type == "none":
        return nn.Identity(in_channels)
    elif attn_type == "Fourmer":
        return ProcessBlock(in_channels)
    elif attn_type == "restormer":
        return TransformerBlock(in_channels,num_heads=4)
    else:
        return LinAttnBlock(in_channels)



def make_res(in_channels, out_channels,temb_channels,dropout, res_type="vanilla"):
    assert res_type in ["vanilla", "Fourmer","MAB","Res_four","Res_four2","Res_four3","Res_four4","Res_four5","Res_four6","Res_four7","Res_four8","Res_four9","Res_four10","Res_four11","Res_four12","none"], f'res_type {res_type} unknown'
    print(f"making res of type '{res_type}' with {in_channels} in_channels")
    if res_type == "vanilla":
        return ResnetBlock(in_channels=in_channels,
                                         out_channels=out_channels,
                                         temb_channels=temb_channels,
                                         dropout=dropout)
    elif res_type == "Fourmer":
        return ProcessBlock(in_channels,out_channels)
    elif res_type == "Res_four":
        return Res_four(in_channels=in_channels,
                                         out_channels=out_channels,
                                         dropout=dropout)
    elif res_type == "Res_four2":
        return Res_four2(in_channels=in_channels,
                                         out_channels=out_channels,
                                         dropout=dropout)
    elif res_type == "Res_four3":
        return Res_four3(in_channels=in_channels,
                                         out_channels=out_channels,
                                         dropout=dropout)
    elif res_type == "Res_four4":
        return Res_four4(in_channels=in_channels,
                                         out_channels=out_channels,
                                         dropout=dropout)
    elif res_type == "Res_four5":
        return Res_four5(in_channels=in_channels,
                                         out_channels=out_channels,
                                         dropout=dropout)
    elif res_type == "Res_four6":
        return Res_four6(in_channels=in_channels,
                                         out_channels=out_channels,
                                         dropout=dropout)
    elif res_type == "Res_four7":
        return Res_four7(in_channels=in_channels,
                                         out_channels=out_channels,
                                         dropout=dropout)
    elif res_type == "Res_four8":
        return Res_four8(in_channels=in_channels,
                                         out_channels=out_channels,
                                         dropout=dropout)
    elif res_type == "Res_four9":
        return Res_four9(in_channels=in_channels,
                                         out_channels=out_channels,
                                         dropout=dropout)

    elif res_type == "Res_four10":
        return Res_four10(in_channels=in_channels,
                                            out_channels=out_channels,
                                            dropout=dropout)
    elif res_type == "Res_four11":
        return Res_four11(in_channels=in_channels,
                                            out_channels=out_channels,
                                            dropout=dropout)
    elif res_type == "Res_four12":
        return Res_four12(in_channels=in_channels,
                                            out_channels=out_channels,
                                            dropout=dropout)
    elif res_type == "MAB":
        return MAB(in_channels, out_channels)

    else:
        return nn.Identity(in_channels)

@ARCH_REGISTRY.register()
class RAVAE(nn.Module):
    def __init__(self,
                 ddconfig,    
                 embed_dim,
                 optim,
                 ckpt_path=None,
                 ignore_keys=[],
                 image_key="image",
                 colorize_nlabels=None,
                 monitor=None,
                 sample_posterior=True,
                 ):
        super().__init__()
        self.image_key = image_key
        self.sample_posterior = sample_posterior
        self.encoder = Encoder(**ddconfig)
        self.decoder = Decoder(**ddconfig)
        # self.loss = instantiate_from_config(lossconfig)
        self.learning_rate = optim["lr"]
        assert ddconfig["double_z"]
        self.quant_conv = torch.nn.Conv2d(2*ddconfig["z_channels"], 2*embed_dim, 1)
        self.post_quant_conv = torch.nn.Conv2d(embed_dim, ddconfig["z_channels"], 1)
        self.embed_dim = embed_dim
        if colorize_nlabels is not None:
            assert type(colorize_nlabels)==int
            self.register_buffer("colorize", torch.randn(3, colorize_nlabels, 1, 1))
        if monitor is not None:
            self.monitor = monitor
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)

    def init_from_ckpt(self, path, ignore_keys=list()):
        sd = torch.load(path, map_location="cpu")["state_dict"]
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        self.load_state_dict(sd, strict=False)
        print(f"Restored from {path}")

    def encode(self, x,sample_posterior):
        h = self.encoder(x)
        moments = self.quant_conv(h)
        posterior = DiagonalGaussianDistribution(moments)
        if sample_posterior:
            z = posterior.sample()
        else:
            z = posterior.mode()
        return z,posterior

    def decode(self, z):
        z = self.post_quant_conv(z)
        dec = self.decoder(z)
        return dec

    def forward(self, input):
        #采样过程合并到encode中
        z,posterior = self.encode(input,self.sample_posterior)
        dec = self.decode(z)
        return dec, posterior

    def get_input(self, batch, k):
        # x = batch[k]
        x = batch
        # print(x.shape)
        if len(x.shape) == 3:
            x = x[..., None]
        # x = x.permute(0, 3, 1, 2).to(memory_format=torch.contiguous_format).float()
        return x
        
        

        
    def configure_optimizers(self):
        lr = self.learning_rate
        opt_ae = torch.optim.Adam(list(self.encoder.parameters())+
                                  list(self.decoder.parameters())+
                                  list(self.quant_conv.parameters())+
                                  list(self.post_quant_conv.parameters()),
                                  lr=lr, betas=(0.5, 0.9))
        opt_disc = torch.optim.Adam(self.loss.discriminator.parameters(),
                                    lr=lr, betas=(0.5, 0.9))
        return [opt_ae, opt_disc], []

    def get_last_layer(self):
        return self.decoder.conv_out.weight

    @torch.no_grad()
    def log_images(self, batch, only_inputs=False, **kwargs):
        log = dict()
        x = self.get_input(batch, self.image_key)
        x = x.to(self.device)
        if not only_inputs:
            xrec, posterior = self(x)
            if x.shape[1] > 3:
                # colorize with random projection
                assert xrec.shape[1] > 3
                x = self.to_rgb(x)
                xrec = self.to_rgb(xrec)
            log["samples"] = self.decode(torch.randn_like(posterior.sample()))
            log["reconstructions"] = xrec
        log["inputs"] = x
        return log

    def to_rgb(self, x):
        assert self.image_key == "segmentation"
        if not hasattr(self, "colorize"):
            self.register_buffer("colorize", torch.randn(3, x.shape[1], 1, 1).to(x))
        x = F.conv2d(x, weight=self.colorize)
        x = 2.*(x-x.min())/(x.max()-x.min()) - 1.
        return x

class Decoder(nn.Module):
    def __init__(self, *, ch, out_ch, ch_mult=(1,2,4,8), num_res_blocks,
                 attn_resolutions, dropout=0.0, freup_type='pad', in_channels,
                 resolution, z_channels, give_pre_end=False, tanh_out=False, use_linear_attn=False,
                 attn_type="vanilla",res_type="vanilla", **ignorekwargs):
        super().__init__()
        if use_linear_attn: attn_type = "linear"
        self.ch = ch
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels
        self.give_pre_end = give_pre_end
        self.tanh_out = tanh_out

        # compute in_ch_mult, block_in and curr_res at lowest res
        in_ch_mult = (1,)+tuple(ch_mult)
        block_in = ch*ch_mult[self.num_resolutions-1]
        curr_res = resolution // 2**(self.num_resolutions-1)
        self.z_shape = (1,z_channels,curr_res,curr_res)
        print("Working with z of shape {} = {} dimensions.".format(
            self.z_shape, np.prod(self.z_shape)))

        # z to block_in
        self.conv_in = torch.nn.Conv2d(z_channels,
                                       block_in,
                                       kernel_size=3,
                                       stride=1,
                                       padding=1)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = make_res(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout,res_type=res_type)
        self.mid.attn_1 = make_attn(block_in, attn_type=attn_type)
        self.mid.block_2 = make_res(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout,res_type=res_type)

        # upsampling
        self.up = nn.ModuleList()
        ch_mult = (1,) + tuple(ch_mult)
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            
            block_out = ch*ch_mult[i_level-1]
            for i_block in range(self.num_res_blocks+1):
                block.append(make_res(in_channels=block_in,
                                         out_channels=block_out,
                                         temb_channels=self.temb_ch,
                                         dropout=dropout,res_type=res_type))
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(make_attn(block_in, attn_type=attn_type))
            up = nn.Module()
            up.block = block
            up.attn = attn
            if i_level != 0:
                up.upsample = fresadd(block_in, freup_type)
                curr_res = curr_res * 2
            self.up.insert(0, up) # prepend to get consistent order

        # end
        self.norm_out = Normalize(block_in)
        self.conv_out = torch.nn.Conv2d(block_in,
                                        out_ch,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)
        self.shuffler = nn.PixelShuffle(2)
    def forward(self, z):
        #assert z.shape[1:] == self.z_shape[1:]
        self.last_z_shape = z.shape

        # timestep embedding
        temb = None

        # z to block_in
        h = self.conv_in(z)

        # middle
        h = self.mid.block_1(h)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h)

        # upsampling
        for i_level in reversed(range(self.num_resolutions)):
            h1 = self.shuffler(h)
            h1 = torch.cat([h1, h1], dim=1)
            for i_block in range(self.num_res_blocks+1):
                h = self.up[i_level].block[i_block](h)
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h)
            if i_level != 0:
                h = self.up[i_level].upsample(h)+h1
                del h1

        # end
        if self.give_pre_end:
            return h

        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        if self.tanh_out:
            h = torch.tanh(h)
        return h
    

    
class Encoder(nn.Module):
    def __init__(self, *, ch, out_ch, ch_mult=(1,2,4,8), num_res_blocks,
                 attn_resolutions, dropout=0.0, resamp_with_conv=True, in_channels,
                 resolution, z_channels, double_z=True, use_linear_attn=False, attn_type="vanilla",res_type="vanilla",
                 **ignore_kwargs):
        super().__init__()
        if use_linear_attn: attn_type = "linear"
        self.ch = ch
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels

        # downsampling
        self.conv_in = torch.nn.Conv2d(in_channels,
                                       self.ch,
                                       kernel_size=3,
                                       stride=1,
                                       padding=1)

        curr_res = resolution
        in_ch_mult = (1,) + tuple(ch_mult)
        self.in_ch_mult = in_ch_mult
        self.down = nn.ModuleList()
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = ch * in_ch_mult[i_level]
            block_out = ch * ch_mult[i_level]
            for i_block in range(self.num_res_blocks):
                block.append(make_res(in_channels=block_in,
                                         out_channels=block_out,
                                         temb_channels=self.temb_ch,
                                         dropout=dropout,res_type=res_type))
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(make_attn(block_in, attn_type=attn_type))
            down = nn.Module()
            down.block = block
            down.attn = attn

            if i_level != self.num_resolutions - 1:
                down.downsample = Downsample(block_in, resamp_with_conv)
                curr_res = curr_res // 2
            self.down.append(down)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = make_res(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout,res_type=res_type)
        self.mid.attn_1 = make_attn(block_in, attn_type=attn_type)
        self.mid.block_2 = make_res(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout,res_type=res_type)

        # end
        self.norm_out = Normalize(block_in)
        self.conv_out = torch.nn.Conv2d(block_in,
                                        2 * z_channels if double_z else z_channels,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)
        self.shuffler = nn.PixelUnshuffle(2)

    def forward(self, x):
        # timestep embedding
        temb = None

        # downsampling
        hs = [self.conv_in(x)]
        for i_level in range(self.num_resolutions):
            h1 = self.shuffler(hs[-1])
            h10 ,h11 = h1.chunk(2,dim=1)
            h1 = torch.mean(torch.stack([h10, h11]), dim=0)
            del h10,h11
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](hs[-1])
                
                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h)
                hs.append(h)
            if i_level != self.num_resolutions - 1:
                hs.append(self.down[i_level].downsample(hs[-1])+h1)
                del h1
        # middle
        h = hs[-1]
        del hs

        h = self.mid.block_1(h)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h)

        # end
        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        return h

if __name__ == '__main__':
    import yaml
    with open("/code/UHD-allinone/options/all-in-one/VAE/8_16_RA.yml") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)["network_g"]
    config.pop('type')
    # config['ddconfig']['res_type'] = "Res_four2"
    model = RAVAE(**config).cuda()
    para_num = sum([param.nelement() for param in model.parameters() if param.requires_grad])
    print(f"para_num:{para_num}")
    from thop import profile
    from thop import clever_format
    input = torch.randn(1,3,256,256).cuda()
    flops,params = profile(model,inputs=(input,))
    flops,params = clever_format([flops,params], "%.3f")
    print(f"params:{params},flops:{flops}")
    output = model(input)