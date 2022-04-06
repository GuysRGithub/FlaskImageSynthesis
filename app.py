from pydoc import describe
import gradio as gr
import torch
from omegaconf import OmegaConf
import sys 
sys.path.append(".")
sys.path.append('./taming-transformers')
sys.path.append('./latent-diffusion')
from taming.models import vqgan 
from ldm.util import instantiate_from_config

#torch.hub.download_url_to_file('https://ommer-lab.com/files/latent-diffusion/nitro/txt2img-f8-large/model.ckpt','txt2img-f8-large.ckpt')

#@title Import stuff
import argparse, os, sys, glob
import numpy as np
from PIL import Image
from einops import rearrange
from torchvision.utils import make_grid
import transformers
import gc
from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler


def load_model_from_config(config, ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cuda")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    model = model.half().cuda()
    model.eval()
    return model

config = OmegaConf.load("latent-diffusion/configs/latent-diffusion/txt2img-1p4B-eval.yaml")
model = load_model_from_config(config, f"txt2img-f8-large.ckpt")
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model = model.to(device)

def run(prompt, steps, width, height, images, scale, eta):
    if images == 6:
        images = 3
        n_iter = 2
    else:
        n_iter = 1
    opt = argparse.Namespace(
        prompt = prompt, 
        outdir='latent-diffusion/outputs',
        ddim_steps = int(steps),
        ddim_eta = eta,
        n_iter = n_iter,
        W=int(width),
        H=int(height),
        n_samples=int(images),
        scale=scale,
        plms=True
    )
    if opt.plms:
        opt.ddim_eta = 0
        sampler = PLMSSampler(model)
    else:
        sampler = DDIMSampler(model)
    
    os.makedirs(opt.outdir, exist_ok=True)
    outpath = opt.outdir

    prompt = opt.prompt


    sample_path = os.path.join(outpath, "samples")
    os.makedirs(sample_path, exist_ok=True)
    base_count = len(os.listdir(sample_path))

    all_samples=list()
    all_samples_images=list()
    with torch.no_grad():
        with torch.cuda.amp.autocast():
            with model.ema_scope():
                uc = None
                if opt.scale > 0:
                    uc = model.get_learned_conditioning(opt.n_samples * [""])
                for n in range(opt.n_iter):
                    c = model.get_learned_conditioning(opt.n_samples * [prompt])
                    shape = [4, opt.H//8, opt.W//8]
                    samples_ddim, _ = sampler.sample(S=opt.ddim_steps,
                                                    conditioning=c,
                                                    batch_size=opt.n_samples,
                                                    shape=shape,
                                                    verbose=False,
                                                    unconditional_guidance_scale=opt.scale,
                                                    unconditional_conditioning=uc,
                                                    eta=opt.ddim_eta)

                    x_samples_ddim = model.decode_first_stage(samples_ddim)
                    x_samples_ddim = torch.clamp((x_samples_ddim+1.0)/2.0, min=0.0, max=1.0)

                    for x_sample in x_samples_ddim:
                        x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
                        all_samples_images.append(Image.fromarray(x_sample.astype(np.uint8)))
                        #Image.fromarray(x_sample.astype(np.uint8)).save(os.path.join(sample_path, f"{base_count:04}.png"))
                        base_count += 1
                    all_samples.append(x_samples_ddim)
                    
    
    # additionally, save as grid
    grid = torch.stack(all_samples, 0)
    grid = rearrange(grid, 'n b c h w -> (n b) c h w')
    grid = make_grid(grid, nrow=2)
    # to image
    grid = 255. * rearrange(grid, 'c h w -> h w c').cpu().numpy()
    
    Image.fromarray(grid.astype(np.uint8)).save(os.path.join(outpath, f'{prompt.replace(" ", "-")}.png'))
    return(Image.fromarray(grid.astype(np.uint8)),all_samples_images)

image = gr.outputs.Image(type="pil", label="Your result")
css = ".output-image{height: 528px !important} .output-carousel .output-image{height:272px !important}"
iface = gr.Interface(fn=run, inputs=[
    gr.inputs.Textbox(label="Prompt",default="A drawing of a cute dog with a funny hat"),
    gr.inputs.Slider(label="Steps - more steps can increase quality but will take longer to generate",default=50,maximum=250,minimum=1,step=1),
    gr.inputs.Slider(label="Width", minimum=64, maximum=256, default=256, step=64),
    gr.inputs.Slider(label="Height", minimum=64, maximum=256, default=256, step=64),
    gr.inputs.Slider(label="Images - How many images you wish to generate", default=4, step=2, minimum=2, maximum=6),
    gr.inputs.Slider(label="Diversity scale - How different from one another you wish the images to be",default=5.0, minimum=1),
    gr.inputs.Slider(label="ETA - between 0 and 1. Lower values can provide better quality, higher values can be more diverse",default=0.0,minimum=0.0, maximum=1.0,step=0.1),
    
    ], 
    outputs=[image,gr.outputs.Carousel(label="Individual images",components=["image"])],
    css=css,
    title="Generate images from text with Latent Diffusion LAION-400M",
    description="<div>By typing a text and clicking submit you can generate images based on this text. This is a text-to-image model created by CompVis, trained on the LAION-400M dataset.<br>For more multimodal ai art check us out <a style='color: rgb(245, 158, 11);font-weight:bold' href='https://twitter.com/multimodalart' target='_blank'>@multimodalart</a></div>")
iface.launch(enable_queue=True)