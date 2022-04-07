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

torch.hub.download_url_to_file('http://batbot.ai/models/latent-diffusion/models/ldm/text2img-large/model.ckpt','txt2img-f8-large.ckpt')

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
from open_clip import tokenizer
import open_clip

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
#NSFW CLIP Filter
clip_model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32-quickgelu', pretrained='laion400m_e32')
text = tokenizer.tokenize(["NSFW", "adult content", "porn", "naked people","genitalia","penis","vagina"])
with torch.no_grad():
  text_features = clip_model.encode_text(text)

def run(prompt, steps, width, height, images, scale):
    opt = argparse.Namespace(
        prompt = prompt, 
        outdir='latent-diffusion/outputs',
        ddim_steps = int(steps),
        ddim_eta = 0,
        n_iter = 1,
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
                        image_vector = Image.fromarray(x_sample.astype(np.uint8))
                        image = preprocess(image_vector).unsqueeze(0)
                        image_features = clip_model.encode_image(image)
                        sims = image_features @ text_features.T
                        if(sims.max()<18):
                            all_samples_images.append(image_vector)
                        else:
                            return(None,None,"Sorry, NSFW content was detected on your outputs. Try again with different prompts. If you feel your prompt was not supposed to give NSFW outputs, this may be due to a bias in the model. Read more about biases in the Biases Acknowledgment section below.")
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
    return(Image.fromarray(grid.astype(np.uint8)),all_samples_images,None)

image = gr.outputs.Image(type="pil", label="Your result")
css = ".output-image{height: 528px !important} .output-carousel .output-image{height:272px !important} a{text-decoration: underline}"
iface = gr.Interface(fn=run, inputs=[
    gr.inputs.Textbox(label="Prompt - try adding increments to your prompt such as 'oil on canvas', 'a painting', 'a book cover'",default="chalk pastel drawing of a dog wearing a funny hat"),
    gr.inputs.Slider(label="Steps - more steps can increase quality but will take longer to generate",default=45,maximum=50,minimum=1,step=1),
    gr.inputs.Radio(label="Width", choices=[32,64,128,256],default=256),
    gr.inputs.Radio(label="Height", choices=[32,64,128,256],default=256),
    gr.inputs.Slider(label="Images - How many images you wish to generate", default=2, step=1, minimum=1, maximum=4),
    gr.inputs.Slider(label="Diversity scale - How different from one another you wish the images to be",default=5.0, minimum=1.0, maximum=15.0),
    #gr.inputs.Slider(label="ETA - between 0 and 1. Lower values can provide better quality, higher values can be more diverse",default=0.0,minimum=0.0, maximum=1.0,step=0.1),
    ], 
    outputs=[image,gr.outputs.Carousel(label="Individual images",components=["image"]),gr.outputs.Textbox(label="Error")],
    css=css,
    title="Generate images from text with Latent Diffusion LAION-400M",
    description="<div>By typing a prompt and pressing submit you can generate images based on this prompt. <a href='https://github.com/CompVis/latent-diffusion' target='_blank'>Latent Diffusion</a> is a text-to-image model created by <a href='https://github.com/CompVis' target='_blank'>CompVis</a>, trained on the <a href='https://laion.ai/laion-400-open-dataset/'>LAION-400M dataset.</a><br>This UI to the model was assembled by <a style='color: rgb(245, 158, 11);font-weight:bold' href='https://twitter.com/multimodalart' target='_blank'>@multimodalart</a></div>",
    article="<h4 style='font-size: 110%;margin-top:.5em'>Biases acknowledgment</h4><div>Despite how impressive being able to turn text into image is, beware to the fact that this model may output content that reinforces or exarcbates societal biases. According to the <a href='https://arxiv.org/abs/2112.10752' target='_blank'>Latent Diffusion paper</a>:<i> \"Deep learning modules tend to reproduce or exacerbate biases that are already present in the data\"</i>. The model was trained on an unfiltered version the LAION-400M dataset, which scrapped non-curated image-text-pairs from the internet (the exception being the the removal of illegal content) and is meant to be used for research purposes, such as this one. <a href='https://laion.ai/laion-400-open-dataset/' target='_blank'>You can read more on LAION's website</a></div><h4 style='font-size: 110%;margin-top:1em'>Who owns the images produced by this demo?</h4><div>Definetly not me! Probably you do. I say probably because the Copyright discussion about AI generated art is ongoing. So <a href='https://www.theverge.com/2022/2/21/22944335/us-copyright-office-reject-ai-generated-art-recent-entrance-to-paradise' target='_blank'>it may be the case that everything produced here falls automatically into the public domain</a>. But in any case it is either yours or is in the public domain.</div>")
iface.launch(enable_queue=True)