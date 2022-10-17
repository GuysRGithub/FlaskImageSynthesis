from base64 import encodebytes
from io import BytesIO
from pydoc import describe
import gradio as gr
import torch
from omegaconf import OmegaConf
from flask import Flask, jsonify, request, make_response, send_file
import sys

sys.path.append(".")
sys.path.append('./taming-transformers')
sys.path.append('./latent-diffusion')
from taming.models import vqgan
from ldm.util import instantiate_from_config
from huggingface_hub import hf_hub_download

model_path_e = hf_hub_download(repo_id="multimodalart/compvis-latent-diffusion-text2img-large",
                               filename="txt2img-f8-large.ckpt")

# @title Import stuff
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


def load_safety_model(clip_model):
    """load the safety model"""
    import autokeras as ak  # pylint: disable=import-outside-toplevel
    from tensorflow.keras.models import load_model  # pylint: disable=import-outside-toplevel
    from os.path import expanduser  # pylint: disable=import-outside-toplevel

    home = expanduser("~")

    cache_folder = home + "/.cache/clip_retrieval/" + clip_model.replace("/", "_")
    if clip_model == "ViT-L/14":
        model_dir = cache_folder + "/clip_autokeras_binary_nsfw"
        dim = 768
    elif clip_model == "ViT-B/32":
        model_dir = cache_folder + "/clip_autokeras_nsfw_b32"
        dim = 512
    else:
        raise ValueError("Unknown clip model")
    if not os.path.exists(model_dir):
        os.makedirs(cache_folder, exist_ok=True)

        from urllib.request import urlretrieve  # pylint: disable=import-outside-toplevel

        path_to_zip_file = cache_folder + "/clip_autokeras_binary_nsfw.zip"
        if clip_model == "ViT-L/14":
            url_model = "https://raw.githubusercontent.com/LAION-AI/CLIP-based-NSFW-Detector/main/clip_autokeras_binary_nsfw.zip"
        elif clip_model == "ViT-B/32":
            url_model = (
                "https://raw.githubusercontent.com/LAION-AI/CLIP-based-NSFW-Detector/main/clip_autokeras_nsfw_b32.zip"
            )
        else:
            raise ValueError("Unknown model {}".format(clip_model))
        urlretrieve(url_model, path_to_zip_file)
        import zipfile  # pylint: disable=import-outside-toplevel

        with zipfile.ZipFile(path_to_zip_file, "r") as zip_ref:
            zip_ref.extractall(cache_folder)

    loaded_model = load_model(model_dir, custom_objects=ak.CUSTOM_OBJECTS)
    loaded_model.predict(np.random.rand(10 ** 3, dim).astype("float32"), batch_size=10 ** 3)

    return loaded_model


def is_unsafe(safety_model, embeddings, threshold=0.5):
    """find unsafe embeddings"""
    nsfw_values = safety_model.predict(embeddings, batch_size=embeddings.shape[0])
    x = np.array([e[0] for e in nsfw_values])
    return True if x > threshold else False


config = OmegaConf.load("latent-diffusion/configs/latent-diffusion/txt2img-1p4B-eval.yaml")
model = load_model_from_config(config, model_path_e)
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model = model.to(device)

# NSFW CLIP Filter
safety_model = load_safety_model("ViT-B/32")
clip_model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='openai')


def run(prompt, steps, width, height, images, scale):
    opt = argparse.Namespace(
        prompt=prompt,
        outdir='latent-diffusion/outputs',
        ddim_steps=int(steps),
        ddim_eta=0,
        n_iter=1,
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

    all_samples = list()
    all_samples_images = list()
    with torch.no_grad():
        with torch.cuda.amp.autocast():
            with model.ema_scope():
                uc = None
                if opt.scale > 0:
                    uc = model.get_learned_conditioning(opt.n_samples * [""])
                for n in range(opt.n_iter):
                    c = model.get_learned_conditioning(opt.n_samples * [prompt])
                    shape = [4, opt.H // 8, opt.W // 8]
                    samples_ddim, _ = sampler.sample(S=opt.ddim_steps,
                                                     conditioning=c,
                                                     batch_size=opt.n_samples,
                                                     shape=shape,
                                                     verbose=False,
                                                     unconditional_guidance_scale=opt.scale,
                                                     unconditional_conditioning=uc,
                                                     eta=opt.ddim_eta)

                    x_samples_ddim = model.decode_first_stage(samples_ddim)
                    x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)

                    for x_sample in x_samples_ddim:
                        x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
                        image_vector = Image.fromarray(x_sample.astype(np.uint8))
                        image_preprocess = preprocess(image_vector).unsqueeze(0)
                        with torch.no_grad():
                            image_features = clip_model.encode_image(image_preprocess)
                        image_features /= image_features.norm(dim=-1, keepdim=True)
                        query = image_features.cpu().detach().numpy().astype("float32")
                        unsafe = is_unsafe(safety_model, query, 0.5)
                        if (not unsafe):
                            all_samples_images.append(image_vector)
                        else:
                            return (None, None,
                                    "Sorry, potential NSFW content was detected on your outputs by our NSFW detection model. Try again with different prompts. If you feel your prompt was not supposed to give NSFW outputs, this may be due to a bias in the model. Read more about biases in the Biases Acknowledgment section below.")
                        # Image.fromarray(x_sample.astype(np.uint8)).save(os.path.join(sample_path, f"{base_count:04}.png"))
                        base_count += 1
                    all_samples.append(x_samples_ddim)

    # additionally, save as grid
    grid = torch.stack(all_samples, 0)
    grid = rearrange(grid, 'n b c h w -> (n b) c h w')
    grid = make_grid(grid, nrow=2)
    # to image
    grid = 255. * rearrange(grid, 'c h w -> h w c').cpu().numpy()

    Image.fromarray(grid.astype(np.uint8)).save(os.path.join(outpath, f'{prompt.replace(" ", "-")}.png'))
    return Image.fromarray(grid.astype(np.uint8)), all_samples_images, None


app = Flask(__name__)


# from flask_ngrok import run_with_ngrok
# run_with_ngrok(app)

@app.route('/')
def index():
    return "Welcome to Golab REST API"


def serve_pil_image(pil_img):
    img_io = BytesIO()
    pil_img.save(img_io, 'JPEG', quality=70)
    img_io.seek(0)
    return send_file(img_io, mimetype='image/jpeg')


def encoded_image(pil_img):
    img_io = BytesIO()
    pil_img.save(img_io, 'JPEG', quality=70)
    encoded_img = encodebytes(img_io.getvalue()).decode('ascii')  # encode as base64
    return encoded_img


@app.route('/api', methods=['POST'])
def api():
    data = request.json
    prompt = data.get('prompt', 'chalk pastel drawing of a dog wearing a funny hat')
    steps = data.get('steps', 45)
    width = data.get('width', 256)
    height = data.get('height', 256)
    images = data.get('num', 2)
    scale = data.get('scale', 5)

    image, samples, err = run(prompt, steps, width, height, images, scale)
    encoded_images = []
    for sample in samples:
        encoded_images.append(encoded_image(sample))

    return jsonify({'image': encoded_image(image), 'carousel': encoded_images, 'error': err})


if __name__ == '__main__':
    app.run()
