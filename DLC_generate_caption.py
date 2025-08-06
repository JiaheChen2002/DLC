import argparse
import os
import random

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from tqdm import tqdm
from datetime import datetime

from minigpt4.common.dist_utils import get_rank
from minigpt4.models import load_preprocess

from minigpt4.common.config import Config
from minigpt4.common.dist_utils import get_rank
from minigpt4.common.registry import registry

from minigpt4.datasets.builders import *
from minigpt4.models import *
from minigpt4.processors import *
from minigpt4.runners import *
from minigpt4.tasks import *
from DLC_logits_processor import DLCLogitsProcessor
from PIL import Image
import json
from pycocotools.coco import COCO
import warnings
from transformers import AutoTokenizer, LogitsProcessorList, AutoModel, AutoProcessor
warnings.filterwarnings("ignore")


MODEL_EVAL_CONFIG_PATH = {
    "minigpt4": "eval_configs/minigpt4_eval.yaml",
    "instructblip": "eval_configs/instructblip_eval.yaml",
    "instructblip-13b": "eval_configs/instructblip_eval_13b.yaml",
    "lrv_instruct": "eval_configs/lrv_instruct_eval.yaml",
    "shikra": "eval_configs/shikra_eval.yaml",
    "llava-1.5": "eval_configs/llava-1.5_eval.yaml",
    "llava-1.5-13b": "eval_configs/llava-1.5_eval_13b.yaml",
}

INSTRUCTION_TEMPLATE = {
    "minigpt4": "###Human: <Img><ImageHere></Img> <question> ###Assistant:",
    "instructblip": "<ImageHere><question>",
    "lrv_instruct": "###Human: <Img><ImageHere></Img> <question> ###Assistant:",
    "shikra": "USER: <im_start><ImageHere><im_end> <question> ASSISTANT:",
    "llava-1.5": "USER: <ImageHere> <question> ASSISTANT:"
}


def parse_args():
    parser = argparse.ArgumentParser(description="Generate captions")
    parser.add_argument("--model", type=str, default="llava-1.5", help="model")
    parser.add_argument("--gpu-id", type=int, default=3, help="specify the gpu to load the model.")
    parser.add_argument("--data-path", type=str, default="", help="data path")
    parser.add_argument("--output-dir", type=str, default="./results", help="output directory")
    parser.add_argument("--clip-model-path", type=str, default="google/siglip-so400m-patch14-384", help="CLIP model path")
    parser.add_argument("--tokenizer-path", type=str, default="llava-hf/llava-1.5-7b", help="tokenizer path")
    parser.add_argument("--num-samples", type=int, default=500, help="number of images to sample")
    parser.add_argument("--batch-size", type=int, default=1, help="batch size")
    parser.add_argument("--num_workers", type=int, default=1, help="num workers")
    parser.add_argument("--max-new-tokens", type=int, default=512, help="Maximum length of generated text")
    parser.add_argument("--window_size", type=int, default=8, help="Window size for visual penalty processor")
    parser.add_argument("--penalty_scale", type=float, default=1.0, help="Penalty scale for visual penalty processor")
    parser.add_argument("--visual_top_k", type=int, default=50, help="Top k for visual penalty processor")    
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default=None)
    parser.add_argument("--sample", action="store_true", default=False, help="Whether to sample or not")
    parser.add_argument("--use_dlc", action="store_true", default=False, help="Whether to use DLC logits processor")
    parser.add_argument("--options")

    args = parser.parse_args()
    return args


def setup_seeds(config):
    seed = config.run_cfg.seed + get_rank()

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    cudnn.benchmark = False
    cudnn.deterministic = True

def create_output_filename(args):
    time_str = datetime.now().strftime('%m%d_%H%M')
    filename = (f'{time_str}_dlc_dosample-{args.sample}_ws{args.window_size}_'
                f'penscale{args.penalty_scale}_p{args.top_p}_k{args.top_k}_'
                f'max{args.max_new_tokens}_samples{args.num_samples}_caption.jsonl')
    return filename

def setup_output_directory(args):
    base_dir = os.path.join(args.output_dir, args.model)
    os.makedirs(base_dir, exist_ok=True)
    return base_dir

def load_models(args, device):
    print("Loading CLIP model and tokenizer...")
    clip_model = AutoModel.from_pretrained(args.clip_model_path).to(device)
    clip_processor = AutoProcessor.from_pretrained(args.clip_model_path)

    if args.model == "llava-1.5":
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, use_fast=False)
    # else:
        # tokenizer = AutoTokenizer.from_pretrained("hf_models/vicuna-7b-v1.1", use_fast=False, truncation_side="left")
        # tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        # tokenizer.add_special_tokens({'bos_token': '</s>'})
        # tokenizer.add_special_tokens({'eos_token': '</s>'})
        # tokenizer.add_special_tokens({'unk_token': '</s>'})
    return clip_model, clip_processor, tokenizer

def main():
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    
    args.cfg_path = MODEL_EVAL_CONFIG_PATH[args.model]
    cfg = Config(args)

    setup_seeds(cfg)
    device = torch.device("cuda") if torch.cuda.is_available() else "cpu"

    clip_model, clip_processor, tokenizer = load_models(args, device)

    print('Initializing Model')

    model_config = cfg.model_cfg
    model_config.device_8bit = args.gpu_id
    model_cls = registry.get_model_class(model_config.arch)
    model = model_cls.from_config(model_config).to(device)
    model.eval()

    vis_processors, txt_processors = load_preprocess(cfg.get_config().preprocess)
    print(vis_processors["eval"].transform)
    print("Done!")

    caption_file = os.path.join(args.data_path, "annotations/captions_val2014.json")
    coco = COCO(caption_file)
    all_img_ids = coco.getImgIds()
    img_ids = random.sample(all_img_ids, 500)

    base_dir = setup_output_directory(args)
    output_file = create_output_filename(args)
    output_path = os.path.join(base_dir, output_file)

    print(f"Processing {len(img_ids)} images...")
    print(f"Results will be saved to: {output_path}")

    for img_id in tqdm(img_ids, desc="Processing images"):
        img_info = coco.loadImgs(img_id)[0]
        image_path = os.path.join(args.data_path, img_info["file_name"])
        
        img_save = {}
        img_save["image_id"] = img_id
        
        raw_image = Image.open(image_path).convert("RGB")
        image = vis_processors["eval"](raw_image).unsqueeze(0)
        image = image.to(device)
        
        clip_inputs = clip_processor(images=raw_image, return_tensors="pt").to(device)
        image_features = clip_model.get_image_features(**clip_inputs)

        qu = "Please describe this image in detail."
        template = INSTRUCTION_TEMPLATE[args.model]
        qu = template.replace("<question>", qu)

        if args.use_dlc:
            visual_penalizer = DLCLogitsProcessor(
                clip_model=clip_model,
                clip_processor=clip_processor,
                image_features=image_features,
                tokenizer=tokenizer,
                window_size=args.window_size,
                penalty_scale=args.penalty_scale,
                top_k=args.visual_top_k
            )
            logits_processor = LogitsProcessorList([visual_penalizer])

            with torch.inference_mode():
                with torch.no_grad():
                    out = model.generate(
                        prompt=[qu],
                        image=image.half(),
                        temperature=args.temperature,
                        max_new_tokens=args.max_new_tokens,
                        logits_processor=logits_processor,
                        top_p=args.top_p,
                        top_k=args.top_k,
                        do_sample=args.sample,
                    )
        else:
            with torch.inference_mode():
                with torch.no_grad():
                    out = model.generate(
                        prompt=[qu],
                        image=image.half(),
                        temperature=args.temperature,
                        max_new_tokens=args.max_new_tokens,
                        # logits_processor=logits_processor,
                        top_p=args.top_p,
                        top_k=args.top_k,
                        do_sample=args.sample,
                    )
        img_save["caption"] = out[0]

        with open(os.path.join(base_dir, output_file), "a") as f:
            json.dump(img_save, f)
            f.write('\n')

if __name__ == "__main__":
    main()
