import argparse
import os
import random

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from tqdm import tqdm
from datetime import datetime

from pope_loader import POPEDataSet
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
import logging
from pathlib import Path
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

POPE_PATH = {
    "coco_random": "pope/coco/coco_pope_random.json",
    "coco_popular": "pope/coco/coco_pope_popular.json",
    "coco_adversarial": "pope/coco/coco_pope_adversarial.json",
    "gpa_random": "pope/gpa/gqa_pope_seem_random.json",
    "gpa_popular": "pope/gpa/gqa_pope_seem_popular.json",
    "gpa_adversarial": "pope/gpa/gqa_pope_seem_adversarial.json",
    "aokvqa_random": "pope/aokvqa/aokvqa_pope_seem_random.json",
    "aokvqa_popular": "pope/aokvqa/aokvqa_pope_seem_popular.json",
    "aokvqa_adversarial": "pope/aokvqa/aokvqa_pope_seem_adversarial.json",
}

INSTRUCTION_TEMPLATE = {
    "minigpt4": "###Human: <Img><ImageHere></Img> <question> ###Assistant:",
    "instructblip": "<ImageHere><question>",
    "lrv_instruct": "###Human: <Img><ImageHere></Img> <question> ###Assistant:",
    "shikra": "USER: <im_start><ImageHere><im_end> <question> ASSISTANT:",
    "llava-1.5": "USER: <ImageHere> <question> ASSISTANT:"
}


def parse_args():
    parser = argparse.ArgumentParser(description="POPE evaluation with DVP on LVLMs.")
    parser.add_argument("--model", type=str, default="llava-1.5", help="model")
    parser.add_argument("--pope-type", type=str, default="", help="pope dataset type")
    parser.add_argument("--gpu-id", type=int, default=6, help="specify the gpu to load the model.")
    parser.add_argument("--data-path", type=str, default="", help="data path")
    parser.add_argument("--batch-size", type=int, default=1, help="batch size")
    parser.add_argument("--num_workers", type=int, default=1, help="num workers")

    parser.add_argument("--max-new-tokens", type=int, default=10, help="Maximum length of generated text")
    parser.add_argument("--window_size", type=int, default=8, help="Window size for visual penalty processor")
    parser.add_argument("--penalty_scale", type=float, default=3.0, help="Penalty scale for visual penalty processor")
    parser.add_argument("--visual_top_k", type=int, default=50, help="Top k for visual penalty processor")

    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default=None)
    parser.add_argument("--sample", action="store_true", default=False, help="Whether to sample or not")
    parser.add_argument("--use_dlc", action="store_true", default=False, help="Whether to use dynamic logits correction")

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


def print_acc(pred_list, label_list, logger):
    pos = 1
    neg = 0
    yes_ratio = pred_list.count(1) / len(pred_list)

    TP, TN, FP, FN = 0, 0, 0, 0
    for pred, label in zip(pred_list, label_list):
        if pred == pos and label == pos:
            TP += 1
        elif pred == pos and label == neg:
            FP += 1
        elif pred == neg and label == neg:
            TN += 1
        elif pred == neg and label == pos:
            FN += 1

    print('TP\tFP\tTN\tFN\t')
    print('{}\t{}\t{}\t{}'.format(TP, FP, TN, FN))

    precision = float(TP) / float(TP + FP)
    recall = float(TP) / float(TP + FN)
    f1 = 2*precision*recall / (precision + recall)
    acc = (TP + TN) / (TP + TN + FP + FN)
    print('Accuracy: {}'.format(acc))
    print('Precision: {}'.format(precision))
    print('Recall: {}'.format(recall))
    print('F1 score: {}'.format(f1))
    print('Yes ratio: {}'.format(yes_ratio))
    logger.info(('Accuracy: {}'.format(acc)))
    logger.info(('Precision: {}'.format(precision)))
    logger.info(('Recall: {}'.format(recall)))
    logger.info(('F1 score: {}'.format(f1)))
    logger.info(('Yes ratio: {}'.format(yes_ratio)))


def recorder(out, pred_list):
    NEG_WORDS = ["No", "not", "no", "NO"]
    for line in out:
        line = line.replace('.', '')
        line = line.replace(',', '')
        words = line.split(' ')
        if any(word in NEG_WORDS for word in words) or any(word.endswith("n't") for word in words):
            pred_list.append(0)
        else:
            pred_list.append(1)
    
    return pred_list


def main():
    args = parse_args()

    def log_string(str):
        logger.info(str)
        print(str)
    
    exp_dir = Path(os.path.join('results', 'log'))
    exp_dir.mkdir(exist_ok=True)
    log_dir = exp_dir.joinpath(args.pope_type)
    log_dir.mkdir(exist_ok=True)
    logger = logging.getLogger(args.model)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('%s/log.txt' % log_dir)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    log_string('PARAMETER ...')
    log_string(args)
    
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    
    args.cfg_path = MODEL_EVAL_CONFIG_PATH[args.model]
    args.pope_path = POPE_PATH[args.pope_type]
    cfg = Config(args)

    setup_seeds(cfg)
    device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
    
    clip_model = AutoModel.from_pretrained("qihoo360/fg-clip-large").to(device)
    clip_processor = AutoProcessor.from_pretrained("qihoo360/fg-clip-large")

    tokenizer = AutoTokenizer.from_pretrained("liuhaotian/llava-v1.5-7b", use_fast=False)

    print('Initializing Model')

    model_config = cfg.model_cfg
    model_config.device_8bit = args.gpu_id
    model_cls = registry.get_model_class(model_config.arch)
    model = model_cls.from_config(model_config).to(device)
    model.eval()

    vis_processors, txt_processors = load_preprocess(cfg.get_config().preprocess)
    print(vis_processors["eval"].transform)
    print("Done!")

    pope_dataset = POPEDataSet(
        pope_path=args.pope_path, 
        data_path=args.data_path, 
        trans=vis_processors["eval"]
    )
    pope_loader = torch.utils.data.DataLoader(
        pope_dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=args.num_workers,
        drop_last=False
    )

    print ("load data finished")

    base_dir = "./pope_results/" + args.model
    if not os.path.exists(base_dir):
        os.makedirs(base_dir, exist_ok=True)

    output_file = f'{args.pope_type}_dosample-{args.sample}_ws{args.window_size}_penscale{args.penalty_scale}_p{args.top_p}_k{args.top_k}_max{args.max_new_tokens}_dlc_{args.use_dlc}.jsonl'

    print(f"Start evaluation, results will be saved to {os.path.join(base_dir, output_file)}")
    print("Start eval...")
    
    pred_list, label_list = [], []
    
    for batch_id, data in tqdm(enumerate(pope_loader), total=len(pope_loader)):
        image = data["image"]
        qu = data["query"]
        label = data["label"]
        label_list = label_list + list(label)

        template = INSTRUCTION_TEMPLATE[args.model]
        qu = [template.replace("<question>", q) for q in qu]

        image = image.to(device)
        label = torch.Tensor(label).to(device)

        image_filename = pope_dataset.image_list[batch_id]
        image_path = os.path.join(args.data_path, image_filename)
        raw_image = Image.open(image_path).convert("RGB")

        clip_inputs = clip_processor(images=raw_image, return_tensors="pt").to(device)
        image_features = clip_model.get_image_features(**clip_inputs)

        if args.use_dlc:
            visual_penalizer = DLCLogitsProcessor(
                clip_model=clip_model,
                clip_processor=clip_processor,
                image_features=image_features,
                tokenizer=tokenizer,
                window_size=args.window_size,
                penalty_scale=args.penalty_scale,
                top_k=args.visual_top_k,
                query=data["query"],
                ispope=True,
            )
            logits_processor = LogitsProcessorList([visual_penalizer])

            with torch.inference_mode():
                with torch.no_grad():
                    out = model.generate(
                        prompt=qu,
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
                        prompt=qu,
                        image=image.half(),
                        temperature=args.temperature,
                        max_new_tokens=args.max_new_tokens,
                        top_p=args.top_p,
                        top_k=args.top_k,
                        do_sample=args.sample,
                    )
        
        pred_list = recorder(out, pred_list)

        result = {
            "question": qu[0],
            "answer": out[0],
            "label": label.item(),
            "predicted": pred_list[-1]
        }
        with open(os.path.join(base_dir, output_file), "a") as f:
            json.dump(result, f)
            f.write('\n')

    print("===============================================")
    if len(pred_list) != 0:
        print_acc(pred_list, label_list, logger)



if __name__ == "__main__":
    main()