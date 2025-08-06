import argparse
import os
import random
import json
from PIL import Image
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import tqdm

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

from shr.shr_utils import *
from shr.gpt_utils import *
import warnings

from DLC_logits_processor import DLCLogitsProcessor
import warnings
from transformers import AutoTokenizer, LogitsProcessorList, AutoModel, AutoProcessor

warnings.filterwarnings("ignore")

os.environ["TOKENIZERS_PARALLELISM"] = "false"

MODEL_EVAL_CONFIG_PATH = {
    "minigpt4": "eval_configs/minigpt4_eval.yaml",
    "instructblip": "eval_configs/instructblip_eval.yaml",
    "lrv_instruct": "eval_configs/lrv_instruct_eval.yaml",
    "shikra": "eval_configs/shikra_eval.yaml",
    "llava-1.5": "eval_configs/llava-1.5_eval.yaml",
}

INSTRUCTION_TEMPLATE = {
    "minigpt4": "###Human: <Img><ImageHere></Img> <question> ###Assistant:",
    "instructblip": "<ImageHere><question>",
    "lrv_instruct": "###Human: <Img><ImageHere></Img> <question> ###Assistant:",
    "shikra": "USER: <im_start><ImageHere><im_end> <question> ASSISTANT:",
    "llava-1.5": "USER: <ImageHere> <question> ASSISTANT:"
}


def parse_args():
    parser = argparse.ArgumentParser(description="SHR evaluation on LVLMs.")

    parser.add_argument("--model", type=str, default="llava-1.5", help="model")
    parser.add_argument("--gpu-id", type=int, default=5, help="specify the gpu to load the model.")
    parser.add_argument(
        "--options",
        nargs="+",
        help="override some settings in the used config, the key-value pair "
        "in xxx=yyy format will be merged into config file (deprecate), "
        "change to --cfg-options instead.",
    )
    parser.add_argument("--clip-model-path", type=str, default="", help="CLIP model path")
    parser.add_argument("--tokenizer-path", type=str, default="", help="tokenizer path (auto-selected if empty)")

    parser.add_argument("--output-dir", type=str, default="./shr_results", help="output directory")

    # SHR parameters
    parser.add_argument("--api-key", type=str, default='', help="key to the OPENAI API.")
    parser.add_argument("--vg-path", type=str, default='/datasets/VG/', help="path to vg file.")
    parser.add_argument("--shr-path", type=str, default='', help="path to SHR annotation file.")
    parser.add_argument("--no-gpt-judge", default=False, action='store_true', help="whether not to do GPT evaluation. If True, only evaluate ngram repitition.")
    parser.add_argument("--resume", default=True, action='store_true', help="whether to resume from checkpoint if available.")

    parser.add_argument("--window_size", type=int, default=8, help="Window size for visual penalty processor")
    parser.add_argument("--penalty_scale", type=float, default=3.0, help="Penalty scale for visual penalty processor")
    parser.add_argument("--visual_top_k", type=int, default=50, help="Top k for visual penalty processor")
    parser.add_argument("--max-new-tokens", type=int, default=512, help="Maximum length of generated text")
    
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default=None)
    parser.add_argument("--sample", action="store_true", default=False, help="Whether to sample or not")

    args = parser.parse_args()
    return args


def setup_seeds(config):
    seed = config.run_cfg.seed + get_rank()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    cudnn.benchmark = False
    cudnn.deterministic = True


def get_result_paths(args):
    base_eval_path = os.path.join(args.output_dir, args.model)
    
    result_dir = f'sample{args.sample}_ws{args.window_size}_ps{args.penalty_scale}_topk{args.visual_top_k}'
    
    if not os.path.exists(os.path.join(base_eval_path)):
        os.makedirs(os.path.join(base_eval_path))
    
    os.makedirs(base_eval_path, exist_ok=True)
    
    eval_path = os.path.join(base_eval_path, result_dir)
    os.makedirs(eval_path, exist_ok=True)
    
    return {
        'eval_path': eval_path,
        'checkpoint_path': os.path.join(eval_path, 'checkpoint.json'),
        'judgement_path': os.path.join(eval_path, 'judgement.json'),
        'metrics_path': os.path.join(eval_path, 'metrics.json'),
        'progress_path': os.path.join(eval_path, 'progress.json')
    }


def load_checkpoint(paths, run_all):
    """Load checkpoint if it exists"""
    checkpoint_path = paths['checkpoint_path']
    progress_path = paths['progress_path']
    
    judgement = {run: {} for run in run_all}
        
    processed_ids = set()
    gram_stats = {'gram1': 0, 'gram2': 0, 'gram3': 0, 'gram4': 0, 'count': 0}
    
    if os.path.exists(checkpoint_path):
        with open(checkpoint_path, 'r') as f:
            checkpoint_data = json.load(f)
            for run in run_all:
                if run in checkpoint_data:
                    judgement[run] = checkpoint_data[run]
                    
    if os.path.exists(progress_path):
        with open(progress_path, 'r') as f:
            progress_data = json.load(f)
            processed_ids = set(progress_data.get('processed_ids', []))
            gram_stats = progress_data.get('gram_stats', gram_stats)
    
    return judgement, processed_ids, gram_stats


def save_checkpoint(paths, judgement, processed_ids, gram_stats):
    """Save checkpoint data"""
    # Save judgement data
    with open(paths['checkpoint_path'], 'w') as f:
        json.dump(judgement, f, indent=2)
    
    # Save progress data
    progress_data = {
        'processed_ids': list(processed_ids),
        'gram_stats': gram_stats
    }
    with open(paths['progress_path'], 'w') as f:
        json.dump(progress_data, f, indent=2)


def load_models(args, device):
    """Load all required models"""
    print("Loading CLIP model and processor...")
    clip_model = AutoModel.from_pretrained(args.clip_model_path).to(device)
    clip_processor = AutoProcessor.from_pretrained(args.clip_model_path)

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, use_fast=False)
    
    # Add special tokens for certain models
    # if args.model in ["instructblip", "minigpt4"]:
    #     tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, use_fast=False, truncation_side="left")
    #     tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    #     tokenizer.add_special_tokens({'bos_token': '</s>'})
    #     tokenizer.add_special_tokens({'eos_token': '</s>'})
    #     tokenizer.add_special_tokens({'unk_token': '</s>'})
    
    return clip_model, clip_processor, tokenizer

def main():
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)

    args.cfg_path = MODEL_EVAL_CONFIG_PATH[args.model]
    cfg = Config(args)

    setup_seeds(cfg)
    device = torch.device("cuda") if torch.cuda.is_available() else "cpu"

    # setup openai
    if not args.no_gpt_judge:
        if not args.api_key:
            print("Warning: No API key provided. GPT evaluation will be skipped.")
            args.no_gpt_judge = True
        else:
            setup_openai(args.api_key)

    paths = get_result_paths(args)
    print(f"Results will be saved to: {paths['eval_path']}")
    
    # Load models
    clip_model, clip_processor, tokenizer = load_models(args, device)

    print('Initializing Model')

    model_config = cfg.model_cfg
    model_config.device_8bit = args.gpu_id
    model_cls = registry.get_model_class(model_config.arch)
    model = model_cls.from_config(model_config).to(device)
    model.eval()

    vis_processors, txt_processors = load_preprocess(cfg.get_config().preprocess)

    # visual genome annotations
    val_images = json.load(open(os.path.join(args.shr_path, "val_images_final.json")))
    vg_image_data = json.load(open(os.path.join(args.shr_path, "image_data.json")))
    id2path = {
        _data["image_id"]:os.path.join(args.vg_path, _data["url"].split("/")[-2], _data["url"].split("/")[-1]) 
        for _data in vg_image_data
    }
    id2img = {_data["image_id"]:_data for _data in vg_image_data}
    region = json.load(open(os.path.join(args.shr_path, "region_descriptions.json")))
    id2reg = {r["regions"][0]["image_id"]:r for r in region}
    
    run_all = ['run1']
    
    # Load checkpoint if resume is enabled
    if args.resume:
        judgement, processed_ids, gram_stats = load_checkpoint(paths, run_all)
        print(f"Resuming from checkpoint. Already processed {len(processed_ids)} images.")
    else:
        judgement = {run: {} for run in run_all}
        processed_ids = set()
        gram_stats = {'gram1': 0, 'gram2': 0, 'gram3': 0, 'gram4': 0, 'count': 0}
    
    # factual information
    factual_inf = {}
    factual_part1 = os.path.join(args.shr_path, "shr_factual_part1.jsonl")
    factual_part2 = os.path.join(args.shr_path, "shr_factual_part2.jsonl")
    for line in open(factual_part1).readlines():
        factual = json.loads(line)
        image_id, factuals = list(factual.keys())[0], list(factual.values())[0]
        factual_inf[image_id] = factuals
    for line in open(factual_part2).readlines():
        factual = json.loads(line)
        image_id, factuals = list(factual.keys())[0], list(factual.values())[0]
        factual_inf[image_id] = factuals

    for i, _data in enumerate(tqdm.tqdm(val_images)):
        image_id = _data["image_id"]
        
        if str(image_id) in processed_ids:
            print(f"Skipping already processed image_id: {image_id}")
            continue
            
        try:
            image_path = id2path[int(image_id)]
            image = Image.open(image_path).convert("RGB")
            image_tensor = vis_processors["eval"](image)
            if type(image_tensor) is list:
                image_tensor = [image.to(model.device, dtype=torch.float16) for image in image_tensor]
            else:
                image_tensor = image_tensor.to(model.device, dtype=torch.float16)

            clip_inputs = clip_processor(images=image, return_tensors="pt").to(device)
            image_features = clip_model.get_image_features(**clip_inputs)
            inp = "Please describe this image in detail."
            template = INSTRUCTION_TEMPLATE[args.model]
            qu = template.replace("<question>", inp)
            image = image_tensor.to(device).unsqueeze(0)

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
                    outputs = model.generate(
                        prompt=[qu],
                        image = image.half(),
                        images_cd= None,
                        prompt_cd =None,
                        temperature=args.temperature,
                        max_new_tokens=args.max_new_tokens,
                        logits_processor=logits_processor,
                        top_p=args.top_p,
                        top_k=args.top_k,
                        do_sample=args.sample,
                        )[0]

            # get GPT judgement
            description = get_desc(id2img, id2reg, int(image_id))
            model_cap_sep, is_repeated = get_model_cap(outputs)
            
            # calculate repetition
            gram1 = cal_repetition(outputs, 1)
            gram2 = cal_repetition(outputs, 2)
            gram3 = cal_repetition(outputs, 3)
            gram4 = cal_repetition(outputs, 4)
            
            # Update gram statistics
            gram_stats['gram1'] += gram1
            gram_stats['gram2'] += gram2
            gram_stats['gram3'] += gram3
            gram_stats['gram4'] += gram4
            gram_stats['count'] += 1
                
            # skip gpt judgement 
            if not args.no_gpt_judge:
                factual_text = ""
                if str(image_id) in factual_inf:
                    for text in factual_inf[str(image_id)]:
                        factual_text += text
                        factual_text += "\n"
                # GPT judgement
                judge_prompt = GPT_JUDGE_PROMPT.format(description, factual_text, model_cap_sep)
                if len(judge_prompt) > 15000:
                    print(f"skip {image_id} for too long prompt!")
                else:
                    for run in run_all:
                        retry_count = 0
                        while retry_count < 3:  # Retry up to 3 times
                            try:
                                judge = get_gpt_response(prompt=judge_prompt, model_name="gpt-4")
                                if "Judgement" not in judge:
                                    print(f"No judgement found for {image_id}, retrying...")
                                    retry_count += 1
                                    continue
                                else:
                                    # post-process
                                    final_judge = post_process_no_revise(judge, outputs)
                                    judgement[run][image_id] = {
                                        "raw_judgement": judge,
                                        "model_response": outputs,
                                        "judgement": final_judge,
                                    }
                                    break
                            except Exception as e:
                                print(f"Error getting GPT response for {image_id}: {e}")
                                retry_count += 1
                                if retry_count >= 3:
                                    print(f"Failed after 3 retries for {image_id}, skipping.")

            # Add to processed IDs and save checkpoint after each image
            processed_ids.add(str(image_id))
            
            # Save checkpoint after each image (or every N images for efficiency)
            if i % 5 == 0 or i == len(val_images) - 1:
                save_checkpoint(paths, judgement, processed_ids, gram_stats)
                print(f"Checkpoint saved after processing {len(processed_ids)} images")
                
        except Exception as e:
            print(f"Error processing image {image_id}: {e}")
            # Save checkpoint even on error
            save_checkpoint(paths, judgement, processed_ids, gram_stats)
            continue

    # Save final results
    if args.no_gpt_judge:
        print(f"gram-1 repetition: {round(gram_stats['gram1']/gram_stats['count'], 3)}")
        print(f"gram-2 repetition: {round(gram_stats['gram2']/gram_stats['count'], 3)}")
        print(f"gram-3 repetition: {round(gram_stats['gram3']/gram_stats['count'], 3)}")
        print(f"gram-4 repetition: {round(gram_stats['gram4']/gram_stats['count'], 3)}")
    else:
        # Save metrics
        metrics = {}
        for run in run_all:
            metrics[run] = {}
            get_metric(judgement[run], metrics[run])
            
        # repetition
        metrics['gram-1-repetition'] = round(gram_stats['gram1']/gram_stats['count'], 3)
        metrics['gram-2-repetition'] = round(gram_stats['gram2']/gram_stats['count'], 3)
        metrics['gram-3-repetition'] = round(gram_stats['gram3']/gram_stats['count'], 3)
        metrics['gram-4-repetition'] = round(gram_stats['gram4']/gram_stats['count'], 3)
        
        # halucination ratio
        metrics["mean_hal_ratio"] = round(
            sum(metrics[run]["hal_sents_ratio"] for run in run_all)/len(run_all), 3
        )
        metrics["model_base"] = args.model
        
        # dump judgement file
        with open(paths['judgement_path'], "w") as f:
            json.dump(judgement, f)
            
        # dump metric file
        with open(paths['metrics_path'], "w") as f:
            json.dump(metrics, f)
        
        print(f"Final results saved to {paths['eval_path']}")

if __name__ == "__main__":
    main()