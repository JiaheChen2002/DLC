import base64
import requests
import re
import argparse
import os
import random
import json
import time
from datetime import datetime
from tqdm import tqdm

GPT_JUDGE_PROMPT = '''
You are required to score the performance of two AI assistants in describing a given image. You should pay extra attention to the hallucination, which refers to the part of descriptions that are inconsistent with the image content, such as claiming the existence of something not present in the image or describing incorrectly in terms of the counts, positions, or colors of objects in the image. Please rate the responses of the assistants on a scale of 1 to 10, where a higher score indicates better performance, according to the following criteria:
1: Accuracy: whether the response is accurate with respect to the image content. Responses with fewer hallucinationsshould be given higher scores.
2: Detailedness: whether the response is rich in necessary details. Note that hallucinated descriptions should not countas necessary details.
Please output the scores for each criterion, containing only two values indicating the scores for Assistant 1 and 2, respectively. The two scores are separated by a space. Following the scores, please provide an explanation of your evaluation, avoiding any potential bias and ensuring that the order in which the responses were presented does not affect your judgment.

[Assistant 1]
{}
[End of Assistant 1]

[Assistant 2]
{}
[End of Assistant 2]

Output format:
Accuracy: <Scores of the two answers>
Reason:

Detailedness: <Scores of the two answers>
Reason: 
'''

API_KEY = ""

def call_api(prompt, image_path, api_key):
    # Function to encode the image
    def encode_image(image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    # Getting the base64 string
    base64_image = encode_image(image_path)

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }

    payload = {
    "model": "gpt-4o-2024-11-20",
    "messages": [
        {
        "role": "user",
        "content": [
            {
            "type": "text",
            "text": prompt
            },
            {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{base64_image}"
            }
            }
        ]
        }
    ],
    "max_tokens": 300
    }

    response = requests.post("https://xiaoai.plus/v1/chat/completions", headers=headers, json=payload)

    print(response.json().keys())
    return response.json()


def get_gpt4v_response(prompt, image_path, api_key, max_retries=2, wait_time=60):
    """Get GPT-4V response and handle possible errors, retry after timeout, return None if max retries exceeded"""
    for attempt in range(max_retries):
        try:
            response = call_api(prompt, image_path, api_key)
            if "choices" in response:
                return response["choices"][0]["message"]["content"]
            else:
                print(f"API返回错误: {response}")
                if attempt < max_retries - 1:
                    wait_seconds = wait_time * (attempt + 1)  # Exponential backoff
                    print(f"Waiting {wait_seconds} seconds before retrying ({attempt+1}/{max_retries})...")
                    time.sleep(wait_seconds)
                    continue
                else:
                    print("Max retries reached, skipping this image")
                    return None
        except Exception as e:
            print(f"Error occurred during request: {e}")
            if attempt < max_retries - 1:
                wait_seconds = wait_time * (attempt + 1)  # Exponential backoff
                print(f"Waiting {wait_seconds} seconds before retrying ({attempt+1}/{max_retries})...")
                time.sleep(wait_seconds)
                continue
            else:
                print("Max retries reached, skipping this image")
                return None

def parse_args():
    parser = argparse.ArgumentParser(description="Load results from JSONL files and evaluate VLM models using GPT-4V.")
    parser.add_argument("--method1_file", type=str, required=True, help="Path to jsonl file containing results of method 1")
    parser.add_argument("--method2_file", type=str, required=True, help="Path to jsonl file containing results of method 2")
    parser.add_argument("--data_path", type=str, default="", 
                        help="Path to COCO image data directory")
    parser.add_argument("--sample_num", type=int, default=500, help="Number of samples to evaluate")
    parser.add_argument("--method1_name", type=str, default="method1", help="Name of method 1")
    parser.add_argument("--method2_name", type=str, default="method2", help="Name of method 2")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--model", type=str, default="llava", help="Model name")
    
    args = parser.parse_args()
    return args


def generate_result_path(args):
    """Generate a unique result save path based on argument configuration"""

    method_names = f"{args.method1_name}_vs_{args.method2_name}"
    timestamp = datetime.now().strftime('%m%d_%H%M')
    result_path = f"{args.model}_bench_comparison_{method_names}_{timestamp}"
    return result_path


def load_results_from_jsonl(file_path):
    """Load generated results from jsonl file"""
    results = {}
    with open(file_path, 'r') as f:
        for line in f:
            data = json.loads(line)
            # Adapt to your JSONL format: image_id is a number, caption is the description
            image_id = data.get('image_id')
            caption = data.get('caption')
            if image_id is not None and caption:
                # Convert numeric ID to COCO format filename
                image_name = f"COCO_val2014_{image_id:012d}.jpg"
                results[image_name] = caption
    print(f"Loaded {len(results)} results from {file_path}")
    return results

def main():
    args = parse_args()
    # Set random seed
    random.seed(args.seed)

    # Load results of two methods
    method1_results = load_results_from_jsonl(args.method1_file)
    method2_results = load_results_from_jsonl(args.method2_file)

    # Find image filenames present in both files
    common_images = set(method1_results.keys()).intersection(set(method2_results.keys()))
    print(f"There are {len(common_images)} common images in both files")

    # If specified sample number is less than number of common images, sample randomly
    if args.sample_num < len(common_images):
        selected_images = random.sample(list(common_images), args.sample_num)
    else:
        selected_images = list(common_images)
        print(f"Specified sample number {args.sample_num} is greater than number of common images {len(common_images)}, evaluating all common images")

    # Create result save directory
    result_folder = generate_result_path(args)
    base_path = "gpt4v-results"
    result_path = os.path.join(base_path, result_folder)
    os.makedirs(result_path, exist_ok=True)

    # Save argument configuration
    with open(os.path.join(result_path, 'config.json'), 'w') as f:
        json_args = {k: v for k, v in vars(args).items() if not k.startswith('_')}
        json.dump(json_args, f, indent=2)

    gpt_answer_records = {}
    assistant_answer_records = {}
    avg_hal_score_1 = 0
    avg_hal_score_2 = 0
    avg_det_score_1 = 0
    avg_det_score_2 = 0
    num_count = 0

    for image_name in tqdm(selected_images, desc="Processing images"):
        # Build full image path
        image_path = os.path.join(args.data_path, image_name)

        # Check if file exists
        if not os.path.exists(image_path):
            print(f"Image does not exist: {image_path}, skipping")
            continue

        # Get generated results of two methods
        model_response_1 = method1_results[image_name]
        model_response_2 = method2_results[image_name]

        # Record results
        assistant_answer_records[image_name] = {
            "assistant_1": model_response_1,
            "assistant_2": model_response_2
        }

        # Evaluate using GPT-4V
        prompt = GPT_JUDGE_PROMPT.format(model_response_1, model_response_2)
        gpt_answer = get_gpt4v_response(prompt, image_path, api_key=API_KEY)

        if gpt_answer is None:
            print(f"Skipping evaluation for image {image_name}")
            continue

        print(gpt_answer)
        gpt_answer_records[image_name] = gpt_answer

        try:
            # Extract numbers using regex
            accuracy_part = gpt_answer.split("Accuracy:")[-1].split("Reason:")[0]
            accuracy_scores = re.findall(r'\d+', accuracy_part)

            detailedness_part = gpt_answer.split("Detailedness:")[-1].split("Reason:")[0]
            detailedness_scores = re.findall(r'\d+', detailedness_part)

            # Ensure at least two scores
            if len(accuracy_scores) >= 2 and len(detailedness_scores) >= 2:
                hal_score_1, hal_score_2 = accuracy_scores[0], accuracy_scores[1]
                det_score_1, det_score_2 = detailedness_scores[0], detailedness_scores[1]
            else:
                print(f"Not enough scores found: Accuracy={accuracy_scores}, Detailedness={detailedness_scores}")
                print(f"Original response snippet: {gpt_answer[:200]}...")
                continue

            # Print extracted scores for debugging
            print(f"Extracted scores - Accuracy: {hal_score_1}, {hal_score_2} | Detailedness: {det_score_1}, {det_score_2}")

            avg_hal_score_1 += int(hal_score_1)
            avg_hal_score_2 += int(hal_score_2)
            avg_det_score_1 += int(det_score_1)
            avg_det_score_2 += int(det_score_2)

        except Exception as e:
            print(f"Failed to parse GPT-4V response format, skipping image {image_name}")
            print(f"Error details: {str(e)}")
            print(f"First 200 chars of response: {gpt_answer[:200]}...")
            continue
        num_count += 1
        print("=========================================")

        # Periodically save results
        if num_count % 5 == 0:
            with open(os.path.join(result_path, 'answers.json'), "w") as f:
                json.dump(assistant_answer_records, f)
            with open(os.path.join(result_path, 'records.json'), "w") as f:
                json.dump(gpt_answer_records, f)

            # Calculate current average scores
            current_avg_hal_1 = float(avg_hal_score_1) / num_count
            current_avg_hal_2 = float(avg_hal_score_2) / num_count
            current_avg_det_1 = float(avg_det_score_1) / num_count
            current_avg_det_2 = float(avg_det_score_2) / num_count
            print(f"Current average accuracy score: {current_avg_hal_1:.2f}; {current_avg_hal_2:.2f}")
            print(f"Current average detailedness score: {current_avg_det_1:.2f}; {current_avg_det_2:.2f}")

    # Calculate final average scores
    if num_count > 0:
        avg_hal_score_1 = float(avg_hal_score_1) / num_count
        avg_hal_score_2 = float(avg_hal_score_2) / num_count
        avg_det_score_1 = float(avg_det_score_1) / num_count
        avg_det_score_2 = float(avg_det_score_2) / num_count
    else:
        print("No images were successfully evaluated, please check input files and API settings")
        return

    # Print final results
    print(f"Final average accuracy score ({args.method1_name} vs {args.method2_name}): {avg_hal_score_1:.2f}; {avg_hal_score_2:.2f}")
    print(f"Final average detailedness score ({args.method1_name} vs {args.method2_name}): {avg_det_score_1:.2f}; {avg_det_score_2:.2f}")
    print(f"Number of evaluated samples: {num_count}")

    # Save final results
    with open(os.path.join(result_path, 'answers.json'), "w") as f:
        json.dump(assistant_answer_records, f)
    with open(os.path.join(result_path, 'records.json'), "w") as f:
        json.dump(gpt_answer_records, f)
    with open(os.path.join(result_path, 'summary.txt'), "w") as f:
        f.write(f"Accuracy score ({args.method1_name} vs {args.method2_name}): {avg_hal_score_1:.2f}; {avg_hal_score_2:.2f}\n")
        f.write(f"Detailedness score ({args.method1_name} vs {args.method2_name}): {avg_det_score_1:.2f}; {avg_det_score_2:.2f}\n")
        f.write(f"Number of samples: {num_count}\n")

if __name__ == "__main__":
    main()