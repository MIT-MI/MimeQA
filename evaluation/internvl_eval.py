import argparse
from pathlib import Path
import torch
import os
import json
import math
from transformers import AutoModel, AutoTokenizer
from eval_utils import MP4_VID_DIRECTORY, TITLE_MAPPING_DIRECTORY, TEST_VIDS, get_augmented_questions, get_model_local_path
import numpy as np
import random
from tqdm import tqdm

import torchvision.transforms as T
from decord import VideoReader, cpu
from PIL import Image
from torchvision.transforms.functional import InterpolationMode

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform

def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio

def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images

def get_index(bound, fps, max_frame, first_idx=0, num_segments=32):
    if bound:
        start, end = bound[0], bound[1]
    else:
        start, end = -100000, 100000
    start_idx = max(first_idx, round(start * fps))
    end_idx = min(round(end * fps), max_frame)
    seg_size = float(end_idx - start_idx) / num_segments
    frame_indices = np.array([
        int(start_idx + (seg_size / 2) + np.round(seg_size * idx))
        for idx in range(num_segments)
    ])
    return frame_indices

def load_video(video_path, bound=None, input_size=448, max_num=1, num_segments=32):
    vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
    max_frame = len(vr) - 1
    fps = float(vr.get_avg_fps())

    pixel_values_list, num_patches_list = [], []
    transform = build_transform(input_size=input_size)
    frame_indices = get_index(bound, fps, max_frame, first_idx=0, num_segments=num_segments)
    for frame_index in frame_indices:
        img = Image.fromarray(vr[frame_index].asnumpy()).convert('RGB')
        img = dynamic_preprocess(img, image_size=input_size, use_thumbnail=True, max_num=max_num)
        pixel_values = [transform(tile) for tile in img]
        pixel_values = torch.stack(pixel_values)
        num_patches_list.append(pixel_values.shape[0])
        pixel_values_list.append(pixel_values)
    pixel_values = torch.cat(pixel_values_list)
    return pixel_values, num_patches_list

def split_model(model_name):
    device_map = {}
    world_size = torch.cuda.device_count()
    num_layers = {
        'InternVL2_5-1B': 24, 'InternVL2_5-2B': 24, 'InternVL2_5-4B': 36, 'InternVL2_5-8B': 32,
        'InternVL2_5-26B': 48, 'InternVL2_5-38B': 64, 'InternVL2_5-78B': 80}[model_name]
    # Since the first GPU will be used for ViT, treat it as half a GPU.
    num_layers_per_gpu = math.ceil(num_layers / (world_size - 0.5))
    num_layers_per_gpu = [num_layers_per_gpu] * world_size
    num_layers_per_gpu[0] = math.ceil(num_layers_per_gpu[0] * 0.5)
    layer_cnt = 0
    for i, num_layer in enumerate(num_layers_per_gpu):
        for j in range(num_layer):
            device_map[f'language_model.model.layers.{layer_cnt}'] = i
            layer_cnt += 1
    device_map['vision_model'] = 0
    device_map['mlp1'] = 0
    device_map['language_model.model.tok_embeddings'] = 0
    device_map['language_model.model.embed_tokens'] = 0
    device_map['language_model.output'] = 0
    device_map['language_model.model.norm'] = 0
    device_map['language_model.model.rotary_emb'] = 0
    device_map['language_model.lm_head'] = 0
    device_map[f'language_model.model.layers.{num_layers - 1}'] = 0

    return device_map
    
def get_evaluation_template(title=None):
    template = """You are an expert in mime performance understanding and question answering. 
    Typically, the mime would use exaggerated gestures or pretend objects to convey a message.
    Answer the question in one sentence using the video, with brief explanations. 
    Do not describe the frames just answer the question. 
    If the mime is using imaginary objects, describe the objects as if they were real.
    """ + (f"The title of the video is {title}\n" if title is not None else "\n") + "Question: {question}"
    return template

def to_internvl_time_format(seconds):
    minutes = int(seconds // 60)
    remaining_seconds = seconds % 60
    return f"{minutes:02} min:{remaining_seconds:.2f} sec"
    # return f"{seconds:.2f}s"

def parse_args():
    parser = argparse.ArgumentParser(description="Script to process frames in a video.")
    parser.add_argument(
        "--text_only",
        action="store_true",
        default=False
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=128,
    )
    parser.add_argument(
        "--max_frames",
        type=int,
        default=64,
    )
    parser.add_argument(
        "--with_titles",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--seed",
        type=int
    )
    return parser.parse_args()

if __name__ == "__main__":
    # Parse args
    args = parse_args()

    # set random seeds
    if args.seed is not None:
        print(f"Setting random seed to {args.seed}")
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        np.random.seed(args.seed)
        random.seed(args.seed)


    # Load model
    model_name = "OpenGVLab/InternVL2_5-78B"
    device_map = split_model('InternVL2_5-78B')
    
    model_path = get_model_local_path(model_name)
    if model_path is None:
        model_path = model_name # download from huggingface
    
    print(f"Model path: {model_path}")

    model = AutoModel.from_pretrained(
      model_path,
      torch_dtype=torch.bfloat16,
      low_cpu_mem_usage=True,
      use_flash_attn=True,
      trust_remote_code=True,
      device_map=device_map
    ).eval()
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, use_fast=False)


    generation_config = dict(max_new_tokens=1024, do_sample=True)

    root = Path(__file__).parent.parent.parent
    data_folder = root / 'data'
    print(data_folder)
    
    df = get_augmented_questions(data_folder, to_internvl_time_format)
    
    videos = list(df['file'].unique())
    eval_videos = TEST_VIDS if args.debug else videos
    
    # loads title mapping
    with open(TITLE_MAPPING_DIRECTORY, 'r') as f:
        title_mapping = json.load(f)
    
    for n_vid, vid_name in enumerate(eval_videos):
        base_name, _ = os.path.splitext(vid_name)
        # convert .webm video name to .mp4
        vid_name_mp4 = base_name + ".mp4"
        print(vid_name_mp4)

        title=None
        if args.with_titles:
            title = title_mapping[vid_name]
            print(title)

        video_path = os.path.join(MP4_VID_DIRECTORY, vid_name_mp4)
        df_video = df[df['file'] == vid_name]
        
        pixel_values, num_patches_list = load_video(video_path, num_segments=8, max_num=1)
        pixel_values = pixel_values.to(torch.bfloat16).cuda()
        video_prefix = ''.join([f'Frame{i+1}: <image>\n' for i in range(len(num_patches_list))])
        
        for index, row in tqdm(df_video.iterrows()):
            question = row['augmented_question']
            prompt = get_evaluation_template(title).format(question=question)
            model_input = video_prefix + prompt
            response, history = model.chat(tokenizer, pixel_values, model_input, generation_config,
                               num_patches_list=num_patches_list, history=None, return_history=True)
            df.at[index, 'generated_response'] = response
            
        seed_str = f"seed_{args.seed}_" if args.seed is not None else ""
        filename = data_folder / f'internvl_text-only_{args.text_only}_fps{args.fps}_{args.max_frames}frames_title_{args.with_titles}_{seed_str}annotation_generated.csv'
        df.to_csv(filename, index=False)