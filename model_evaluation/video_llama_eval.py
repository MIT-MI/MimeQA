import argparse
from pathlib import Path
import torch
import os
import json
from transformers import AutoModelForCausalLM, AutoProcessor
from eval_utils import MP4_VID_DIRECTORY, TITLE_MAPPING_DIRECTORY, TEST_VIDS, get_augmented_questions, get_model_local_path
import numpy as np
import random

from tqdm import tqdm

import warnings

warnings.filterwarnings(
    "ignore",
    message="`num_logits_to_keep` is deprecated and will be removed in version 4.50.*",
    category=FutureWarning,
)

    
def get_evaluation_template(title=None):
    template = """You are an expert in mime performance understanding and question answering. 
    Typically, the mime would use exaggerated gestures or pretend objects to convey a message.
    Answer the question in one sentence using the video, with brief explanations. 
    Do not describe the frames just answer the question. 
    If the mime is using imaginary objects, describe the objects as if they were real.
    """ + (f"The title of the video is {title}\n" if title is not None else "\n") + "Question: {question}"
    return template

def to_videollama_time_format(seconds):
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
        "--max_frames",
        type=int,
        default=180,
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
    model_name = "DAMO-NLP-SG/VideoLLaMA3-7B"

    model_path = get_model_local_path(model_name)
    if model_path is None:
        model_path = model_name # download from huggingface
    
    print(f"Model path: {model_path}")

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)

    @torch.inference_mode()
    def infer(conversation):
        inputs = processor(
            conversation=conversation,
            add_system_prompt=True,
            add_generation_prompt=True,
            return_tensors="pt"
        )
        inputs = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
        if "pixel_values" in inputs:
            inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)
        output_ids = model.generate(**inputs, max_new_tokens=1024)
        response = processor.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
        return response

    root = Path(__file__).parent.parent.parent
    data_folder = root / 'data'
    print(data_folder)
    
    df = get_augmented_questions(data_folder, to_videollama_time_format)
    
    videos = list(df['file'].unique())
    eval_videos = TEST_VIDS if args.debug else videos
    
    print("Videos to evaluate:", eval_videos)
    
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
        
        for index, row in tqdm(df_video.iterrows()):
            question = row['augmented_question']
            prompt = get_evaluation_template(title).format(question=question)

            conversation = [
                {"role": "system", "content": "You are a helpful assistant."},
                {
                    "role": "user",
                    "content": [
                        {
                          "type": "video", 
                          "video": {"video_path": f"{video_path}", "fps": args.fps, "max_frames": args.max_frames}
                        },
                        {"type": "text", "text": f"{prompt}"},
                    ]
                },
            ]
            
            response = infer(conversation)

            df.at[index, 'generated_response'] = response
            
        seed_str = f"seed_{args.seed}_" if args.seed is not None else ""
        filename = data_folder / f'video_llama_text-only_{args.text_only}_fps{args.fps}_{args.max_frames}frames_title_{args.with_titles}_{seed_str}annotation_generated.csv'
        df.to_csv(filename, index=False)