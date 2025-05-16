import argparse
from pathlib import Path
import torch
import os
import json
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from eval_utils import MP4_VID_DIRECTORY, TITLE_MAPPING_DIRECTORY, TEST_VIDS, get_augmented_questions, get_fine_tune_test_vids
from tqdm import tqdm
import numpy as np
import random

def inference(video_path, prompt, text_only, fps, max_frames, 
              max_new_tokens=2048, total_pixels=20480 * 28 * 28, min_pixels=16 * 28 * 28):
    content = [{"type": "text", "text": prompt}]
    if not text_only:
        content.append({"video": video_path, "total_pixels": total_pixels, "min_pixels": min_pixels, "fps": fps, "max_frames": max_frames})
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": content},
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs, video_kwargs = process_vision_info([messages], return_video_kwargs=True)
    fps_inputs = video_kwargs['fps']
    if not text_only:
        print("video input:", video_inputs[0].shape)
        num_frames, _, resized_height, resized_width = video_inputs[0].shape
        print("num of video tokens:", int(num_frames / 2 * resized_height / 28 * resized_width / 28))
    inputs = processor(text=[text], images=image_inputs, videos=video_inputs, fps=fps_inputs, padding=True, return_tensors="pt")
    inputs = inputs.to('cuda')

    output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
    generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, output_ids)]
    output_text = processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
    return output_text[0]

def batch_inference(video_path, prompts, text_only, fps, max_frames,
                    max_new_tokens=2048, total_pixels=20480 * 28 * 28, min_pixels=16 * 28 * 28):
    # prepare the vision info
    vision_content = {"video": video_path, "total_pixels": total_pixels, "min_pixels": min_pixels, "fps": fps, "max_frames": max_frames}
    vision_message = [{"role": "user", "content": [vision_content]}] if not text_only else []
    
    image_inputs, video_inputs, video_kwargs = process_vision_info([vision_message], return_video_kwargs=True)
    fps_inputs = video_kwargs['fps']
    if not text_only:
        print("video input:", video_inputs[0].shape)
        num_frames, _, resized_height, resized_width = video_inputs[0].shape
        print("num of video tokens:", int(num_frames / 2 * resized_height / 28 * resized_width / 28))
    
    outputs = []
    for prompt in tqdm(prompts):
        content = [{"type": "text", "text": prompt}]
        if not text_only:
            content.append(vision_content) 
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": content},
        ]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[text], images=image_inputs, videos=video_inputs, fps=fps_inputs, padding=True, return_tensors="pt")
        inputs = inputs.to('cuda')
        output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
        generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, output_ids)]
        output_text = processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        outputs.append(output_text[0])
    return outputs
    
def get_evaluation_template(title=None):
    template = """You are an expert in mime performance understanding and question answering. 
    Typically, the mime would use exaggerated gestures or pretend objects to convey a message.
    Answer the question in one sentence using the video, with brief explanations. 
    Do not describe the frames just answer the question. 
    If the mime is using imaginary objects, describe the objects as if they were real.
    """ + (f"The title of the video is {title}\n" if title is not None else "\n") + "Question: {question}"
    return template

def to_qwen_time_format(seconds):
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
        default=2.0,
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=128,
    )
    parser.add_argument(
        "--max_frames",
        type=int,
        default=768,
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
    parser.add_argument(
        "--fine_tune_test",
        action="store_true",
        default=True
    )
    parser.add_argument(
        "--sft_model_path",
        type=str,
        required=True
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
    model_name = "Qwen/Qwen2.5-VL-7B-Instruct"

    # if args.sft_model_path is not None:
    #     model_path = get_model_local_path(model_name, args.fine_tune_test)
    #     if model_path is None:
    #         model_path = model_name # download from huggingface
    # else:
    model_path = args.sft_model_path
    # model_path = model_name
    
    print(f"Model path: {model_path}")
    
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map="auto"
    )
    print("Model loaded")
    processor = AutoProcessor.from_pretrained(model_path)
    print("Processor loaded")

    root = Path(__file__).parent.parent.parent
    data_folder = root / 'data'
    print(data_folder)
    
    df = get_augmented_questions(data_folder, to_qwen_time_format)
    
    videos = list(df['file'].unique())
    eval_videos = TEST_VIDS if args.debug else get_fine_tune_test_vids(data_folder) if args.fine_tune_test else videos

    print("Evaluating on videos:", eval_videos)
    
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
        responses = []
        df_video = df[df['file'] == vid_name]
        
        indices = df_video.index.tolist()
        prompts = [get_evaluation_template(title).format(question=df_video.at[i, 'augmented_question']) for i in indices]
        print(prompts)
        responses = batch_inference(video_path, prompts, text_only=args.text_only, fps=args.fps, max_frames=args.max_frames, max_new_tokens=args.max_tokens)
        df.loc[indices, 'generated_response'] = responses
        
        # for index, row in df_video.iterrows():
        #     question = row['augmented_question']
        #     prompt = get_evaluation_template().format(question=question)
        #     response = inference(video_path, prompt, text_only=args.text_only, fps=args.fps, max_frames=args.max_frames)
        #     df.at[index, 'generated_response'] = response
            
        seed_str = f"seed_{args.seed}_" if args.seed is not None else ""
        fine_tune_test_str = "fine_tune_test_" if args.fine_tune_test else ""
        model_path_name = Path(args.sft_model_path).name
        if "siq" in args.sft_model_path:
            model_path_name = "siq_" + model_path_name
        print("model_path_name", model_path_name)
        filename = data_folder / "sft" / f'qwen_{model_path_name}_annotation_generated.csv'
        df.to_csv(filename, index=False)