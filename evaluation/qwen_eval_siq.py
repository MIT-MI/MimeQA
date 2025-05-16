import argparse
from pathlib import Path
import torch
import os
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from eval_utils import SIQ_VID_DIRECTORY, get_siq_augmented_questions, get_model_local_path
from tqdm import tqdm
import csv

def batch_inference(video_path, prompts, fps, max_frames,
                    max_new_tokens=2048, total_pixels=20480 * 28 * 28, min_pixels=16 * 28 * 28):
    # prepare the vision info
    vision_content = {"video": video_path, "total_pixels": total_pixels, "min_pixels": min_pixels, "fps": fps, "max_frames": max_frames}
    vision_message = [{"role": "user", "content": [vision_content]}]
    
    image_inputs, video_inputs, video_kwargs = process_vision_info([vision_message], return_video_kwargs=True)
    fps_inputs = video_kwargs['fps']
    print("video input:", video_inputs[0].shape)
    num_frames, _, resized_height, resized_width = video_inputs[0].shape
    print("num of video tokens:", int(num_frames / 2 * resized_height / 28 * resized_width / 28))
    
    outputs = []
    for prompt in tqdm(prompts):
        content = [{"type": "text", "text": prompt}]
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
    
def get_evaluation_template(open_ended=False):
    if open_ended:
        template = """You are given a video clip depicting a social situation. Based on the video, answer the following question in one sentence using the video, with brief explanations.
        Question: [{question}]
        """
    else:
        template = """You are given a video clip depicting a social situation. Based on the video, answer the following question by selecting the most appropriate option.
        Question: [{question}]
        Choices:  
        0. [{choice_0}]  
        1. [{choice_1}]  
        2. [{choice_2}]  
        3. [{choice_3}]
        Briefly explain your reasoning before outputting your answer in one sentence. At the end of your solution, write your answer in the following format, where <answer> is the number id of the correct choice:
        Answer: <answer>
        """ 
    return template

def to_qwen_time_format(seconds):
    minutes = int(seconds // 60)
    remaining_seconds = seconds % 60
    return f"{minutes:02} min:{remaining_seconds:.2f} sec"

def parse_args():
    parser = argparse.ArgumentParser(description="Script to process frames in a video.")
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
        "--fine_tune",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--open_ended",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
    )
    return parser.parse_args()

if __name__ == "__main__":
    # Parse args
    args = parse_args()

    # Load model
    if args.small:
        model_name = "Qwen/Qwen2.5-VL-7B-Instruct"
    else:
        model_name = "Qwen/Qwen2.5-VL-72B-Instruct"

    if args.fine_tune:
        if args.small:
            model_path = "/scratch/mime_sft/output/qwen25_vl_lora_sft_small"
        else:
            model_path = "/scratch/mime_sft/output/qwen25_vl_lora_sft_full"
    else:
        model_path = get_model_local_path(model_name)
        if model_path is None:
            model_path = model_name # download from huggingface
    print("Model name:", args.model_name)
    print(f"Model path: {model_path}")

    
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map="auto"
    )
    processor = AutoProcessor.from_pretrained(model_path)

    root = Path(__file__).parent.parent.parent
    data_folder = root / 'data'
    print(data_folder)

    filename = data_folder / f'qwen_siq_finetune_{args.fine_tune}_open_{args.open_ended}_name_{args.model_name}_annotation_generated.csv'
    print("will be saved to", filename)
    
    df = get_siq_augmented_questions(data_folder, to_qwen_time_format)
    
    eval_videos = list(df['file'].unique())

    print("Number of videos:", len(eval_videos))
    print("Evaluating on videos:", eval_videos)
    
    for n_vid, vid_name in enumerate(eval_videos):
        vid_name_mp4 = vid_name + ".mp4"
        print(vid_name_mp4)
        video_path = os.path.join(SIQ_VID_DIRECTORY, vid_name_mp4)
        responses = []
        df_video = df[df['file'] == vid_name]
        indices = df_video.index.tolist()
        prompts = [get_evaluation_template(args.open_ended).format(
            question=df_video.at[i, 'augmented_question'],
            choice_0=df_video.at[i, 'a0'],
            choice_1=df_video.at[i, 'a1'],
            choice_2=df_video.at[i, 'a2'],
            choice_3=df_video.at[i, 'a3']
            ) for i in indices]
        print(prompts)
        responses = batch_inference(video_path, prompts, fps=args.fps, max_frames=args.max_frames, max_new_tokens=args.max_tokens)
        df.loc[indices, 'generated_response'] = responses
        
        df.to_csv(filename, index=False, quoting=csv.QUOTE_MINIMAL)