import argparse
import asyncio
import os
import sys
import base64
import cv2
from pathlib import Path
import numpy as np
import nest_asyncio
from io import BytesIO

from openai import OpenAI
from eval_utils import get_augmented_questions, MP4_VID_DIRECTORY, TEST_VIDS
from PIL import Image
import asyncio
from tqdm.asyncio import tqdm_asyncio
import pandas as pd
from tenacity import retry, stop_after_attempt, wait_random_exponential

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)

from openai_utils import LLM

client = OpenAI(api_key=os.environ.get("MIT_OPENAI_API_KEY"))


def to_base64(frame):
    """
    Convert a frame (numpy array) to a base64-encoded PNG string.
    """
    # Convert the frame to an image using PIL
    pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    
    
    # Save the image to a BytesIO object
    buffered = BytesIO()
    pil_img.save(buffered, format="JPEG", quality=95)
    
    # Encode the image as base64
    encoded_image = base64.b64encode(buffered.getvalue()).decode("utf-8")
    
    return encoded_image

def to_video_gpt_time_format(seconds):
    minutes = int(seconds // 60)
    remaining_seconds = seconds % 60
    return f"{minutes:02} min:{remaining_seconds:.2f} sec"

def get_video_frames(video_path, max_frames_num=384, force_sample=False):
    """
    If it's a .webm or if you just want to standardize everything to MP4,
    re-encode and then decode with decord.
    """
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        raise ValueError("Error opening video file")

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_time = total_frames / fps if fps > 0 else 0
    frame_idx = [i for i in range(0, total_frames, fps)]  # 1 frame per second
    frame_time = [i / fps for i in frame_idx]
    
    if len(frame_idx) >= max_frames_num or force_sample:
        # Sample frames uniformly
        uniform_sampled_frames = np.linspace(0, total_frames - 1, max_frames_num, dtype=int)
        frame_idx = uniform_sampled_frames.tolist()
        frame_time = [i / fps for i in frame_idx]
        
    frames = []
    for idx in frame_idx:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frames.append(frame)

    cap.release()
    
    # Convert each frame to base64
    base64_frames = [to_base64(frame) for frame in frames]
    video_time = to_video_gpt_time_format(video_time)
    frame_time = [to_video_gpt_time_format(timestamp) for timestamp in frame_time]
    return video_path, video_time, base64_frames, frame_time


def get_evaluation_template():
     # Evaluation template
    template = """You are an expert in mime performance understanding and question answering. 
    Typically, the mime would use exaggerated gestures or pretend objects to convey a message. 
    The video lasts for {video_time}, and {num_frames} frames are uniformly sampled from it.
    These frames are located at {frame_time}. Answer the question in one sentence using the video, with brief explanations. 
    Do not describe the frames just answer the question. 
    If the mime is using imaginary objects, describe the objects as if they were real.
    Question: {question}"""
    return template

def get_split_evaluation_template():
     # Evaluation template
    video_template = """You are an expert in mime performance understanding and question answering. 
    Typically, the mime would use exaggerated gestures or pretend objects to convey a message. 
    The video lasts for {video_time}, and {num_frames} frames are uniformly sampled from it.
    These frames are located at {frame_time}. """
    question_template = """Answer the question in one sentence using the video, with brief explanations. 
    Do not describe the frames just answer the question. 
    If the mime is using imaginary objects, describe the objects as if they were real.
    Question: {question}"""
    return video_template, question_template

def get_interleaved_prompt(video_time, num_frames, frame_time, base64_frames, question, text_only=False,):
     # Evaluation template
    content = [{"type": "text", "text": f"""You are an expert in mime performance understanding and question answering. 
    Typically, the mime would use exaggerated gestures or pretend objects to convey a message.
    The video lasts for {video_time}, and {num_frames} frames are uniformly sampled from it."""}]
    for frame, frame_timestamp in zip(base64_frames, frame_time):
        content.append({"type": "text", "text": f"Frame at {frame_timestamp}"})
        if not text_only:
            content.append({"image": frame, "resize": 512})
    content.append(
        {"type": "text", "text": f"""Answer the question in one sentence using the video, with brief explanations. 
    Do not describe the frames just answer the question. 
    If the mime is using imaginary objects, describe the objects as if they were real.
    Question: {question}"""}
    )
    return content

def parse_args():
    parser = argparse.ArgumentParser(description="Script to process frames in a video.")
    parser.add_argument(
        "--max_frames_num",
        type=int,
        default=64,  
        help="Maximum number of frames to process (0 for all frames)."
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0,  
        help="GPT temperature setting"
    )
    parser.add_argument(
        "--prompt_mode",
        choices=["regular", "split", "interleave"],
        default="split"
    )
    parser.add_argument(
        "--text_only",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=128,
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=None
    )
    return parser.parse_args()

sema = asyncio.Semaphore(5)

async def generate_worker(model, messages, **kwargs):
    async with sema:
        llm = LLM(llm_str=model)
        response = await llm.create_completion(messages, **kwargs)
        return response

@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(3)) 
def generated_response_sync(model, messages, **kwargs):
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        **kwargs,
    )
        
        
async def main():
    nest_asyncio.apply()
    # IMPORTANT: make sure to do export DECORD_EOF_RETRY_MAX=20480
    # Parse args
    args = parse_args()
    
    save_file = f'gpt4o_max-frames_{args.max_frames_num}_text-only_{args.text_only}_annotation_generated.csv'
        
    # Get data
    root = Path(__file__).parent.parent.parent
    data_folder = root / 'data'
    print(data_folder)
    if os.path.exists(data_folder / save_file):
        df = pd.read_csv(data_folder / save_file)
    else:
        df = get_augmented_questions(data_folder, to_video_gpt_time_format, save_file)
    videos = list(df['file'].unique())
    eval_videos = TEST_VIDS if args.debug else videos
    
    curr_progress = 32
    delay_count = 0
    eval_videos = eval_videos[curr_progress:]
    
    for n_vid, vid_name in enumerate(eval_videos):
        base_name, _ = os.path.splitext(vid_name)
        vid_name_mp4 = base_name + ".mp4"
        print("Evaluating: " + vid_name_mp4)
        video_path = os.path.join(MP4_VID_DIRECTORY, vid_name_mp4)
        
        # Process the frames
        video_path, video_time, base64_frames, frame_time = get_video_frames(video_path, max_frames_num=args.max_frames_num, force_sample=False)
        frame_time_str = ",".join(frame_time)
        
        # save frames for inspection
        # for i, frame in enumerate(base64_frames):
        #     img = base64.b64decode(frame)
        #     with open(f"frames/frame_{i}.png", "wb") as f:
        #         f.write(img)
        
        df_video = df[df['file'] == vid_name]
        tasks = []
        model = "gpt-4o"
        kwargs = {
            "temperature": args.temperature,
            "max_tokens": args.max_tokens,
            "timeout": args.timeout
        }
        
        for index, row in df_video.iterrows():
            # Get the prompt 
            question = row['augmented_question']
            frames_copy = base64_frames.copy()
            imgs = [] if args.text_only else map(lambda x: {"image": x, "resize": 512}, frames_copy)
            if args.prompt_mode == "interleave":
                interleaved_prompt = get_interleaved_prompt(video_time, len(frames_copy), frame_time, frames_copy, question, text_only=args.text_only)
                PROMPT_MESSAGES = [
                    {
                        "role": "user",
                        "content": interleaved_prompt,
                    },
                ]
            elif args.prompt_mode == "split":
                # This prompt allows for more cacheing by putting the question at the end of the prompt
                video_template, question_template = get_split_evaluation_template()
                video_prompt = video_template.format(video_time=video_time, num_frames=len(frames_copy), frame_time=frame_time_str)
                question_prompt = question_template.format(question=question)
                # print(video_prompt, question_prompt)
                PROMPT_MESSAGES = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": video_prompt},
                            *imgs,
                            {"type": "text", "text": question_prompt},
                        ],
                    },
                ]
            else:
                text_prompt = get_evaluation_template().format(video_time=video_time, question=question, num_frames=len(frames_copy), frame_time=frame_time_str)
                PROMPT_MESSAGES = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": text_prompt},
                            *imgs,
                        ],
                    },
                ]
            messages = PROMPT_MESSAGES
            
            tasks.append(generate_worker(model, messages, **kwargs))
            
        print("generating")
        responses = await tqdm_asyncio.gather(*tasks, desc=f"Generating responses for {curr_progress + n_vid}")
        
        print("Writing responses to dataframe...")
        for df_index, response in zip(df_video.index, responses):
            df.at[df_index, 'generated_response'] = response.choices[0].message.content if response is not None else None
        df.to_csv(data_folder / save_file, index=False)
        
        delay_count += 1
        
        if delay_count % 3 == 0:
            print("Reset, sleep for 120 seconds...")
            await asyncio.sleep(120)
        else:
            print("Sleeping for 60 seconds...")
            await asyncio.sleep(60)
    
        

if __name__ == "__main__":
    asyncio.run(main())
        