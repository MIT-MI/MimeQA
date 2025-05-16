import argparse
import os
from pathlib import Path
import pandas as pd
import google.generativeai as genai
from google.generativeai import types
import nest_asyncio
import asyncio
from tenacity import retry, stop_after_attempt, wait_exponential
from google.generativeai import caching
import datetime
from math import floor, ceil
import time
from tqdm.asyncio import tqdm_asyncio

from eval_utils import MP4_VID_DIRECTORY, TEST_VIDS

class TimeoutError(Exception):
    pass

sema = asyncio.Semaphore(3)

@retry(
    stop=stop_after_attempt(5),  # Retry up to 5 times
    wait=wait_exponential(multiplier=1, min=1, max=60),  # Wait exponentially between retries
)
async def get_response(model, prompt, video_file, max_tokens, timeout):
    async with sema:
        if video_file is None:
            msg = [prompt]
        else:
            msg = [video_file, prompt]
            
        response = await model.generate_content_async(
            msg,
            generation_config=types.GenerationConfig(
                max_output_tokens=max_tokens,
                temperature=0
            ),
            request_options={"timeout": timeout})
        return response

def to_gemini_time_format(seconds):
    return f"{int(seconds)//60:02d}:{int(seconds)%60:02d}"

# convert timestamp column from string to tuple of floats
def convert_timestamp(timestamp):
    timestamp = timestamp[1:-1]
    timestamp = timestamp.split(", ")
    timestamp = tuple(map(float, timestamp))
    return timestamp

def parse_args():
    parser = argparse.ArgumentParser(description="Script to process frames in a video.")
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

async def main():
    nest_asyncio.apply()
    args = parse_args()
    genai.configure(api_key=os.environ['GEMINI_API_KEY'])
    
    save_file = f'gemini_text-only_{args.text_only}_annotation_generated.csv'

    # get the root directory of the project for notebook
    root = Path(__file__).parent.parent.parent
    data_folder = root / 'data'
    
    print(data_folder)
    if os.path.exists(data_folder / save_file):
        df = pd.read_csv(data_folder / save_file)
    else:
        df = pd.read_csv(data_folder / 'annotation_final.csv')
        # adds column to dataframe with the generated response
        df['generated_response'] = None

    # adds new column `augmented question` that includes the timestamp
    df["timestamp"] = df["timestamp"].apply(lambda x: convert_timestamp(x) if pd.notnull(x) else x)
    template = "From timestamp {time_start} to {time_end}, {question}"
    global_question_types = ['Social Judgment', 'Working Memory', 'Perspective Taking']
    df['augmented_question'] = df.apply(
        lambda x: template.format(
            time_start = to_gemini_time_format(floor(x['timestamp'][0])),
            time_end = to_gemini_time_format(ceil(x['timestamp'][1])),
            question = x['question']
        ) if not x['question type'] in global_question_types else x['question'] , axis=1
    )
    
    # Evaluation template
    template = """You are an expert in mime performance understanding and question answering. 
    Typically, the mime would use exaggerated gestures or pretend objects to convey a message.
    Answer the question in one sentence using the video, with brief explanations. 
    Do not describe the frames just answer the question, and say nothing else.
    If the mime is using imaginary objects, describe the objects as if they were real.
    Question: {question}"""

    videos = list(df['file'].unique())
    eval_videos = TEST_VIDS if args.debug else videos
    print(eval_videos)
    video_map = {}
    
    start_from = 0
    delay_count = 0
    
    for n_vid, vid_name in enumerate(eval_videos[start_from:]):
        base_name, _ = os.path.splitext(vid_name)
        vid_name_mp4 = base_name + ".mp4"
        print("Evaluating: " + vid_name_mp4)
        video_path = os.path.join(MP4_VID_DIRECTORY, vid_name_mp4)
        
        if not args.text_only:
            print(f"Uploading file {video_path}...")
            video_file = genai.upload_file(path=video_path)
            print(f"Completed upload: {video_file.uri}")
            video_map[vid_name] = video_file
            
            print(f"Checking file {video_file.name}...", end='')
            while video_file.state.name == "PROCESSING":
                print('.', end='')
                time.sleep(5)
                video_file = genai.get_file(video_file.name)

            if video_file.state.name == "FAILED":
                raise ValueError(video_file.state.name)
            print("Done")
            video_map[vid_name] = video_file
    
            cache = caching.CachedContent.create(
                model='models/gemini-1.5-pro-002',
                display_name=vid_name, # used to identify the cache
                contents=[video_file],
                ttl=datetime.timedelta(minutes=10),
            )

            model = genai.GenerativeModel.from_cached_content(cache)
        else:
            model = genai.GenerativeModel(model_name="gemini-1.5-pro")
            video_file = None
            
            
        # for each row in the dataframe, generate response
        tasks = []
        df_video = df[df['file'] == vid_name]
        for index, row in df_video.iterrows():
            question = row['augmented_question']
            prompt = template.format(question=question)
            tasks.append(get_response(model, prompt, video_file, args.max_tokens, args.timeout))
        
        responses = await tqdm_asyncio.gather(*tasks, desc=f"Generating responses for {start_from + n_vid}")
        for df_index, response in zip(df_video.index, responses):
            df.at[df_index, 'generated_response'] = response.text

        # temporarily save the dataframe
        df.to_csv(data_folder / save_file, index=False)
        
        delay_count += 1
        
        if not args.text_only:
            cache.delete()
            if delay_count % 10 == 0:
                print("reset quota...sleep for 2 minutes")
                await asyncio.sleep(120)
            else:
                print("waiting for 30 seconds...")
                await asyncio.sleep(30)

asyncio.run(main())




