from llava.model.builder import load_pretrained_model
from llava.mm_utils import tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from llava.conversation import conv_templates
from pathlib import Path
import copy
import warnings
from decord import VideoReader, cpu
import numpy as np
import os
import argparse
warnings.filterwarnings("ignore")

from eval_utils import get_augmented_questions, MP4_VID_DIRECTORY, TITLE_MAPPING_DIRECTORY, TEST_VIDS
from decord import VideoReader, cpu
import numpy as np
import json

def get_video_frames(video_path, max_frames_num=384, force_sample=False):
    """
    If it's a .webm or if you just want to standardize everything to MP4,
    re-encode and then decode with decord.
    """
    vr = VideoReader(video_path, ctx=cpu(0))
    total_frames = len(vr)
    video_time = total_frames / vr.get_avg_fps()
    fps = round(vr.get_avg_fps())
    frame_idx = [i for i in range(0, len(vr), fps)]
    # Do 1 frame per second
    frame_time = [i/fps for i in frame_idx]
    if len(frame_idx) > max_frames_num or force_sample:
        # If # of frames exceeds max_num_frames, sample uniformly
        sample_fps = max_frames_num
        uniform_sampled_frames = np.linspace(0, total_frames - 1, sample_fps, dtype=int)
        frame_idx = uniform_sampled_frames.tolist()
        frame_time = [i/vr.get_avg_fps() for i in frame_idx]
    frames = vr.get_batch(frame_idx).asnumpy()
    video_time = to_video_llava_time_format(video_time)
    frame_time = ",".join([to_video_llava_time_format(timestamp) for timestamp in frame_time])
    return video_path, video_time, frames, frame_time

def to_video_llava_time_format(seconds):
    minutes = int(seconds // 60)
    remaining_seconds = seconds % 60
    return f"{minutes:02} min:{remaining_seconds:.2f} sec"
    # return f"{seconds:.2f}s"


def get_evaluation_template(title=None):
     # Evaluation template
    template = """You are an expert in mime performance understanding and question answering. 
    Typically, the mime would use exaggerated gestures or pretend objects to convey a message.
    The video lasts for {video_time}, and {num_frames} frames are uniformly sampled from it.
    These frames are located at {frame_time}. Answer the question in one sentence using the video, with brief explanations. 
    Do not describe the frames just answer the question. 
    If the mime is using imaginary objects, describe the objects as if they were real.
    """ + (f"The title of the video is {title}\n" if title is not None else "\n") + "Question: {question}"
    return template


def check_videos():
    broken_vids = []
    # max_frames_num = 64
    print(len(os.listdir(MP4_VID_DIRECTORY)))
    for vid_name_mp4 in sorted(os.listdir(MP4_VID_DIRECTORY)):
        print(vid_name_mp4)
        # base_name, _ = os.path.splitext(vid_name)
        # vid_name_mp4 = base_name + ".mp4"
        # print(vid_name_mp4)
        video_path = os.path.join(MP4_VID_DIRECTORY, vid_name_mp4)
        try:
            video_path, video_time, video, frame_time = get_video_frames(video_path, args.max_frames_num)
        except (RuntimeError):
            print(f"Broken: {vid_name_mp4}")
            broken_vids.append(vid_name_mp4)
    print(broken_vids)

def parse_args():
    parser = argparse.ArgumentParser(description="Script to process frames in a video.")
    parser.add_argument(
        "--max_frames_num",
        type=int,
        default=64,  
        help="Maximum number of frames to process (0 for all frames)."
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=128,
    )
    parser.add_argument(
        "--text_only",
        action="store_true",
        default=False
    )
    parser.add_argument(
        "--with_titles",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        default=False,
    )
    return parser.parse_args()

if __name__ == "__main__":
    # IMPORTANT: make sure to do export DECORD_EOF_RETRY_MAX=20480
    # Parse args
    args = parse_args()
    # Get model
    pretrained = "lmms-lab/LLaVA-Video-72B-Qwen2"
    model_name = "llava_qwen"
    device = "cuda"
    device_map = "auto"
    tokenizer, model, image_processor, max_length = load_pretrained_model(pretrained, None, model_name, torch_dtype="bfloat16", device_map=device_map)  # Add any other thing you want to pass in llava_model_args
    model.eval()
    # Get data
    root = Path(__file__).parent.parent.parent
    data_folder = root / 'data'
    print(data_folder)
    df = get_augmented_questions(data_folder, to_video_llava_time_format)
    videos = list(df['file'].unique())
    eval_videos = TEST_VIDS if args.debug else videos
    # loads title mapping
    with open(TITLE_MAPPING_DIRECTORY, 'r') as f:
        title_mapping = json.load(f)

    for n_vid, vid_name in enumerate(eval_videos):
        base_name, _ = os.path.splitext(vid_name)
        vid_name_mp4 = base_name + ".mp4"
        print(vid_name_mp4)
        title=None
        if args.with_titles:
            title = title_mapping[vid_name]
            print(title)
        video_path = os.path.join(MP4_VID_DIRECTORY, vid_name_mp4)
        # Extract frames 
        video_path, video_time, video, frame_time = get_video_frames(video_path, args.max_frames_num)
        video = image_processor.preprocess(video, return_tensors="pt")["pixel_values"].cuda().bfloat16()
        conv_template = "qwen_1_5"  # Make sure you use correct chat template for different models
        responses = []
        df_video = df[df['file'] == vid_name]
        for index, row in df_video.iterrows():
            # Get the prompt 
            question = row['augmented_question']
            template = get_evaluation_template().format(video_time=video_time, question=question, 
                                                        num_frames=len(video), frame_time=frame_time, title=None)
            if args.text_only:
                # Text only prompt
                question = template
                input_images = None
            else:
                question = DEFAULT_IMAGE_TOKEN + template
                input_images = video
            conv = copy.deepcopy(conv_templates[conv_template])
            conv.append_message(conv.roles[0], question)
            conv.append_message(conv.roles[1], None)
            prompt_question = conv.get_prompt()
            input_ids = tokenizer_image_token(prompt_question, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(device)
            cont = model.generate(
                input_ids,
                images=input_images,
                modalities= ["video"],
                do_sample=False,
                temperature=0,
                max_new_tokens=args.max_tokens,
            )
            text_outputs = tokenizer.batch_decode(cont, skip_special_tokens=True)[0].strip()
            df.at[index, 'generated_response'] = text_outputs
        df.to_csv(data_folder / f'llava_video_max-frames_{args.max_frames_num}_text-only_{args.text_only}_title_{args.with_titles}_annotation_generated.csv', index=False)