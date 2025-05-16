import pandas as pd
import os
from pathlib import Path
import json


MP4_VID_DIRECTORY = "/scratch/projects/mime_bench/mime_vids/mp4"
SIQ_VID_DIRECTORY = "/scratch/Social-IQ-2.0-Challenge/siq2/video"
TITLE_MAPPING_DIRECTORY = "/scratch/projects/mime_bench/mime_vids/file_mapping.json"
MODEL_DIRECTORY = "/scratch/hengzhil/huggingface"
TEST_VIDS = ["0002.mp4"]


# convert timestamp column from string to tuple of floats
def convert_timestamp(timestamp):
    timestamp = timestamp[1:-1]
    timestamp = timestamp.split(", ")
    timestamp = tuple(map(float, timestamp))
    return timestamp

def get_fine_tune_test_vids(data_folder):
    json_file = data_folder / 'test_files.json'
    with open(json_file, 'r') as f:
        test_files = json.load(f)
    return test_files

def get_siq_fine_tune_test_vids(data_folder):
    json_file = data_folder / 'siq_test_files.json'
    with open(json_file, 'r') as f:
        test_files = json.load(f)
    return test_files

def get_model_local_path(model_name, fine_tune=False):
    model_paths = {
        "Qwen/Qwen2.5-VL-72B-Instruct": "models--Qwen--Qwen2.5-VL-72B-Instruct/",
        "Qwen/Qwen2.5-VL-7B-Instruct": "models--Qwen--Qwen2.5-VL-7B-Instruct/",
        "lmms-lab/LLaVA-Video-72B-Qwen2": "models--lmms-lab--LLaVA-Video-72B-Qwen2/",
        "DAMO-NLP-SG/VideoLLaMA3-7B": "models--DAMO-NLP-SG--VideoLLaMA3-7B",
        "OpenGVLab/InternVL2_5-78B": "models--OpenGVLab--InternVL2_5-78B",
        "rhymes-ai/Aria": "models--rhymes-ai--Aria"
    }
    if model_name in model_paths:
        if model_name == "Qwen/Qwen2.5-VL-72B-Instruct" and fine_tune:
            print("Using fine-tuned model")
            return "/scratch/mime_sft/output/qwen25_vl_lora_sft_train"
        else:
            model_cache_dir = Path(MODEL_DIRECTORY) / model_paths[model_name]
            with open(os.path.join(model_cache_dir, "refs", "main"), "r") as f:
                commit_hash = f.read().strip()
            snapshot_path = model_cache_dir / "snapshots" / commit_hash
            return snapshot_path
    else:
        return None


def get_augmented_questions(data_folder, time_format_fn):
    # get the root directory of the project for notebook
    df = pd.read_csv(data_folder / 'annotation_final.csv')
    
    # adds column to dataframe with the generated response
    df['generated_response'] = None
    # adds new column `augmented question` that includes the timestamp
    df["timestamp"] = df["timestamp"].apply(lambda x: convert_timestamp(x) if pd.notnull(x) else x)
    template = "From timestamp {time_start} to {time_end}, {question}"
    global_question_types = ['Social Judgment', 'Working Memory', 'Perspective Taking']
    df['augmented_question'] = df.apply(
        lambda x: template.format(
            time_start = time_format_fn(x['timestamp'][0]),
            time_end = time_format_fn(x['timestamp'][1]),
            question = x['question']
        ) if not x['question type'] in global_question_types else x['question'] , axis=1
    )
    return df

def convert_siq_timestamp(time_range: str):
    start_str, end_str = time_range.split('-')
    start_sec = float(start_str)
    end_sec = float(end_str)
    return start_sec, end_sec


def get_siq_augmented_questions(data_folder, time_format_fn):
    df = pd.read_json(data_folder / "siq" / "qa_test.json", lines=True)
    df['generated_response'] = None
    df['timestamp'] = df['timestamp'].apply(lambda x: convert_siq_timestamp(x) if pd.notnull(x) else x)
    template = "From timestamp {time_start} to {time_end}, {question}"
    df['augmented_question'] = df.apply(
        lambda x: template.format(
            time_start = time_format_fn(x['timestamp'][0]),
            time_end = time_format_fn(x['timestamp'][1]),
            question = x['question']
        ), axis=1
    )
    return df
