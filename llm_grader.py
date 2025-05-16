# Grade LLM with a human baseline
import argparse
import os
import openai
from pydantic import BaseModel
from openai_utils import LLM, create_message
from tqdm.asyncio import tqdm
from pathlib import Path
import asyncio
import nest_asyncio
import pandas as pd
nest_asyncio.apply()

class Evaluation(BaseModel):
    correct: bool
    explanation: str
    
openai.api_key = os.getenv("MIT_OPENAI_API_KEY")

    
GRADE_INSTRUCTION = """
Answer Grading Instructions:
Carefully consider the following question and answers regarding understanding of a mime performance.
You will be shown a "gold-standard" answer from a human annotator, referred to as the "Reference Answer", and a "Candidate Answer".
Your task is to determine whether the candidate answer is a good answer in place of the "gold" reference using the following criteria:

1. The candidate directly answers the question without deviation or misunderstanding.
2. The candidate does not contain misleading information and does not hallucinate story plots not present in the reference answer.
3. Since the videos are mime performances, invisible actions, objects, or the mime actor portraying objects should be considered correct if and only if they are relevant to the question.
4. The candidate answer can be a good answer in place of the reference answer even if they are not semantically equivalent, as long as they are 
in the same ballpark, given that there can be multiple interpretations of a mime acting out invisible actions or objects.

Your response should be one word, "TRUE" or "FALSE", and a brief explanation of your decision. You should respond "TRUE" if the candidate is a good answer in place of the reference answer, and "FALSE" otherwise.
"""

GRADE_PROMPT = """
Question:
"{question}"
Candidate Answer:
"{candidate_answer}"
Reference Answer:
"{ref_answer}"
Equivalent? (True/False, and why?):
"""
    
async def check_answer(question: str, candidate_answer: str, reference_answer: str) -> Evaluation:
    grader = LLM(llm_str = "gpt-4o", default_instructions=GRADE_INSTRUCTION) #gpt4o
    prompt = GRADE_PROMPT.format(question=question, candidate_answer=candidate_answer, ref_answer=reference_answer,)
    message = create_message("user", prompt)
    response = await grader.parse_completion([message], response_format=Evaluation)
    return response

# iterate through each row and evaluate the candidate answer
async def evaluate_worker(semaphore, question: str, candidate_answer: str, reference_answer):
    async with semaphore:
        response = await check_answer(question, candidate_answer, reference_answer)
        evaluation = response.choices[0].message.parsed
        await asyncio.sleep(2)
        return evaluation

async def evaluate_all(reference_df, df):
    max_workers = 10
    semaphore = asyncio.Semaphore(max_workers)
    tasks = []
    indices = []
    for idx, row in df.iterrows():
        question = row["augmented_question"]
        candidate_answer = row["generated_response"]
        reference_answer = reference_df.loc[idx, "reference_answer"]
        tasks.append(evaluate_worker(semaphore, question, candidate_answer, reference_answer))
        indices.append(idx)
    
    results = await tqdm.gather(*tasks, desc=f"Evaluating questions")
    
    for idx, result in zip(indices, results):
        df.loc[idx, "Auto Eval"] = result.correct
        df.loc[idx, "Auto Eval Explanation"] = result.explanation

def parse_args():
    parser = argparse.ArgumentParser(description="Script to automatically grade answers")
    parser.add_argument(
        "--results_pref",
        type=str,
        default=None,  
        help="Prefix of results csv (assumed to end with _annotation_generated.csv)"
    )
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    root = Path(__file__).parent.parent
    data_dir = root / "data"
    reference_df = pd.read_csv(data_dir / f"annotation_final_human.csv")
    df = pd.read_csv(data_dir / f"{args.results_pref}_annotation_generated.csv")
    df["Auto Eval"] = None
    df["Auto Eval Explanation"] = None
    asyncio.run(evaluate_all(reference_df, df))
    output_dir = root / "data_human"
    os.makedirs(output_dir, exist_ok=True)
    df.to_csv(output_dir / f"{args.results_pref}_annotation_auto_eval_evaluated.csv", index=False)