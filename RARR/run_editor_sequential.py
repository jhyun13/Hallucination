"""Runs the RARR editor on a JSONL file of claims.

Runs question generation, retrieval, agreement gate, and editing on a file with claims
using Llama-3.1-8B and Wiki search.
"""
import argparse
import json
import os
from typing import Any, Dict

import jsonlines
import Levenshtein
import tqdm
import time
import sys
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)


from prompts.rarr_prompt import (QGEN_PROMPT, AGREEMENT_GATE_PROMPT, EDITOR_PROMPT)

from models import (
    agreement_gate,
    editor,
    evidence_selection,
    search,
    question_generation,
)


def run_editor_one_instance(
    claim: str,
    model_path: str,
    search_url: str,
    num_rounds_qgen: int = 3,
    max_search_results_per_query: int = 5,
    max_sentences_per_passage: int = 4,
    sliding_distance: int = 1,
    max_passages_per_search_result: int = 1,
    max_evidences_per_question: int = 1,
    max_edit_ratio: float = 100,
) -> Dict[str, Any]:
    
    # Load tokenizer and model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path).to(device)
    
    original_claim = claim
    agreement_gates = []

    # 1. Generate questions for the claim
    questions = question_generation.run_rarr_question_generation(
        claim=claim,
        model=model,
        tokenizer = tokenizer,
        prompt=QGEN_PROMPT,
        num_rounds=num_rounds_qgen
    )

    # 2. Run search on generated question for the claim
    evidences_for_questions = [
        search.run_search(
            query=query,
            search_url = search_url,
            max_search_results_per_query=max_search_results_per_query,
            max_sentences_per_passage=max_sentences_per_passage,
            sliding_distance=sliding_distance,
            max_passages_per_search_result_to_return=max_passages_per_search_result,
        )
        for query in questions
    ]

    # 3. Flatten the evidences per question into a single list.
    used_evidences = [
        e
        for cur_evids in evidences_for_questions
        for e in cur_evids[:max_evidences_per_question]
    ]
    
    print(f"used_evidences ::: {used_evidences}\n")

    # 4. Iterative editing over each evidence
    revision_steps = []
    for evid in used_evidences:
        # 5. Run the agreement gate on the current (claim, context, query, evidence) tuple
        gate = agreement_gate.run_agreement_gate(
            claim=claim,
            query=evid["query"],
            model=model,
            tokenizer=tokenizer,
            evidence=evid["text"],
            prompt=AGREEMENT_GATE_PROMPT
        )
        agreement_gates.append(gate)

        # 6. Run the editor gate if the agreement gate is open
        if gate["is_open"]:
            edited_claim = editor.run_rarr_editor(
                claim=claim,
                model = model,
                tokenizer = tokenizer,
                query=evid["query"],
                evidence=evid["text"],
                prompt=EDITOR_PROMPT,
            )["text"]

            # Don't keep the edit if the editor makes a huge change
            if Levenshtein.distance(claim, edited_claim) / len(claim) <= max_edit_ratio:
                claim = edited_claim

        revision_steps.append({"text": claim})

    result = {
        "text": original_claim,
        "questions": questions,
        "evidences_for_questions": evidences_for_questions,
        "revisions": [
            {
                "original_text": original_claim,
                "revised_text": revision_steps[-1]["text"],
                "evidences": used_evidences,
                "agreement_gates": agreement_gates,
                "revision_steps": revision_steps,
            }
        ],
    }
    selected_evidences = evidence_selection.select_evidences(result)
    result["selected_evidences"] = selected_evidences
    
    return result


def get_args() -> argparse.Namespace:
    """Gets command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_file",
        type=str,
        required=True,
        help="JSONLines file of claims to run RARR on.",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        required=True,
        help="JSONLines file to write revisions to.",
    )
    parser.add_argument(
        "--claim_field",
        default="model_outputs_explanation",
        type=str,
        help="Field of the JSONL file to run the claim editing on.",
    )
    parser.add_argument(
        "--model",
        default="meta-llama/Llama-3.1-8B-Instruct",
        type=str,
        help="meta Llama3.1 8B model to use.",
    )
    parser.add_argument(
        "--search_url",
        default="http://127.0.0.1:8000/",
        type=str,
        help="Search API endpoint URL",
    )
    parser.add_argument(
        "--num_rounds_qgen",
        default=3,
        type=int,
        help="Number of times to re-sample queries for a claim.",
    )
    parser.add_argument(
        "--hallucinate_evidence",
        action="store_true",
        help="If this flag is set, we hallucinate evidence instead of retrieving it. "
        "This flag should NEVER be set when trying to improve attribution as evidence  "
        "may be inaccurate and is only provided to quickly experiment with repository "
        "setting up the search API first.",
    )
    parser.add_argument(
        "--max_search_results_per_query",
        default=5,
        type=int,
        help="Maximum number of search results we get per query.",
    )
    parser.add_argument(
        "--max_sentences_per_passage",
        default=4,
        type=int,
        help="Maximum number of sentences per evidence passage.",
    )
    parser.add_argument(
        "--sliding_distance",
        default=1,
        type=int,
        help="Sliding window distance for extracting passages from a search result.",
    )
    parser.add_argument(
        "--max_passages_per_search_result",
        default=1,
        type=int,
        help="Maximum number of passages to return for each search result. A passage"
        " ranker is applied to get the top passages per query.",
    )
    parser.add_argument(
        "--max_evidences_per_question",
        default=1,
        type=int,
        help="Maximum number of evidences to consider per question.",
    )
    parser.add_argument(
        "--max_edit_ratio",
        default=100,
        type=float,
        help="Maximum edit ratio between claim and edit for each round.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resumes the editing process if broken by loading the output file.",
    )
    args = parser.parse_args()

    # Write all args to file
    with open(args.output_file + "_args", "w", encoding="utf-8") as writer:
        json.dump(args.__dict__, writer, indent=4)
    return args


def main() -> None:
    """Loads a RARR evaluation set and runs GPT-3 RARR editing."""
    args = get_args()

    # Load the finished results by mapping from the claim name to the results.
    if args.resume and os.path.exists(args.output_file):
        print(f"Resuming with results from {args.output_file}")
        finished_results = {
            l["input_info"][args.claim_field]: l["result"]
            for l in jsonlines.open(args.output_file)
        }
        print(f"Found {len(finished_results)} finished lines.")
    else:
        finished_results = None

    with open(args.output_file, "w", encoding="utf-8") as writer:
        lines = list(jsonlines.open(args.input_file))
        for line in tqdm.tqdm(lines):
            # latency 측정 위함
            start_time = time.time()
            
            claim = line["input_info"]["model_outputs_explanation"]

            # Search for finished result
            if finished_results and claim in finished_results:
                line["result"] = finished_results[claim]
            else:
                line["result"] = run_editor_one_instance(
                    claim=claim,
                    model_path=args.model,
                    search_url=args.search_url,
                    num_rounds_qgen=args.num_rounds_qgen,
                    max_search_results_per_query=args.max_search_results_per_query,
                    max_sentences_per_passage=args.max_sentences_per_passage,
                    sliding_distance=args.sliding_distance,
                    max_passages_per_search_result=args.max_passages_per_search_result,
                    max_evidences_per_question=args.max_evidences_per_question,
                    max_edit_ratio=args.max_edit_ratio
                )
            end_time = time.time()
            latency = end_time - start_time
            
            line["result"]["total_latency"] = latency
            
            writer.write(json.dumps(line, indent=4, ensure_ascii=False) + "\n")


    # json파일 csv 파일로 변경해서 저장하기 (진행 중)

if __name__ == "__main__":
    main()