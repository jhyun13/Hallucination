import argparse
import os
import sys
import pandas as pd
import time
import tqdm
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from prompts.rarr_prompt import QGEN_PROMPT, AGREEMENT_GATE_PROMPT, EDITOR_PROMPT
from prompts.shared_prompt import MERGE_PROMPT
from models import agreement_gate, editor, evidence_selection, search, question_generation, merger


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
):
    # Load tokenizer and model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path).to(device)
    
    original_claim = claim
    # agreement_gates = []
    agreement = []
    reasoning = []

    # 1. Generate questions for the claim
    questions = question_generation.run_rarr_question_generation(
        claim=claim,
        model=model,
        tokenizer=tokenizer,
        prompt=QGEN_PROMPT,
        num_rounds=num_rounds_qgen
    )
    
    print(f"question :: {questions}\n\n")

    # 2. Run search on generated questions
    evidences_for_questions = [
        search.run_search(
            query=query,
            search_url=search_url
        )
        for query in questions
    ]
    
    print(f"evidences_for_questions :: {evidences_for_questions}\n\n")

    # 3. Iterative editing over each evidence
    revision_steps = []
    for qry, evid in zip(questions, evidences_for_questions):
        # Run the agreement gate
        gate = agreement_gate.run_agreement_gate(
            claim=claim,
            query=qry,
            model=model,
            tokenizer=tokenizer,
            evidence=evid,
            prompt=AGREEMENT_GATE_PROMPT
        )
        # agreement_gates.append(gate)
        reasoning.append(gate["reason"])

        # Run the editor gate if the agreement gate is open
        if gate["is_open"]:
            agreement.append("Hallucination")
            
            edited_claim = editor.run_rarr_editor(
                claim=claim,
                model=model,
                tokenizer=tokenizer,
                query=qry,
                evidence=evid,
                prompt=EDITOR_PROMPT,
            )

            # Don't keep the edit if the editor makes a huge change
            if abs(len(claim) - len(edited_claim)) / len(claim) <= max_edit_ratio:
                claim = edited_claim
        else :
            agreement.append("not Hallucination")

        revision_steps.append(claim)
    
    # 4. merge revised texts
    if len(revision_steps) > 1:
        merge_text = merger.run_merge(
            claim=original_claim,
            revision_steps=revision_steps,
            model=model,
            tokenizer=tokenizer,
            prompt=MERGE_PROMPT
        )
    else:
        merge_text = revision_steps[0] if revision_steps else claim    

    result = {
        "text": original_claim,
        "questions": questions,
        "evidences_for_questions": evidences_for_questions,
        "revised_text": revision_steps,
        "evidences": evidences_for_questions,
        "agreement": agreement,
        "reason": reasoning,
        "merge_text": merge_text
    }
    # selected_evidences = evidence_selection.select_evidences(result)
    # result["selected_evidences"] = selected_evidences
    
    return result


def get_args():
    """Gets command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, required=True, help="Input CSV file of claims.")
    parser.add_argument("--output_file", type=str, required=True, help="Output CSV file to write results.")
    parser.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct", type=str, help="Model path.")
    parser.add_argument("--search_url", default="http://127.0.0.1:8000/", type=str, help="Search API endpoint URL.")
    parser.add_argument("--num_rounds_qgen", default=3, type=int, help="Number of question generation rounds.")
    parser.add_argument("--max_search_results_per_query", default=5, type=int, help="Maximum search results per query.")
    parser.add_argument("--max_sentences_per_passage", default=4, type=int, help="Maximum sentences per passage.")
    parser.add_argument("--sliding_distance", default=1, type=int, help="Sliding window distance for passages.")
    parser.add_argument("--max_passages_per_search_result", default=1, type=int, help="Maximum passages per search result.")
    parser.add_argument("--max_evidences_per_question", default=1, type=int, help="Maximum evidences per question.")
    parser.add_argument("--max_edit_ratio", default=100, type=float, help="Maximum edit ratio for claims.")
    args = parser.parse_args()
    return args


def main():
    args = get_args()

    # Load input CSV file
    data = pd.read_csv(args.input_file)

    # List to store results
    all_results = []

    # Process each claim
    for idx, row in tqdm.tqdm(data.iterrows(), total=len(data)):
        claim = row['input']

        start_time = time.time()
        result = run_editor_one_instance(
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

        # Add latency and results
        result["total_latency"] = end_time - start_time
        all_results.append({
            "claim": result["text"],
            "questions": result["questions"],
            "evidences": result["evidences_for_questions"],
            "revised_text": result["revised_text"],
            "agreement": result["agreement"],
            "reason": result["reason"],
            "merge_text": result["merge_text"],
            "total_latency": result["total_latency"],
        })

    # Save results to CSV
    result_df = pd.DataFrame(all_results)
    result_df.to_csv(args.output_file, index=False, encoding="utf-8-sig")
    print(f"Results saved to {args.output_file}")


if __name__ == "__main__":
    main()

# python run_editor_sequential_csv.py --input_file ../data/nq/nq_data2.csv --output_file ./outputs/nq_finish.csv