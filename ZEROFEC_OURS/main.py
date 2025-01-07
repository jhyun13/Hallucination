from zerofec_freehal import ZeroFECFreeHal
from types import SimpleNamespace
import argparse
import json
import csv
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument('--input_path', required=True)
parser.add_argument('--output_path', required=True)
parser.add_argument('--dataset', required=True, help="nq(NQ), sqa(SQA), haluqa(HaluQA)")
parser.add_argument('--search_url', default="http://127.0.0.1:8000/", type=str, help="Search API endpoint URL.")
args = parser.parse_args()


def get_fever_model_args():
    
    model_path = 'meta-llama/Llama-3.1-8B-Instruct'

    args = {
        'qg_path': model_path,
        'qa_model_path': model_path,
        'qa_tokenizer_path': model_path,
        'entailment_model_path': '/shared/nas/data/m1/khhuang3/info_correct/docnli/docnli-roberta_pubmed_bioasq.pt',
        'entailment_tokenizer_path':'roberta-large',
        'qa2s_tokenizer_path': model_path,
        'qa2s_model_path': model_path,
        'use_scispacy': False
    }
    # This way the algo can call args with args.model_name, not args["model_name"].
    # So the core code is still compatible with argparse'd arguments
    args = SimpleNamespace(**args)

    return args  


print("Loading FEVER model")
model_args = get_fever_model_args()

zerofec_freehal = ZeroFECFreeHal(model_args)

# CSV 파일 읽어서 각 행을 개별 DataFrame으로 변환
with open(args.input_path, 'r') as f:
    reader = csv.DictReader(f)  # CSV의 각 행을 딕셔너리로 읽음
    inputs = [pd.DataFrame([row]) for row in reader]  # 각 행을 DataFrame으로 변환하여 리스트에 저장


outputs = zerofec_freehal.batch_correct(inputs)

outputs.to_csv(args.output_path, index=False)

# python main.py --input_path ../data/nq/nq_data2.csv --output_path ./outputs/nq_finish.csv