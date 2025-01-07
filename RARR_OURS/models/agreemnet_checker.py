import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from prompts.rarr_freehal_prompt import AGREEMENT_PROMPT
import pandas as pd
import torch
import random
from tqdm import tqdm
import time
import transformers

# fixed seed
FIXED_SEED = 42
torch.manual_seed(FIXED_SEED)
torch.cuda.manual_seed(FIXED_SEED)
torch.cuda.manual_seed_all(FIXED_SEED)
random.seed(FIXED_SEED)

class AgreementChecker:
    def __init__(self, args, pipeline, tokenizer):
        self.args = args        
        self.pipeline = pipeline
        self.tokenizer = tokenizer
        
        transformers.set_seed(FIXED_SEED)
        print("[Agreement Checker] Initialized with provided model and tokenizer.\n")
        
        
    def generating(self, inputs: str):
        len_input = len(inputs)
        
        results = self.pipeline(
            inputs,
            max_new_tokens = 500,
            repetition_penalty = 1.0,
            top_p = 1.0,
            do_sample = False
        )
        
        # 생성된 텍스트 가져오기
        outputs = results[0]["generated_text"][len_input:]
        
        # "\n\n"에서 텍스트를 잘라내기 -> stop 후처리
        if "\n\n" in outputs:    
            outputs = outputs.split("\n\n")[0]

        return outputs
    
    def agreement_check(self, data: pd.DataFrame):
        print("[Agreement Checker] Chekcing agreement for retrieved doc ...\n")
        
        atomic_text_list = data['atomic_text']
        query_list = data['query']
        selected_evd_list = data['selected_evidence']
        agreement_list = []
        agreement_latency_list = []
        reasoning_list = []
        
        for atomic_text, query, evd in tqdm(zip(atomic_text_list, query_list, selected_evd_list)):
            start_time = time.time()
            
            agreement_prompt = AGREEMENT_PROMPT % (atomic_text, query, evd)
            outputs = self.generating(agreement_prompt)
            
            if "- Reasoning: " in outputs:
                # Reasoning 부분 추출
                reasoning = outputs.split("- Reasoning: ")[-1].split("\n- Therefore: ")[0].strip()
                # Therefore 부분 추출
                outputs = outputs.split("- Therefore: ")[-1].strip()
                # Agreement 상태 결정
                if "disagrees" in outputs:
                    agreement = "Hallucination"
                else:
                    agreement = "not Hallucination"
                    
            end_time = time.time()
            latency = end_time - start_time
            
            reasoning_list.append(reasoning)
            agreement_list.append(agreement)
            agreement_latency_list.append(latency)

        
        data['reasoning'] = reasoning_list
        data['agreement'] = agreement_list
        data['agreement_latency'] = agreement_latency_list
        
        # 파일 경로 생성
        output_dir = "./outputs"
        os.makedirs(output_dir, exist_ok=True) 
        output_file = os.path.join(output_dir, f"{self.args.dataset}_agreement_check.csv")

        # 데이터프레임 저장
        data.to_csv(output_file, index=False, encoding='utf-8-sig')
            
        
        print("[Agreement Checker] Completed agreement check.")
        
        return data