import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from prompts.cove_freehal_prompt import PLAN_VERIFICATION_PROMPT
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


class PlanVerifier:
    def __init__(self, args, pipeline, tokenizer):
        self.args = args        
        self.pipeline = pipeline
        self.tokenizer = tokenizer
        
        transformers.set_seed(FIXED_SEED)
        print("[Plan Verifier] Initialized with provided model and tokenizer.\n")
        
        
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
    
    def plan_verification(self, data: pd.DataFrame):
        print("[Plan Verifier] Verificating plan for atomic text  ...\n")
        
        atomic_text_list = data['atomic_text']
        plan_list = []
        plan_latency_list = []
        
        for i, atomic_text in enumerate(tqdm(atomic_text_list)):
            start_time = time.time()
            
            query_prompt = PLAN_VERIFICATION_PROMPT % atomic_text
            outputs = self.generating(query_prompt)
            
            if "- plan: " in outputs:
                outputs = outputs.split("- plan: ")[-1]
            else:
                print(f"Error {i} : Plan verifier")
                
            plan_list.append(outputs)
            
            end_time = time.time()
            latency = end_time - start_time
            plan_latency_list.append(latency)
            
        data['plan'] = plan_list
        data['plan_latency'] = plan_latency_list
        
        # 파일 경로 생성
        output_dir = "./outputs"
        os.makedirs(output_dir, exist_ok=True) 
        output_file = os.path.join(output_dir, f"{self.args.dataset}_plan_verification.csv")

        # 데이터프레임 저장
        data.to_csv(output_file, index=False, encoding='utf-8-sig')
    
        print("[Plan Verifier] Plan verification complete.\n")

        return data