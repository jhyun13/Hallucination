import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from prompts.rarr_freehal_prompt import QUERY_GENERATION_PROMPT
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

class QueryGenerator:
    def __init__(self, args, pipeline, tokenizer):
        self.args = args        
        self.pipeline = pipeline
        self.tokenizer = tokenizer
        
        transformers.set_seed(FIXED_SEED)
        print("[Query Generator] Initialized with provided model and tokenizer.\n")
        
        
    def generating(self, inputs: str):
        len_input = len(inputs)
        
        results = self.pipeline(
            inputs,
            max_new_tokens = 500,
            # temperature = 0.0,
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
    
    def generate_query(self, data: pd.DataFrame):
        print("[Query Generator] Generating query for atomic text  ...\n")
        
        atomic_text_list = data['atomic_text']
        query_list = []
        query_latency_list = []
        
        for i, atomic_text in enumerate(tqdm(atomic_text_list)):
            start_time = time.time()
            
            query_prompt = QUERY_GENERATION_PROMPT % atomic_text
            outputs = self.generating(query_prompt)
            
            if "- I googled: " in outputs:
                outputs = outputs.split("- I googled: ")[-1]
            else:
                print(f"Error {i} : Query Generator")
                
            query_list.append(outputs)
            
            end_time = time.time()
            latency = end_time - start_time
            query_latency_list.append(latency)
            
        data['query'] = query_list
        data['query_latency'] = query_latency_list
        
        # 파일 경로 생성
        output_dir = "./outputs"
        os.makedirs(output_dir, exist_ok=True) 
        output_file = os.path.join(output_dir, f"{self.args.dataset}_query_generation.csv")

        # 데이터프레임 저장
        data.to_csv(output_file, index=False, encoding='utf-8-sig')
    
        print("[Query Generator] Query generation complete.\n")

        return data