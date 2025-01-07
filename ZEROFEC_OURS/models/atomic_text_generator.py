import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from prompts.shared_prompt import ATOMIC_TEXT_PROMPT
import torch
import transformers
import numpy as np
import pandas as pd
import random
from tqdm import tqdm
import time

FIXED_SEED = 42
torch.manual_seed(FIXED_SEED)
torch.cuda.manual_seed(FIXED_SEED)
torch.cuda.manual_seed_all(FIXED_SEED)
random.seed(FIXED_SEED)

class AtomicTextGenerator:
    def __init__(self, args, model, tokenizer):
        transformers.set_seed(FIXED_SEED)
        self.args = args        
        self.pipeline = model
        self.tokenizer = tokenizer
        print("[Atomic Text Generator] Initialized with provided model and tokenizer.\n")
        
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
    
    def generate_atomic(self, data:pd.DataFrame):
        print("[Atomic Text Generator] Generating atomic text for input  ...\n")
        
        inputs = data['input']
        latency_list = []
        input_list = []  # hallucinated_input 값을 저장할 리스트
        atomic_results = []  # atomic 결과를 저장할 리스트
        
        for i, hallucinated_input in enumerate(tqdm(inputs)):
            atomic_list = []
            hallucinated_input_list = []
            start_time = time.time()
            
            atomic_prompt = ATOMIC_TEXT_PROMPT % hallucinated_input
            outputs = self.generating(atomic_prompt)
            
            if "- I decomposed: " in outputs:
                atomic_list = outputs.split("- I decomposed: ")
            else:
                print(f"Error {i} : Atomic Generator")
                atomic_list = [outputs]
                
            # 결과 처리
            atomic_list = [f.replace("\n", "") for f in atomic_list if f != ""]
            # hallucinated_input을 atomic_list 개수에 맞게 복제
            hallucinated_input_list.extend([hallucinated_input] * len(atomic_list))

            # 전체 리스트에 추가
            input_list.extend(hallucinated_input_list)
            atomic_results.extend(atomic_list)
            
            end_time = time.time()
            latency = end_time - start_time
            
            latency_list.extend([latency] * len(atomic_list))
        
        # latency = np.mean(latency_list)
        
        # 결과를 데이터프레임으로 변환
        result_df = pd.DataFrame({
            'input_text': input_list,
            'atomic_text': atomic_results,
            'atomic_latency': latency_list,
        })
        
        # 파일 경로 생성
        output_dir = "./outputs"
        os.makedirs(output_dir, exist_ok=True) 
        output_file = os.path.join(output_dir, f"{self.args.dataset}_atomic_text.csv")

        # 데이터프레임 저장
        result_df.to_csv(output_file, index=False, encoding='utf-8-sig')
    
        print("[Atomic Text Generator] Atomic text generation complete.\n")

        return result_df