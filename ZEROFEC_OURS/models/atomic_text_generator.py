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
        
        input = data['input']
        atomic_results = []  # atomic 결과를 저장할 리스트
        
        start_time = time.time()
        atomic_prompt = ATOMIC_TEXT_PROMPT % input
        outputs = self.generating(atomic_prompt)
            
        if "- I decomposed: " in outputs:
            atomic_results = outputs.split("- I decomposed: ")
        else:
            print(f"Error : Atomic Generator")
            atomic_results = [outputs]
                
        # 결과 처리
        atomic_results = [f.replace("\n", "") for f in atomic_results if f != ""]
        
            
        end_time = time.time()
        latency = end_time - start_time
        
        # 결과를 데이터프레임으로 변환
        result_df = pd.DataFrame({
            'input_text': input,
            'atomic_text': atomic_results,
            'atomic_latency': latency
        })
    
        print("[Atomic Text Generator] Atomic text generation complete.\n")

        return result_df