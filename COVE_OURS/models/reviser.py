import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from prompts.cove_freehal_prompt import REVISION_PROMPT
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

class Reviser:
    def __init__(self, args, pipeline, tokenizer):
        self.args = args        
        self.pipeline = pipeline
        self.tokenizer = tokenizer
        
        transformers.set_seed(FIXED_SEED)
        print("[Reviser] Initialized with provided model and tokenizer.\n")
        
        
    # vllm 안쓰는 경우, 허깅페이스에서 모델 불러옴
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
        # print(f'outputs:: {outputs}\n')
        
        # "\n\n"에서 텍스트를 잘라내기 -> stop 후처리
        if "\n\n" in outputs:    
            outputs = outputs.split("\n\n")[0]
            
        # print(f'\\n\\n 후처리한 output:: {outputs}\n\n')

        return outputs
    
    def revise_text(self, data: pd.DataFrame):
        print("[Reviser] Revising text ...")
        
        atomic_text_list = data['atomic_text']
        plan_list = data['plan']
        reasoning_list = data['reasoning']
        agreement_list = data['agreement']
        
        revised_text_list = []
        revision_latency_list = []
        
        for atomic_text, plan, reasoning, agreement in tqdm(zip(atomic_text_list, plan_list, reasoning_list, agreement_list)):
            start_time = time.time()
            
            if agreement == "Hallucination":
                revision_prompt = REVISION_PROMPT % (atomic_text, plan, reasoning)
                outputs = self.generating(revision_prompt)
                
                if "- My fix:" in outputs:
                    outputs = outputs.split("- My fix:")[1].strip()
            else:
                outputs = atomic_text
            
            revised_text_list.append(outputs)
            
            end_time = time.time()
            latency = end_time - start_time
            revision_latency_list.append(latency)
            
        data['revised_text'] = revised_text_list
        data['revision_latency'] = revision_latency_list
        
        # 파일 경로 생성
        output_dir = "./outputs"
        os.makedirs(output_dir, exist_ok=True) 
        output_file = os.path.join(output_dir, f"{self.args.dataset}_revision.csv")

        # 데이터프레임 저장
        data.to_csv(output_file, index=False, encoding='utf-8-sig')
        
        print("[Reviser] Revision complete.")
        
        return data