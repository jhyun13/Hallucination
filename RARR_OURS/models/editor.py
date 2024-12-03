import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from prompts.rarr_freehal_prompt import REVISION_PROMPT
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

class Editor:
    def __init__(self, args, pipeline, tokenizer):
        self.args = args        
        self.pipeline = pipeline
        self.tokenizer = tokenizer
        
        transformers.set_seed(FIXED_SEED)
        print("[Editor] Initialized with provided model and tokenizer.\n")
        
        
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
        print("[Editor] Editing text ...")
        
        atomic_text_list = data['atomic_text']
        query_list = data['query']
        selected_evd_list = data['selected_evidence']
        
        revised_text_list = []
        revision_latency_list = []
        
        for atomic_text, query, selected_evd in tqdm(zip(atomic_text_list, query_list, selected_evd_list)):
            start_time = time.time()
            
            revision_prompt = REVISION_PROMPT % (atomic_text, query, selected_evd)
            outputs = self.generating(revision_prompt)
            
            if "- My fix:" in outputs:
                outputs = outputs.split("- My fix:")[1].strip()
            
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
        
        print("[Editor] Editing complete.")
        
        return data