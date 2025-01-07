import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from prompts.shared_prompt import MERGE_PROMPT
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

class Merger:
    def __init__(self, args, pipeline, tokenizer):
        self.args = args        
        self.pipeline = pipeline
        self.tokenizer = tokenizer
        
        transformers.set_seed(FIXED_SEED)
        print("[Merger] Initialized with provided model and tokenizer.\n")
        
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
    
    
    def merge_atomic_text(self, data: pd.DataFrame):
        print("[Merger] Merging revised texts ...")
        
        input_text = data['input_text']
        merged_text_list = []
        merge_latency_list = []
        
        # input_text를 기준으로 edit_text를 "/ "로 결합
        data['combined_revised_text'] = data.groupby('input_text')['revised_text'].transform(lambda x: '/ '.join(x))
        
        # 중복 제거를 위해 unique한 행만 남기기
        unique_data = data[['input_text', 'combined_revised_text']].drop_duplicates().reset_index(drop=True)

        # for 루프 실행
        for _, row in unique_data.iterrows():
            start_time = time.time()
            
            input_text = row['input_text']
            combined_text = row['combined_revised_text']
            merged_prompt = MERGE_PROMPT % (input_text, combined_text)
            outputs = self.generating(merged_prompt)
            if "\n- My merge: " in outputs:
                outputs = outputs.split("\n- My merge: ")[-1].strip()
            else:
                outputs = outputs.strip()
            
            # 원래 데이터프레임의 input_text 개수만큼 복제
            num_occurrences = (data['input_text'] == input_text).sum()
            merged_text_list.extend([outputs] * num_occurrences)
            
            end_time = time.time()
            latency = end_time - start_time
            merge_latency_list.extend([latency] * num_occurrences)
        
        # 결과를 원래 데이터프레임에 추가
        data['merged_text'] = merged_text_list
        data['merge_latency'] = merge_latency_list
        
        # 파일 경로 생성
        output_dir = "./outputs"
        os.makedirs(output_dir, exist_ok=True) 
        output_file = os.path.join(output_dir, f"{self.args.dataset}_merge.csv")

        # 데이터프레임 저장
        data.to_csv(output_file, index=False, encoding='utf-8-sig')
        
        print("[Merger] Merging complete.")

        return data