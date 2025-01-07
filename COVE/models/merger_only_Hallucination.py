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
    
    
    def merge_text(self, data: pd.DataFrame):
        print("[Merger] Merging revised texts ...")
        
        input_text = data['input_text']
        merged_text_list = []
        merge_latency_list = []
        
        # 중복 제거를 위해 unique한 행만 남기기
        unique_data = data['input_text'].drop_duplicates().reset_index(drop=True)

        for input_text in unique_data:
            
            # input_text에 해당하는 전체 agreement와 revised_text를 가져옴
            agreements = data.loc[data['input_text'] == input_text, 'agreement'].tolist()
            revised_texts = data.loc[data['input_text'] == input_text, 'revised_text'].tolist()

            start_time = time.time()

            if len(agreements) == 1:
                # Case 1: Single agreement
                outputs = revised_texts[0]
            elif all(agr == "not Hallucination" for agr in agreements):
                # Case 2: All "not Hallucination"
                outputs = revised_texts[0]
            else:
                # Case 3: At least one "Hallucination"
                hallucinated_texts = [rev for rev, agr in zip(revised_texts, agreements) if agr == "Hallucination"]
                combined_hallucinated_texts = " / ".join(hallucinated_texts)
                
                merged_prompt = MERGE_PROMPT % (input_text, combined_hallucinated_texts)
                print(f"merged_prompt :::: {merged_prompt}\n\n")
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