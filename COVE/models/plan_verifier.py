import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from prompts.cove_prompt import PLAN_VERIFICATION_PROMPT
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
            
        print(f'\\n\\n 후처리한 output:: {outputs}\n\n')

        return outputs
    
    def plan_verification(self, data: pd.DataFrame):
        print("[Plan Verifier] Verificating plan for input text  ...\n")
        
        inputs = data['input']
        input_list = []
        plan_list = []
        plan_latency_list = []
        
        for i, input in enumerate(tqdm(inputs)):
            tmp_input_list = []
            tmp_plan_list = []
            start_time = time.time()
            
            plan_prompt = PLAN_VERIFICATION_PROMPT % input
            outputs = self.generating(plan_prompt)
            
            if "- plan: " in outputs:
                tmp_plan_list = outputs.split("- plan: ")
            else:
                print(f"Error {i} : Plan verifier")
                tmp_plan_list = [outputs]
                
            # 결과 처리
            tmp_plan_list = [f.replace("\n", "").strip() for f in tmp_plan_list if f.strip() != ""]

            tmp_input_list.extend([input] * len(tmp_plan_list))
            
            # 전체 리스트에 추가
            input_list.extend(tmp_input_list)
            plan_list.extend(tmp_plan_list)
            
            end_time = time.time()
            latency = end_time - start_time
            
            plan_latency_list.extend([latency] * len(tmp_plan_list))
            
        result_df = pd.DataFrame({
            'input_text': input_list,
            'plan': plan_list,
            'plan_latency': plan_latency_list            
        })
        
        # 파일 경로 생성
        output_dir = "./outputs"
        os.makedirs(output_dir, exist_ok=True) 
        output_file = os.path.join(output_dir, f"{self.args.dataset}_plan_verification.csv")

        # 데이터프레임 저장
        result_df.to_csv(output_file, index=False, encoding='utf-8-sig')
    
        print("[Plan Verifier] Plan verification complete.\n")

        return result_df