import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from prompts.cove_freehal_prompt import EXECUTE_VERIFICATION_PROMPT
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


class ExecutionVerifier:
    def __init__(self, args, pipeline, tokenizer):
        self.args = args        
        self.pipeline = pipeline
        self.tokenizer = tokenizer
        
        transformers.set_seed(FIXED_SEED)
        print("[Execution Verifier] Initialized with provided model and tokenizer.\n")
        
        
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
    
    def execute_verification(self, data: pd.DataFrame):
        print("[Execution Verifier] Executing verification for plan  ...\n")
        
        atomic_text_list = data['atomic_text']
        plan_list = data['plan']
        reasoning_list = []
        agreement_list = []
        agreement_latency_list = []
        
        for atomic_text, plan in tqdm(zip(atomic_text_list, plan_list)):
            reasoning = None
            agreement = None
            start_time = time.time()
            
            agreement_prompt = EXECUTE_VERIFICATION_PROMPT % (atomic_text, plan)
            outputs = self.generating(agreement_prompt)            
            
            if "- Reasoning: " in outputs and "- Therefore: " in outputs:
                try:
                    # Reasoning 부분 추출
                    reasoning_start = outputs.find("- Reasoning: ") + len("- Reasoning: ")
                    reasoning_end = outputs.find("\n- Therefore: ")
                    reasoning = outputs[reasoning_start:reasoning_end].strip()

                    # Therefore 부분 추출
                    therefore_start = outputs.find("- Therefore: ") + len("- Therefore: ")
                    outputs = outputs[therefore_start:].strip()

                    # Agreement 상태 결정
                    if "disagrees" in outputs.lower():
                        agreement = "Hallucination"
                    else:
                        agreement = "not Hallucination"

                except Exception as e:
                    # 예외 처리 및 디버깅 정보 출력
                    print(f"Error processing outputs: {outputs}")
                    print(f"Exception: {e}")
                    reasoning = None
                    agreement = "Unknown"
            else:
                # 필요한 키워드가 없을 경우 기본값 설정
                print(f"Outputs missing required sections: {outputs}")
                reasoning = None
                agreement = "Unknown"

                    
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
        output_file = os.path.join(output_dir, f"{self.args.dataset}_execute_verification.csv")

        # 데이터프레임 저장
        data.to_csv(output_file, index=False, encoding='utf-8-sig')
          
        print("[Execution Verifier] Execution & Verification complete.\n")

        return data