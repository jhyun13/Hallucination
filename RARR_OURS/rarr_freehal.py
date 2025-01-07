from models.atomic_text_generator import AtomicTextGenerator
from models.query_generator import QueryGenerator
from models.DPR_retriever import DPRRetriever
from models.evidence_selector import EvidenceSelector
from models.agreemnet_checker import AgreementChecker
from models.reviser import Reviser
from models.merger import Merger

import os
import transformers
import torch
import pandas as pd
import random

FIXED_SEED = 42
torch.manual_seed(FIXED_SEED)
torch.cuda.manual_seed(FIXED_SEED)
torch.cuda.manual_seed_all(FIXED_SEED)
random.seed(FIXED_SEED)

class RARRFreeHal:
    def __init__(self, args):
        
        self.args = args
        self.input_data = pd.read_csv(args.input_path)
        
        # 모델과 토크나이저를 한 번만 로드
        print("Loading model and tokenizer...")
        model_name = args.model_path
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
        self.pipeline = transformers.pipeline(
            "text-generation",
            model=model_name,
            device_map="auto",
            torch_dtype=torch.bfloat16, 
            pad_token_id=self.tokenizer.eos_token_id 
        )
        transformers.set_seed(FIXED_SEED)
        
        # init all the model
        
        self.atomic_text_generator = AtomicTextGenerator(args, self.pipeline, self.tokenizer)
        self.query_generator = QueryGenerator(args, self.pipeline, self.tokenizer)
        self.DPR_retriever = DPRRetriever(args)
        self.evidenc_selector = EvidenceSelector(args)
        self.agreement_checker = AgreementChecker(args, self.pipeline, self.tokenizer)
        self.editor = Reviser(args, self.pipeline, self.tokenizer)
        self.merger = Merger(args, self.pipeline, self.tokenizer)
        
        
        print("Finish loading models.")
        
    # 데이터 변환 함수
    def transform_dataframe(self, data: pd.DataFrame):
        # input_text 순서 유지
        unique_input_text_order = data['input_text'].unique()
        results = []
        
        for input_text in unique_input_text_order:
            # input_text별 그룹화
            group = data[data['input_text'] == input_text]
            
            # 각 열의 데이터를 리스트로 묶음
            atomic_text_list = group['atomic_text'].tolist()
            query_list = group['query'].tolist()
            selected_evidence_list = group['selected_evidence'].tolist()
            agreement_list = group['agreement'].tolist()
            revised_text_list = group['revised_text'].tolist()
            
            # mid_latency 계산
            group['mid_latency'] = (
                group['query_latency'] +
                group['retrieved_latency'] +
                group['selected_evd_latency'] +
                group['agreement_latency'] +
                group['revision_latency']
            )
            # 최대 mid_latency 값
            max_mid_latency = group['mid_latency'].max()
            
            # total_latency 계산
            total_latency = group['atomic_latency'].iloc[0] + max_mid_latency + group['merge_latency'].iloc[0]
            
            # 병합된 결과 저장
            results.append({
                'total_latency': total_latency,
                'input_text': input_text,
                'atomic_text': atomic_text_list,
                'atomic_query': query_list,
                'selected_evidence': selected_evidence_list,
                'agreement': agreement_list,
                'revised_text': revised_text_list,
                'merged_text': group['merged_text'].iloc[0]
            })
            
        # 결과 데이터프레임 생성
        result_df = pd.DataFrame(results)
        
        # total_latency 평균 계산
        average_total_latency = result_df['total_latency'].mean()
        
        # 파일 경로 생성
        output_dir = "./outputs"
        os.makedirs(output_dir, exist_ok=True) 
        output_file = os.path.join(output_dir, f"{self.args.dataset}_finish.csv")

        # 데이터프레임 저장
        result_df.to_csv(output_file, index=False, encoding='utf-8-sig')
        
        # 평균 total_latency 반환 추가
        return result_df, average_total_latency

        
    
    def correct(self):
        data = self.atomic_text_generator.generate_atomic(self.input_data)
        data = self.query_generator.generate_query(data)        
        data = self.DPR_retriever.search_query(data)
        data = self.evidenc_selector.select_evidence(data)
        data = self.agreement_checker.agreement_check(data)
        data = self.editor.revise_text(data)
        data = self.merger.merge_atomic_text(data)
        
        final_data, total_latency_avg = self.transform_dataframe(data)
        
        return final_data, total_latency_avg