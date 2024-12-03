import os
import pandas as pd
import torch
import random
from tqdm import tqdm
from typing import Any, Dict, List, Tuple
import requests
import json
import time

# fixed seed
FIXED_SEED = 42
torch.manual_seed(FIXED_SEED)
torch.cuda.manual_seed(FIXED_SEED)
torch.cuda.manual_seed_all(FIXED_SEED)
random.seed(FIXED_SEED)

class DPRRetriever:
    def __init__(self, args):
        self.args = args 
        self.search_url = args.search_url    
        self.batch_size = 1000
        #self.batch_size = args.batch_size 
        
        print("[DPR Retriever] Initialized with search url.\n")
        
    # 문자열 정리 함수
    @staticmethod
    def clean_json_string(json_string: str) -> str:
        """
        JSON 문자열에서 특수 문자를 제거하고 깨끗한 텍스트를 반환합니다.
        """
        return (
            json_string.replace('\\"', '"')  # \" -> "
            .replace("\"", '"')             # \" -> "
            .replace("\\n", " ")            # \n -> 공백
            .replace("\\'", "'")            # \' -> '
            .replace('"""', '"')            # """ -> "
            .replace('\n', ' ')            # "\n" -> 공백
            .replace('""""', '"')            # "\n" -> 공백
            .strip()                        # 양쪽 공백 제거
        )
        
    # Search Function
    def search(self, query: List[str]):
        headers = {"User-Agent": "Test Client"}
        pload = {
            "query": query,
        }
        response = requests.post(self.search_url, headers=headers, json=pload)
        data = json.loads(response.content)
        outputs = data["document"]
        
        processed_outputs = [
            [self.clean_json_string(doc) for doc in docs] for docs in outputs
        ]

        return processed_outputs


    def __call__(self, data: pd.DataFrame):
        print(f"[DPR Retriever] Retrieving documents for query ...")

        query_list = data['query']
        retrieved_doc_list = []
        retreived_latency_list = []
        
        for qry in tqdm(query_list):
            start_time = time.time()
            
            documents = self.search(qry)
            retrieved_doc_list.append(documents)
            
            end_time = time.time()
            latency = end_time - start_time()
            retreived_latency_list.append(latency)
        
        data['retrieved_evidence'] = retrieved_doc_list
        data['retrieved_latency'] = retreived_latency_list
        
        # 파일 경로 생성
        output_dir = "./outputs"
        os.makedirs(output_dir, exist_ok=True) 
        output_file = os.path.join(output_dir, f"{self.args.dataset}_retrieved.csv")

        # 데이터프레임 저장
        data.to_csv(output_file, index=False, encoding='utf-8-sig')
    
        print("[DPR Retriever] Document retrieval complete.\n")

        return data