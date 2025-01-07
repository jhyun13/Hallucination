import os
import pandas as pd
import torch
import random
from tqdm import tqdm
import time
from sentence_transformers import CrossEncoder
import itertools
from typing import List

# fixed seed
FIXED_SEED = 42
torch.manual_seed(FIXED_SEED)
torch.cuda.manual_seed(FIXED_SEED)
torch.cuda.manual_seed_all(FIXED_SEED)
random.seed(FIXED_SEED)

class EvidenceSelector:
    def __init__(self, args):
        self.args = args
        self.PASSAGE_RANKER = CrossEncoder(
            "cross-encoder/ms-marco-MiniLM-L-6-v2",
            max_length=512,
            device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        )
        
        print("[Evidence Selector] Initialized \n")
    

    def compute_score_matrix(self, 
        questions: List[str], evidences: List[str]
    ) -> List[List[float]]:
        
        score_matrix = []
        for q in questions:
            evidence_scores = self.PASSAGE_RANKER.predict([(q, e) for e in evidences]).tolist()
            score_matrix.append(evidence_scores)
            
        return score_matrix


    def question_coverage_objective_fn(self, 
        score_matrix: List[List[float]], evidence_indices: List[int]
    ) -> float:
        total = 0.0
        for scores_for_question in score_matrix:
            total += max(scores_for_question[j] for j in evidence_indices)
            
        return total        
    
    
    def select_evidence(self, data: pd.DataFrame):
        print("[Evidence Selector] Selecting evidence from list of retrieved documents...")
        
        selected_evidences = []
        evd_latency_list = []
        max_selected = 6
        prefer_fewer = False

        for query, list_evidences in tqdm(zip(data['query'], data['retrieved_evidence'])):
            start_time = time.time()
            
            docs = [evid.replace("\n", "") for evid in list_evidences if isinstance(evid, str)]


            # 중복 제거 및 정렬
            questions = sorted(set([query]))  # query는 문자열이므로 리스트로 감쌈
            evidences = sorted(set(docs))

            num_evidences = len(evidences)

            # 점수 행렬 계산
            score_matrix = self.compute_score_matrix(questions, evidences)

            best_combo = tuple()
            best_objective_value = float("-inf")
            max_selected = min(max_selected, num_evidences)
            min_selected = 1 if prefer_fewer else max_selected

            # 최고 점수를 가진 조합을 찾음
            for num_selected in range(min_selected, max_selected + 1):
                for combo in itertools.combinations(range(num_evidences), num_selected):
                    objective_value = self.question_coverage_objective_fn(score_matrix, combo)
                    if objective_value > best_objective_value:
                        best_combo = combo
                        best_objective_value = objective_value

            # 최고 점수를 가진 증거를 저장
            selected_evidences.append([{"text": evidences[idx]} for idx in best_combo][0]['text'])
            
            end_time = time.time()
            latency = end_time - start_time
            evd_latency_list.append(latency) 
            
        data['selected_evidence'] = selected_evidences
        data['selected_evd_latency'] = evd_latency_list
        
        # 파일 경로 생성
        output_dir = "./outputs"
        os.makedirs(output_dir, exist_ok=True) 
        output_file = os.path.join(output_dir, f"{self.args.dataset}_selected_evidence.csv")

        # 데이터프레임 저장
        data.to_csv(output_file, index=False, encoding='utf-8-sig')
        
        print("[EvidenceSelector] Evidence selection complete.")

        return data