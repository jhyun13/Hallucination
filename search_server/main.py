import os
# Pyserini 캐시 경로를 수동으로 지정
os.environ["PYSERINI_CACHE"] = "/home/work/.cache/pyserini"
import asyncio
from typing import Optional, List
from fastapi import FastAPI
from pydantic import BaseModel
import torch
import numpy as np
import random
import json

# import threading
from pyserini.search.faiss import FaissSearcher, DprQueryEncoder
from pyserini.search.lucene import LuceneSearcher
# from pyserini.encode import DprQueryEncoder

app = FastAPI()

FIXED_SEED = 42

def seed_everything(seed: int = FIXED_SEED):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)  # type: ignore
    torch.backends.cudnn.deterministic = True  # type: ignore
    torch.backends.cudnn.benchmark = True  # type: ignore

seed_everything()

# API 요청으로 들어오는 데이터를 검증하고 구조화
class Item(BaseModel):
    query: Optional[List[str]] = None
    documents: Optional[List[str]] = None


class SearchClass:
    def __init__(self):
        self.encoder = DprQueryEncoder(
            "facebook/dpr-question_encoder-multiset-base", device="cuda:2"
        )
        
        # # 다운로드 받고, 압축해제 한 후에
        # self.searcher = FaissSearcher.from_prebuilt_index(
        #     "wikipedia-dpr-100w.dpr-multi", self.encoder
        # )
        
        # 이 코드 이용해서 직접 로드하기
        # Prebuilt index 대신 로컬 경로로 직접 초기화
        index_path = "/home/work/.cache/pyserini/indexes/faiss.wikipedia-dpr-100w.dpr_multi.20200127.f403c3.fe307ef2e60ab6e6f3ad66e24a4144ae/"
        self.dense_searcher = FaissSearcher(index_path, self.encoder)
        
        # Sparse Index 초기화
        self.sparse_searcher = LuceneSearcher.from_prebuilt_index("wikipedia-dpr-100w")
        
    def get_content_from_sparse_index(self, docid: str) -> str:
        document = self.sparse_searcher.doc(docid)        
        if document:
            print(f"Document Found for docid {docid}: {document.raw()}")
            try:
                parsed_document = json.loads(document.raw())
                content = parsed_document.get("contents", "Content field not found")
                # 맨 앞과 맨 끝의 " 제거
                if content.startswith('"') and content.endswith('"'):
                    content = content[1:-1]
                
                return content

            except json.JSONDecodeError:
                return "Invalid JSON format in document"
            
        print(f"No document found for docid {docid}")
        return "Content not found"   
        

    # 쿼리를 str로 받아야 하는지, list로 받아야 하는지
    # 일단은 str.
    async def __call__(self, query: List[str]):
        # Dense Index 검색
        print(f"Query received by batch_search: {query}")
        q_embs = np.array([self.query_encoder.encode(q) for q in query])  # 벡터화
        print(f"Encoded query embeddings shape: {q_embs.shape}")
        # q_ids = [str(i) for i in range(len(query))]
        # dense_results = self.dense_searcher.search(query=query, k=5, threads=10)
        # dense_results = self.dense_searcher.batch_search(query, q_ids=q_ids, k=5, threads=10)
        dense_results = self.dense_searcher.batch_search(q_embs, q_ids=[str(i) for i in range(len(query))], k=5, threads=10)

        print(f"Dense Search Results: {dense_results}")

        # 결과를 Sparse Index에서 처리 
        results_with_content = []
        for qid, documents in dense_results.items():
            query_contents = []  # 각 쿼리에 대한 콘텐츠만 저장
            print(f"Documents for Query ID {qid}: {documents}")
            for document in documents:
                docid = document.docid
                content = self.get_content_from_sparse_index(docid)  # Sparse Index에서 내용 조회
                query_contents.append(content)  # 콘텐츠만 저장
        #     results_with_content.append(query_contents)  # 최종 결과에 추가
            
        # return results_with_content
        
        # query_contents = []  # 각 쿼리에 대한 콘텐츠만 저장
        # print(f"Documents for Query: {dense_results}")
        # for document in dense_results:
        #     docid = document.docid
        #     content = self.get_content_from_sparse_index(docid)  # Sparse Index에서 내용 조회
        #     query_contents.append(content)  # 콘텐츠만 저장
            
        return query_contents


search = SearchClass()
# multi-processing 적용해서 CPU 코어 수 만큼 스레드 생성하기


async def process_search(query: str):
    return await search(query)


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.post("/")
async def read_item(data_request: Item):
    # Start the search process in the background
    search_task = asyncio.create_task(process_search(data_request.query))
    # Do other things while the search is in progress
    result = {
        "Hello": "POST",
        "query": data_request.query,
        "status": "search in progress",
    }

    # Wait for the search to complete and update the result
    result["document"] = await search_task
    result["status"] = "search completed"
    return result



###### uvicorn main:app --reload --host 127.0.0.1 --port 8000