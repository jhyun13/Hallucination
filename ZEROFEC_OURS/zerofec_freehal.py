from models.atomic_text_generator import AtomicTextGenerator
from models.answer_selector import AnswerSelector
from models.question_generator import QuestionGenerator
from models.question_answer import QuestionAnswerer
from models.candidate_generator import CandidateGenerator
from models.entailment_model import EntailmentModel
from tqdm import tqdm
from typing import List, Dict
import pandas as pd
import transformers
import torch
import random

import nltk
nltk.download('punkt')

FIXED_SEED = 42
torch.manual_seed(FIXED_SEED)
torch.cuda.manual_seed(FIXED_SEED)
torch.cuda.manual_seed_all(FIXED_SEED)
random.seed(FIXED_SEED)

class ZeroFECFreeHal:

    def __init__(self, args) -> None:
        
        # 모델과 토크나이저를 한 번만 로드
        print("Loading model and tokenizer...")
        self.model_name = args.qg_path
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(self.model_name)
        # self.pipeline = transformers.pipeline(
        #     "text-generation",
        #     model=self.model_name,
        #     device_map="auto",
        #     torch_dtype=torch.bfloat16, 
        #     pad_token_id=self.tokenizer.eos_token_id 
        # )
        transformers.set_seed(FIXED_SEED)
        
        # init all the model
        
        self.atomic_text_generator = AtomicTextGenerator(args, self.model_name, self.tokenizer)
        self.answer_selector = AnswerSelector(args)    
        self.question_generator = QuestionGenerator(args, self.model_name, self.tokenizer)
        self.question_answerer = QuestionAnswerer(args,self.model_name, self.tokenizer)
        self.candidate_generator = CandidateGenerator(args)
        self.entailment_model = EntailmentModel(args)
        
        
        print("Finish loading models.")

    
    def correct(self, data: pd.DataFrame):
        '''
        data is Dict containing at least two fields:
            inputs: str, the claim to be corrected.
            evidence: str, the list of reference article to check against.
        '''
        print(f"data :: {data}\n")

        data = self.atomic_text_generator.generate_atomic(data)
        data = self.answer_selector.select_answers(data)
        data = self.question_generator.generate_questions(data)
        data = self.question_answerer.generate_answers(data)
        data = self.candidate_generator.generate_candidate(data)
        data = self.entailment_model.run_entailment_prediction(data)

        
        return data

    def batch_correct(self, datas: List[pd.DataFrame]):

        return [self.correct(data) for data in tqdm(datas, total=len(datas))]
        