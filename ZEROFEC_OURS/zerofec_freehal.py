from .models.atomic_text_generator import AtomicTextGenerator
from .models.answer_selector import AnswerSelector
from .models.question_generator import QuestionGenerator
from .models.question_answerer import QuestionAnswerer
from .models.candidate_generator import CandidateGenerator
from .models.entailment_model import EntailmentModel
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

class ZeroFEC:

    def __init__(self, args) -> None:
        
        self.input_data = pd.read_csv(args.input_path)
        
        # 모델과 토크나이저를 한 번만 로드
        print("Loading model and tokenizer...")
        model_name = args.qg_path
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
        
        self.atomic_text_generator = AtomicTextGenerator(args)
        self.answer_selector = AnswerSelector(args)    
        self.question_generator = QuestionGenerator(args, self.model, self.tokenizer)
        self.question_answerer = QuestionAnswerer(args)
        self.candidate_generator = CandidateGenerator(args)
        self.entailment_model = EntailmentModel(args)
        
        
        print("Finish loading models.")

    
    def correct(self, sample: Dict):
        '''
        sample is Dict containing at least two fields:
            inputs: str, the claim to be corrected.
            evidence: str, the list of reference article to check against.
        '''

        sample = self.atomic_text_generator.generate_atomic(self.input_data)
        sample = self.answer_selector.select_answers(sample)
        sample = self.question_generator.generate_questions(sample)
        sample = self.question_answerer.generate_answers(sample)
        sample = self.candidate_generator.generate_candidate(sample)
        sample = self.entailment_model.run_entailment_prediction(sample)

        
        return sample

    def batch_correct(self, samples: List[Dict]):

        return [self.correct(sample) for sample in tqdm(samples, total=len(samples))]
        