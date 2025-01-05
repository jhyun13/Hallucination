from .models.answer_selector import AnswerSelector
from .models.question_generator import QuestionGenerator
from .models.question_answerer import QuestionAnswerer
from .models.candidate_generator import CandidateGenerator
from .models.entailment_model import EntailmentModel
from tqdm import tqdm
from typing import List, Dict
import pandas as pd

import nltk
nltk.download('punkt')

class ZeroFEC:

    def __init__(self, args) -> None:
        # init all the model
        
        
        self.answer_selector = AnswerSelector(args)    
        self.question_generator = QuestionGenerator(args)
        self.question_answerer = QuestionAnswerer(args)
        self.candidate_generator = CandidateGenerator(args)
        self.entailment_model = EntailmentModel(args)
        
        
        print("Finish loading models.")

    
    def correct(self, data: pd.DataFrame):
        '''
        data is Dict containing at least two fields:
            inputs: str, the claim to be corrected.
            evidence: str, the list of reference article to check against.   
        '''

        data = self.answer_selector.select_answers(data)
        data = self.question_generator.generate_questions(data)
        data = self.question_answerer.generate_answers(data)
        data = self.candidate_generator.generate_candidate(data)
        data = self.entailment_model.run_entailment_prediction(data)

        
        return data

    def batch_correct(self, datas: List[pd.DataFrame]):

        return [self.correct(data) for data in tqdm(datas, total=len(datas))]
        