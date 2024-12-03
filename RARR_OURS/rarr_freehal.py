from .models.atomic_text_generator import AtomicTextGenerator
from .models.query_generator import QueryGenerator
from .models.DPR_retriever import DPRRetriever
from .models.evidence_selector import EvidenceSelector
from .models.agreemnet_checker import AgreementChecker
from .models.editor import Editor
from .models.merger import Merger

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
        self.editor = Editor(args, self.pipeline, self.tokenizer)
        self.merger = Merger(args, self.pipeline, self.tokenizer)
        
        
        print("Finish loading models.")
        
    
    def correct(self):
        data = self.atomic_text_generator.generate_atomic(self.input_data)
        data = self.query_generator.generate_query(data)
        data = self.DPR_retriever.search_query(data)
        data = self.evidenc_selector.select_evidence(data)
        data = self.agreement_checker.agreement_check(data)
        data = self.editor.revise_text(data)
        data = self.merger.merge(data)
        
        return data