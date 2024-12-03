import sys
import os
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

class EvidenceSelector:
    def __init__(self, args):
        self.args = args
        
        print("[Evidence Selector] Initialized \n")
        
        
    def evidence_selection(self, data: pd.DataFrame):
        print("[Evidence Selector] Selecting evidence from list of retrieved documents...")
        
        print("[EvidenceSelector] Evidence selection complete.")
        return data
