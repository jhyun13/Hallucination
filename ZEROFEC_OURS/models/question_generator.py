import os
import torch
import transformers
import numpy as np
import pandas as pd
import transformers
import random
import ast
import tqdm
import time

FIXED_SEED = 42
torch.manual_seed(FIXED_SEED)
torch.cuda.manual_seed(FIXED_SEED)
torch.cuda.manual_seed_all(FIXED_SEED)
random.seed(FIXED_SEED)


def format_inputs(context: str, answer: str):
    # return f"answer:{answer} context:{context}"
    return f"{answer} \\n {context}"
    
class QuestionGenerator:
    def __init__(self, args, model, tokenizer):
        transformers.set_seed(FIXED_SEED)
        
        # self.model = AutoModelForSeq2SeqLM.from_pretrained(qg_path).cuda()
        # self.tokenizer = AutoTokenizer.from_pretrained(qg_path)
        self.args = args
        self.pipeline = model
        self.tokenizer = tokenizer
        
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
        # print(f'outputs:: {outputs}\n')
        
        # "\n\n"에서 텍스트를 잘라내기 -> stop 후처리
        if "\n\n" in outputs:
            outputs = outputs.split("\n\n")[0]
            
        # print(f'\\n\\n 후처리한 output:: {outputs}\n\n')

        return outputs
    

    def generate_batch_questions(self, sample: pd.DataFrame):
        batch_size = 10
        manipulated_sentence = sample['atomic_text']
        sample['generated_question'] = []
        
        for idx in range(0, len(sample['candidate_answers']), batch_size):
            texts = [format_inputs(manipulated_sentence, candidate) for candidate in sample['candidate_answers'][idx:idx+batch_size]]
            
            input_ids = self.tokenizer(texts, return_tensors="pt", padding='longest', truncation=True, max_length=1024).input_ids.cuda()
            generated_ids = self.model.generate(input_ids, max_length=32, num_beams=4)
            output = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            
            sample['generated_question'] += output
            
        return sample
    
    def generate_questions(self, sample: pd.DataFrame):
        atomic_text_list = sample['atomic_text']
        cand_ans_list = sample['candidate_answers']
        question_list = []
        question_latency_list = []
        
        for atomic_text, candidate in tqdm(zip(atomic_text_list, cand_ans_list)):
            start_time = time.time()
            questions = []
            candidate = ast.literal_eval(candidate)
            for cand in candidate:
                # prompt에 few-shot이 필요한가?
                input_prompt = format_inputs(atomic_text, cand)
                generated_question = self.generating(input_prompt)
                questions.append(generated_question)
                
            question_list.append(questions)
            end_time = time.time()
            latency = end_time - start_time
            question_latency_list.append(latency)
        
        sample['question'] = question_list
        sample['question_latency'] = question_latency_list
        
        # 파일 경로 생성
        output_dir = "./outputs"
        os.makedirs(output_dir, exist_ok=True) 
        output_file = os.path.join(output_dir, f"{self.args.dataset}_question_generation.csv")

        # 데이터프레임 저장
        sample.to_csv(output_file, index=False, encoding='utf-8-sig')
        
        return sample