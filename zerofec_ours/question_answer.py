import os
import random
import torch
import transformers
import pandas as pd
from typing import Dict
from nltk import word_tokenize

FIXED_SEED = 42
torch.manual_seed(FIXED_SEED)
torch.cuda.manual_seed(FIXED_SEED)
torch.cuda.manual_seed_all(FIXED_SEED)
random.seed(FIXED_SEED)


def format_inputs_qa(context: str, question: str):
    # return f"extract answers: <hl> Beyonce further expanded her acting career, starring as blues singer Etta James in the 2008 musical biopic, Cadillac Records. <hl> Her performance in the film received praise from critics, and she garnered several nominations for her portrayal of James, including a Satellite Award nomination for Best Supporting Actress, and a NAACP Image Award nomination for Outstanding Supporting Actress."
    return f"{question} \n {context}"

class QuestionAnswerer:
    def __init__(self, args, model, tokenizer):
        transformers.set_seed(FIXED_SEED)
        self.args = args        
        self.model = model
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

    
    def generate_answers(self, sample: pd.DataFrame):
        generated_questions = sample['generated_question']
        answer_list = []
        answer_latency_list = []
        
        for question in generated_questions:
            
            
            
            question_answers = []
            for _, ctx in enumerate(sample['evidence']):

                words = word_tokenize(ctx)
                passage_size = 400
                for i in range(0, len(words), passage_size):
                    context = ' '.join(words[i:i+passage_size])
                    
                    input_ids = self.tokenizer.encode(f"{question} \n {context}", return_tensors='pt').cuda()
                    
                    with torch.no_grad():
                        
                        outputs = self.model.generate(input_ids, num_beams=4, do_sample=False)
                        predict_answer_tokens_string = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]        
                        question_answers.append(predict_answer_tokens_string.strip())
                
            sample['answer'].append(question_answers)
        
        
        return sample