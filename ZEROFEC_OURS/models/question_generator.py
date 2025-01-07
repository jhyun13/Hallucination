import torch
import transformers
import pandas as pd
import transformers
import random
import time

FIXED_SEED = 42
torch.manual_seed(FIXED_SEED)
torch.cuda.manual_seed(FIXED_SEED)
torch.cuda.manual_seed_all(FIXED_SEED)
random.seed(FIXED_SEED)

## prompt 변경 필요 !!
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
        
        # "\n\n"에서 텍스트를 잘라내기 -> stop 후처리
        if "\n\n" in outputs:
            outputs = outputs.split("\n\n")[0]
            
        return outputs
    
    
    def generate_questions(self, sample: pd.DataFrame):
        batch_size = 10
        manipulated_sentences = sample['atomic_text']
        candidate_answers = sample['candidate_answers']

        # Initialize the generated_question column
        sample['generated_question'] = [[] for _ in range(len(manipulated_sentences))]

        # Initialize a latency column
        sample['latency'] = [0.0 for _ in range(len(manipulated_sentences))]

        # Process each atomic_text with its corresponding candidate_answers
        for i, (manipulated_sentence, answers) in enumerate(zip(manipulated_sentences, candidate_answers)):
            
            start_time = time.time() 

            for idx in range(0, len(answers), batch_size):
                texts = [format_inputs(manipulated_sentence, candidate) for candidate in answers[idx:idx+batch_size]]

                input_ids = self.tokenizer(texts, return_tensors="pt", padding='longest', truncation=True, max_length=1024).input_ids.cuda()
                generated_ids = self.model.generate(input_ids, max_length=32, num_beams=4)
                output = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

                # Append generated questions to the corresponding list
                sample['generated_question'][i] += output

            end_time = time.time()
            
            # Calculate latency for this row
            sample['question_generation_latency'][i] = end_time - start_time

        return sample