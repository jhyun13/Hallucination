import pandas as pd
import torch
import random
from tqdm import tqdm
from typing import List

# fixed seed
FIXED_SEED = 42
torch.manual_seed(FIXED_SEED)
torch.cuda.manual_seed(FIXED_SEED)
torch.cuda.manual_seed_all(FIXED_SEED)
random.seed(FIXED_SEED)    

def parse_api_response(api_respnse: str) -> str:
    search_string = "- My merge: "
    if search_string in api_respnse:
        outputs = api_respnse.split("\n- My merge: ")[-1].strip()
    else:
        outputs = api_respnse.strip()
    
    return outputs

def run_merge(
    claim: str,
    revised_text: List[str],
    model,
    tokenizer,
    prompt: str,
    device: str = "cuda"
) -> str:
    
    revised_text_combined = " / ".join(revised_text)
    
    input_prompt = prompt.format(claim, revised_text_combined)
    len_input = len(input_prompt)
    
    try:
        inputs = tokenizer(input_prompt, return_tensors="pt").to(device)
        outputs = model.generate(
            **inputs,
            max_new_tokens=500,
            temperature=0.0,
            do_sample=False,
            top_p=1.0,
            pad_token_id=tokenizer.eos_token_id,
        )
        response_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        generated_text = response_text[len_input:].strip()
        print(f"[Merge] generated_text \n")
        
        if "\n\n" in generated_text:
            generated_text = generated_text.split("\n\n")[0]
            merge_text = parse_api_response(generated_text)
    except Exception as e:
        print(f"{e} - [Merge]")
    
    return merge_text