"""Utils for running the editor."""
import time
from typing import Dict, Union


def parse_api_response(api_response: str) -> str:
    """Extract the agreement gate state and the reasoning from the GPT-3 API response.

    Our prompt returns a reason for the edit and the edit in two consecutive lines.
    Only extract out the edit from the second line.

    Args:
        api_response: Editor response from GPT-3.
    Returns:
        edited_claim: The edited claim.
    """
    api_response = api_response.strip().split("\n")
    if len(api_response) < 2:
        print("Editor error.")
        return None
    edited_claim = api_response[1].split("My fix:")[-1].strip()
    return edited_claim


def run_rarr_editor(
    claim: str,
    model,
    tokenizer,
    query: str,
    evidence: str,
    prompt: str,
    num_retries: int = 5,
    device: str = "cuda",
) -> Dict[str, str]:
    """Runs a GPT-3 editor on the claim given a query and evidence to support the edit.

    Args:
        claim: Text to edit.
        query: Query to guide the editing.
        evidence: Evidence to base the edit on.
        model: Name of the OpenAI GPT-3 model to use.
        prompt: The prompt template to query GPT-3 with.
        num_retries: Number of times to retry OpenAI call in the event of an API failure.
    Returns:
        edited_claim: The edited claim.
    """
    input_prompt = prompt.format(claim=claim, query=query, evidence=evidence).strip()
    len_input = len(input_prompt)

    for attempt in range(num_retries):
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
            print(f"[Editor] generated_text :: {generated_text}\n")
            
            if "\n\n" in generated_text:    
                generated_text = generated_text.split("\n\n")[0]
                
            break
        except Exception as e:
            print(f"Error [Editor]: {e}. Retrying... ({attempt + 1}/{num_retries})")
            time.sleep(2)

    edited_claim = parse_api_response(response_text)
    # If there was an error in GPT-3 generation, return the claim.
    if not edited_claim:
        edited_claim = claim
    output = {"text": edited_claim}
    
    return output