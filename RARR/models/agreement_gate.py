"""Utils for running the agreement gate."""
import os
import time
from typing import Any, Dict, Tuple


def parse_api_response(api_response: str) -> Tuple[bool, str, str]:
    """Extract the agreement gate state and the reasoning from the GPT-3 API response.

    Our prompt returns questions as a string with the format of an ordered list.
    This function parses this response in a list of questions.

    Args:
        api_response: Agreement gate response from GPT-3.
    Returns:
        is_open: Whether the agreement gate is open.
        reason: The reasoning for why the agreement gate is open or closed.
        decision: The decision of the status of the gate in string form.
    """
    api_response = api_response.strip().split("\n")
    if len(api_response) < 2:
        reason = "Failed to parse."
        decision = None
        is_open = False
    else:
        reason = api_response[0]
        decision = api_response[1].split("Therefore:")[-1].strip()
        is_open = "disagrees" in api_response[1]
    return is_open, reason, decision


def run_agreement_gate(
    claim: str,
    model,
    tokenizer,
    query: str,
    evidence: str,
    prompt: str,
    num_retries: int = 5,
    device: str = "cuda",
) -> Dict[str, Any]:
    """Checks if a provided evidence contradicts the claim given a query.

    Checks if the answer to a query using the claim contradicts the answer using the
    evidence. If so, we open the agreement gate, which means that we allow the editor
    to edit the claim. Otherwise the agreement gate is closed.

    Args:
        claim: Text to check the validity of.
        query: Query to guide the validity check.
        evidence: Evidence to judge the validity of the claim against.
        model: Name of the OpenAI GPT-3 model to use.
        prompt: The prompt template to query GPT-3 with.
        num_retries: Number of times to retry OpenAI call in the event of an API failure.
    Returns:
        gate: A dictionary with the status of the gate and reasoning for decision.
    """
    input_prompt = prompt.format(claim=claim, query=query, evidence=evidence).strip()
    len_input = len(input_prompt)

    for _ in range(num_retries):
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
            
            if "\n\n" in generated_text:    
                generated_text = generated_text.split("\n\n")[0]
                print(f"[Agreement Gate] generated_text :: {generated_text}\n")
                
            break
        except Exception as e:
            print(f"{e}. Retrying...")
            time.sleep(2)

    is_open, reason, decision = parse_api_response(generated_text)
    
    print(f"is_open : {is_open}, reason : {reason}, decision : {decision}\n\n")
    gate = {"is_open": is_open, "reason": reason, "decision": decision}
    return gate