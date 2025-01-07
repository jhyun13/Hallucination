"""Utils for running question generation."""
import time
from typing import List


def parse_api_response(api_response: str) -> List[str]:
    """Extract questions from the GPT-3 API response.

    Our prompt returns questions as a string with the format of an ordered list.
    This function parses this response in a list of questions.

    Args:
        api_response: Question generation response from GPT-3.
    Returns:
        questions: A list of questions.
    """
    search_string = "I googled:"
    questions = []
    for question in api_response.split("\n"):
        # Remove the search string from each question
        if search_string not in question:
            continue
        question = question.split(search_string)[1].strip()
        questions.append(question)

    return questions


def run_rarr_question_generation(
    claim: str,
    model,
    tokenizer,
    prompt: str,
    num_rounds: int,
    num_retries: int = 5,
    device: str = "cuda"
) -> List[str]:
    """Generates questions that interrogate the information in a claim.

    Given a piece of text (claim), we use GPT-3 to generate questions that question the
    information in the claim. We run num_rounds of sampling to get a diverse set of questions.

    Args:
        claim: Text to generate questions off of.
        model: Name of the OpenAI GPT-3 model to use.
        prompt: The prompt template to query GPT-3 with.
        temperature: Temperature to use for sampling questions. 0 represents greedy deconding.
        num_rounds: Number of times to sample questions.
    Returns:
        questions: A list of questions.
    """
    input_prompt = prompt.format(claim=claim).strip()
    len_input = len(input_prompt)

    questions = set()
    for _ in range(num_rounds):
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
                print(f"[Question Generation] generated_text \n")
                
                if "\n\n" in generated_text:    
                    generated_text = generated_text.split("\n\n")[0]
                    extracted_questions = parse_api_response(generated_text)
                    print(f"generated_text :: {extracted_questions}")
                questions.update(extracted_questions)
                break
            except Exception as e:
                print(f"{e}. Retrying...")
                time.sleep(1)

    questions = list(sorted(questions))
    return questions