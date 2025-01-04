import requests
import json
from typing import List, Dict, Any

def clean_json_string(json_string: str) -> str:
    """
    Cleans a JSON string by removing escape characters and unnecessary whitespace.
    """
    return (
        json_string.replace('"', '"')
        .replace("\"", '"')
        .replace("\\n", " ")
        .replace("\\'", "'")
        .replace('"""', '"')
        .replace('\n', ' ')
        .replace('""""', '"')
        .strip()
    )

def chunk_text(
    text: str,
    sentences_per_passage: int,
    filter_sentence_len: int,
    sliding_distance: int = None,
) -> List[str]:
    """
    Chunks text into passages using a sliding window.

    Args:
        text: Text to chunk into passages.
        sentences_per_passage: Number of sentences for each passage.
        filter_sentence_len: Maximum number of chars of each sentence before being filtered.
        sliding_distance: Sliding distance over the text. Allows the passages to have
            overlap. The sliding distance cannot be greater than the window size.
    Returns:
        passages: Chunked passages from the text.
    """
    if not sliding_distance or sliding_distance > sentences_per_passage:
        sliding_distance = sentences_per_passage
    assert sentences_per_passage > 0 and sliding_distance > 0

    passages = []
    try:
        import spacy
        nlp = spacy.load("en_core_web_sm", disable=["ner", "tagger", "lemmatizer"])
        doc = nlp(text[:500000])  # Take 500k chars to not break tokenization.
        sents = [
            s.text
            for s in doc.sents
            if len(s.text) <= filter_sentence_len  # Long sents are usually metadata.
        ]
        for idx in range(0, len(sents), sliding_distance):
            passages.append(" ".join(sents[idx : idx + sentences_per_passage]))
    except UnicodeEncodeError as _:  # Sometimes run into Unicode error when tokenizing.
        print("Unicode error when using Spacy. Skipping text.")

    return passages

def score_and_sort_passages(passages: List[str], query: str) -> List[Dict[str, Any]]:
    """
    Scores passages by relevance to the query and sorts them by score.

    Args:
        passages: A list of text passages.
        query: The query string.

    Returns:
        A list of dictionaries containing passages and their scores, sorted by score.
    """
    from sentence_transformers import CrossEncoder

    # Load the cross-encoder for scoring
    model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", device="cuda")

    # Score each passage
    scores = model.predict([(query, passage) for passage in passages]).tolist()

    # Combine passages and scores, then sort by score in descending order
    scored_passages = [
        {"text": passage, "score": score}
        for passage, score in zip(passages, scores)
    ]
    scored_passages.sort(key=lambda x: x["score"], reverse=True)

    return scored_passages

def run_search(
    query: List[str], 
    search_url: str,
    sentences_per_passage: int = 5, 
    filter_sentence_len: int = 250, 
    sliding_distance: int = 1,
) -> List[List[Dict[str, Any]]]:
    """
    Sends queries to the search server, retrieves results, chunks the texts into passages,
    scores the passages, and sorts them by relevance.

    Args:
        queries: A list of query strings.
        search_url: URL of the search server.
        sentences_per_passage: Number of sentences per passage.
        filter_sentence_len: Maximum number of characters per sentence.
        sliding_distance: Sliding distance for overlapping passages.

    Returns:
        A list of lists where each sublist contains scored and sorted passages for the corresponding query.
    """
    headers = {"User-Agent": "Test Client"}
    payload = {"query": [query]}

    try:
        response = requests.post(search_url, headers=headers, json=payload)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        print(f"Error in search server request: {e}")
        return [[] for _ in query]

    try:
        data = json.loads(response.content)
        outputs = data.get("document", [])
        processed_outputs = [clean_json_string(doc) for docs in outputs for doc in docs]

        # final_results = []
        for qry, text in zip(query, processed_outputs):
            passages = chunk_text(
                text=text,
                sentences_per_passage=sentences_per_passage,
                filter_sentence_len=filter_sentence_len,
                sliding_distance=sliding_distance,
            )
            scored_passages = score_and_sort_passages(passages, qry)
        #     final_results.append(scored_passages)

        # return final_results
            if scored_passages:
                # Take the highest scoring passage's text
                highest_scoring_text = scored_passages[0]["text"]
            else:
                highest_scoring_text = ""

        return highest_scoring_text
    except (KeyError, json.JSONDecodeError) as e:
        print(f"Error parsing search server response: {e}")
        return [[] for _ in query]
