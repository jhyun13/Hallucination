import requests
import json
from typing import List

def clean_json_string(json_string: str) -> str:
    """
    Cleans a JSON string by removing escape characters and unnecessary whitespace.
    """
    return (
        json_string.replace('\"', '"')
        .replace("\"", '"')
        .replace("\\n", " ")
        .replace("\\'", "'")
        .replace('"""', '"')
        .replace('\n', ' ')
        .replace('""""', '"')
        .strip()
    )

def run_search(queries: List[str], search_url: str) -> List[List[str]]:
    """
    Sends queries to the search server and retrieves results.

    Args:
        queries: A list of query strings.
        search_url: URL of the search server.

    Returns:
        A list of lists containing cleaned retrieved documents for each query.
    """
    headers = {"User-Agent": "Test Client"}
    payload = {"query": queries}

    try:
        response = requests.post(search_url, headers=headers, json=payload)
        response.raise_for_status()  # Raise an error for bad HTTP status codes
    except requests.exceptions.RequestException as e:
        print(f"Error in search server request: {e}")
        return [[] for _ in queries]

    try:
        data = json.loads(response.content)
        outputs = data.get("document", [])
        processed_outputs = [
            [clean_json_string(doc) for doc in docs]
            for docs in outputs
        ]
        return processed_outputs
    except (KeyError, json.JSONDecodeError) as e:
        print(f"Error parsing search server response: {e}")
        return [[] for _ in queries]
