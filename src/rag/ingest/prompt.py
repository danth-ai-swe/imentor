
CLASSIFIER_PROMPT_TEMPLATE = """You are an expert classifier.

Choose EXACTLY ONE item from the candidate list below.
Each item is a valid (node_name, category) pair from the knowledge node file.

Candidates:
{candidates}

Chunk text:
\"\"\"
{chunk_text}
\"\"\"

Return JSON ONLY in this format:
{{"category": "...", "node_name": "..."}}

Rules:
- category must be one of the candidate categories
- node_name must be one of the candidate node_name values
- the chosen (category, node_name) pair must exist in the candidates
- no explanation text, no markdown"""