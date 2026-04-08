CLASSIFY_NODE_PROMPT = """You are a knowledge graph expert.

Given a text chunk and a list of candidate nodes, identify the SINGLE node that best represents the chunk's main topic.

Candidate nodes (Node ID | Category | Node Name):
{candidates}

Text chunk:
{chunk_text}

Return ONLY valid JSON — no markdown, no commentary:
{{
  "node_id":   "<Node ID>",
  "node_name": "<Node Name>",
  "category":  "<Category>"
}}"""

CLASSIFY_PROMPT = """You are a document analysis expert.
Examine the image and return ONLY valid JSON — no markdown, no explanation:

{{
  "type": "<table|chart|text_data|other>",
  "reason": "<one sentence in English>"
}}

Classification rules:
- "table"     : grid of rows and columns containing data
- "chart"     : any chart or graph (bar, line, pie, scatter, etc.)
- "text_data" : dense text block worth extracting
- "other"     : decorative image, logo, or illustration with no extractable data"""

EXTRACT_TABLE_PROMPT = """This image contains a data table.

Extract:
1. The full table as a Markdown table.
2. A concise caption (1–2 sentences) describing what the table shows.

Respond in exactly this format, nothing else:
CAPTION: <caption>

<markdown table>"""

EXTRACT_CHART_PROMPT = """This image is a chart or graph.

Provide:
1. A detailed analysis: chart type, axes, data series, trends, and key takeaways.
2. A concise caption (1–2 sentences).

Respond in exactly this format, nothing else:
CAPTION: <caption>

ANALYSIS:
<detailed analysis>"""

EXTRACT_TEXT_PROMPT = """This image contains text.

Extract all visible text, preserving the original paragraph structure.
Return only the extracted text — no additional explanation."""
