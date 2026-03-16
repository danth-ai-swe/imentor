from tavily import TavilyClient

tavily_client = TavilyClient(api_key="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")
response = tavily_client.search("chiến sự diễn ra ở iran đang như thế nào")
print(response)


def only_en_vi(text):
    import re
    pattern = r'^[a-zA-ZÀ-ỹ\s.,!?;:()\'"-]+$'
    return bool(re.match(pattern, text))


from deep_translator import GoogleTranslator

result = GoogleTranslator(source='auto', target='vi').translate("test")
print(result)

from firecrawl import Firecrawl

app = Firecrawl(api_key="fc-ada04a54f4054027ac8dcd26b043d999")

data = app.scrape(
    "cellphones.com.vn/iphone-17-pro.html",
    only_main_content=False,
    max_age=172800000,
    parsers=["pdf"],
    formats=["markdown"]
)

print(data)

import tiktoken


def openai_token_count(text: str, model: str = "gpt-4o-mini") -> int:
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")

    tokens = encoding.encode(text)
    return len(tokens)
