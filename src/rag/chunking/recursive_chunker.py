from functools import lru_cache

import tiktoken
from langchain_text_splitters import RecursiveCharacterTextSplitter

@lru_cache(maxsize=1)
def get_recursive_token_chunk(chunk_size=256, model_name="text-embedding-3-small"):
    MARKDOWN_SEPARATORS = [
        "\n#{1,6} ",
        "```\n",
        "\n\\*\\*\\*+\n",
        "\n---+\n",
        "\n___+\n",
        "\n\n",
        "\n",
        " ",
        "",
    ]

    encoding = tiktoken.encoding_for_model(model_name)

    def openai_token_len(text: str) -> int:
        return len(encoding.encode(text))

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=int(chunk_size / 10),
        length_function=openai_token_len,
        add_start_index=True,
        strip_whitespace=True,
        separators=MARKDOWN_SEPARATORS,
    )

    return text_splitter
