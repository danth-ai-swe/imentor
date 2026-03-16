import re
from functools import lru_cache

from tqdm import tqdm

from src.utils.utils import openai_token_count
from .recursive_chunker import get_recursive_token_chunk
from ..llm.chat_llm import get_openai_chat_client


class AgenticChunker:

    def __init__(self):
        self.splitter = get_recursive_token_chunk(chunk_size=50)
        self.client = get_openai_chat_client()

    @staticmethod
    def get_prompt(chunked_input, current_chunk=0, invalid_response=None):

        messages = [
            {
                "role": "system",
                "content": (
                    "You are an assistant specialized in splitting text into thematically consistent sections. "
                    "The text has been divided into chunks, each marked with <|start_chunk_X|> and <|end_chunk_X|> tags. "
                    "Your task is to identify where topic splits should occur. "
                    "Respond in the form: split_after: 3,5"
                )
            },
            {
                "role": "user",
                "content": (
                        "CHUNKED_TEXT: " + chunked_input +
                        "\n\nRespond only with split positions."
                        f"\nSplits must be >= {current_chunk}."
                        + (
                            f"\nPrevious response {invalid_response} was invalid. Try again."
                            if invalid_response else ""
                        )
                )
            }
        ]

        return messages

    def split_text(self, text):

        chunks = self.splitter.split_text(text)

        split_indices = []
        current_chunk = 0

        with tqdm(total=len(chunks), desc="Processing chunks") as pbar:

            while True:

                if current_chunk >= len(chunks) - 4:
                    break

                token_count = 0
                chunked_input = ""

                for i in range(current_chunk, len(chunks)):

                    token_count += openai_token_count(chunks[i])

                    chunked_input += (
                        f"<|start_chunk_{i + 1}|>"
                        f"{chunks[i]}"
                        f"<|end_chunk_{i + 1}|>"
                    )

                    if token_count > 800:
                        break

                messages = self.get_prompt(chunked_input, current_chunk)

                while True:

                    result_string = self.client.create_agentic_chunker_message(
                        system_prompt=messages[0]["content"],
                        messages=messages[1:]
                    )

                    split_after_lines = [
                        line for line in result_string.split("\n")
                        if "split_after:" in line
                    ]

                    if not split_after_lines:
                        invalid_response = []
                        messages = self.get_prompt(
                            chunked_input,
                            current_chunk,
                            invalid_response
                        )
                        continue

                    numbers = re.findall(r"\d+", split_after_lines[0])
                    numbers = list(map(int, numbers))

                    is_ascending = numbers == sorted(numbers)
                    all_valid = all(n >= current_chunk for n in numbers)

                    if numbers and is_ascending and all_valid:
                        break
                    else:

                        invalid_response = numbers
                        messages = self.get_prompt(
                            chunked_input,
                            current_chunk,
                            invalid_response
                        )

                split_indices.extend(numbers)
                current_chunk = numbers[-1]

                if len(numbers) == 0:
                    break

                pbar.update(current_chunk - pbar.n)

        chunks_to_split_after = [i - 1 for i in split_indices]

        docs = []
        current_chunk_text = ""

        for i, chunk in enumerate(chunks):

            current_chunk_text += chunk + " "

            if i in chunks_to_split_after:
                docs.append(current_chunk_text.strip())
                current_chunk_text = ""

        if current_chunk_text:
            docs.append(current_chunk_text.strip())

        return docs


@lru_cache(maxsize=1)
def get_agentic_chunker() -> AgenticChunker:
    return AgenticChunker()
