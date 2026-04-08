import re
from functools import lru_cache

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from src.rag.llm.embedding_llm import get_openai_embedding_client


class SemanticChunker:

    def __init__(self, buffer_size=1, percentile_threshold=95):
        self.buffer_size = buffer_size
        self.percentile_threshold = percentile_threshold
        self.embedder = get_openai_embedding_client()

    @staticmethod
    def split_sentences(text):
        sentences = re.split(r'(?<=[.?!])\s+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        return sentences

    @staticmethod
    def create_sentence_dict(sentences):
        return [{'sentence': s, 'index': i} for i, s in enumerate(sentences)]

    def combine_sentences(self, sentences):

        for i in range(len(sentences)):

            combined_sentence = ''

            for j in range(i - self.buffer_size, i):
                if j >= 0:
                    combined_sentence += sentences[j]['sentence'] + ' '

            combined_sentence += sentences[i]['sentence']

            for j in range(i + 1, i + 1 + self.buffer_size):
                if j < len(sentences):
                    combined_sentence += ' ' + sentences[j]['sentence']

            sentences[i]['combined_sentence'] = combined_sentence

        return sentences

    def embed_sentences(self, sentences):

        embeddings = self.embedder.embed_documents(
            [x['combined_sentence'] for x in sentences]
        )

        for i, sentence in enumerate(sentences):
            sentence['embedding'] = embeddings[i]

        return sentences

    @staticmethod
    def calculate_distances(sentences):

        distances = []

        for i in range(len(sentences) - 1):
            current = sentences[i]['embedding']
            next_emb = sentences[i + 1]['embedding']

            similarity = cosine_similarity(current, next_emb)

            distance = 1 - similarity

            sentences[i]['distance_to_next'] = distance
            distances.append(distance)

        return distances, sentences

    def create_chunks(self, sentences, distances):

        threshold = np.percentile(distances, self.percentile_threshold)

        breakpoints = [
            i for i, d in enumerate(distances)
            if d > threshold
        ]

        chunks = []
        start = 0

        for bp in breakpoints:
            group = sentences[start:bp + 1]
            chunk = ' '.join([x['sentence'] for x in group])

            chunks.append(chunk)

            start = bp + 1

        if start < len(sentences):
            group = sentences[start:]
            chunk = ' '.join([x['sentence'] for x in group])
            chunks.append(chunk)

        return chunks

    def chunk(self, text):

        sentences = self.split_sentences(text)

        sentences = self.create_sentence_dict(sentences)

        sentences = self.combine_sentences(sentences)

        sentences = self.embed_sentences(sentences)

        distances, sentences = self.calculate_distances(sentences)

        chunks = self.create_chunks(sentences, distances)

        return chunks


@lru_cache(maxsize=1)
def get_semantic_chunker() -> SemanticChunker:
    return SemanticChunker()
