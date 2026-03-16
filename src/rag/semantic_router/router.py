import numpy as np


class SemanticRouter:
    def __init__(self, embedding, routes):
        self.routes = routes
        self.embedding = embedding
        self.routes_embedding = {}

        for route in self.routes:
            self.routes_embedding[
                route.name
            ] = self.embedding.encode(route.samples)

    def get_routes(self):
        return self.routes

    def guide(self, query):
        query_embedding = self.embedding.encode([query])
        query_embedding = query_embedding / np.linalg.norm(query_embedding)
        scores = []

        # Calculate the cosine similarity of the query embedding with the sample embeddings of the router.

        for route in self.routes:
            routes_embedding = self.routes_embedding[route.name] / np.linalg.norm(self.routes_embedding[route.name])
            score = np.mean(np.dot(routes_embedding, query_embedding.T).flatten())
            scores.append((score, route.name))

        scores.sort(reverse=True)
        return scores[0]
