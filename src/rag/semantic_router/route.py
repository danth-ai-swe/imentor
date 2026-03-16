from typing import List


class Route:
    def __init__(self, name: str = None, samples: List = None):
        if samples is None:
            samples = []
        self.name = name
        self.samples = samples
