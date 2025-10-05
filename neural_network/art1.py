"""
ART1 (Adaptive Resonance Theory 1) Algorithm
Source: https://en.wikipedia.org/wiki/Adaptive_resonance_theory

ART1 is a neural network model for clustering binary input vectors.

Time Complexity: O(n * m) for n inputs and m clusters
"""

from typing import List, Optional
import numpy as np

class ART1:
    def __init__(self, input_size: int, vigilance: float = 0.75):
        self.input_size = input_size
        self.vigilance = vigilance
        self.weights = []  # list of weight vectors

    def _match(self, x: np.ndarray, w: np.ndarray) -> float:
        return np.sum(np.minimum(x, w)) / (np.sum(x) + 1e-9)

    def train(self, X: List[List[int]]) -> List[int]:
        """
        Train ART1 on binary vectors X.

        >>> art = ART1(input_size=4, vigilance=0.75)
        >>> labels = art.train([[1,0,1,0],[1,0,1,1],[0,1,0,1]])
        >>> len(labels) == 3
        True
        """
        X = np.array(X)
        labels = []

        for x in X:
            assigned = False
            for i, w in enumerate(self.weights):
                if self._match(x, w) >= self.vigilance:
                    # update weights
                    self.weights[i] = np.minimum(x, w)
                    labels.append(i)
                    assigned = True
                    break
            if not assigned:
                self.weights.append(x.copy())
                labels.append(len(self.weights) - 1)
        return labels
