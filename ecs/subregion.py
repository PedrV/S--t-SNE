from abc import ABC, abstractmethod
from typing import final

import numpy as np
import numpy.typing as npt


class SubRegion(ABC):
    decay_iteration = 1 / 100  # eta
    decay_points = 1 / 100
    beta = 1.6
    alpha = 0.88

    def __init__(self, centroid_x, centroid_y, min_x, max_x):
        self.centroid_x = centroid_x
        self.centroid_y = centroid_y
        self.min_x = min_x
        self.max_x = max_x

    @final
    def consult_iteration_consensus(self, iteration):
        return SubRegion.alpha * np.exp(-iteration * SubRegion.decay_iteration + SubRegion.beta) + 1

    @final
    def consult_point_consensus(self, iteration):
        return np.exp(iteration * SubRegion.decay_points)

    @abstractmethod
    def construct_region(self, convex_region, **kwargs):
        pass

    @abstractmethod
    def do_iteration(self, new_points_inside: npt.NDArray, against=None):
        pass

    @abstractmethod
    def evaluate_new_point(self, new_point: np.ndarray(shape=(2,))):
        pass

    @abstractmethod
    def get_details(self):
        pass

    @abstractmethod
    def sliced(self, against=None):
        pass
