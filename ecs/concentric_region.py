import numpy as np
import numpy.typing as npt
from scipy.spatial import Delaunay

from ecs.subregion import SubRegion


class ConcentricRegion(SubRegion):
    def __init__(self, centroid_x, centroid_y, min_x, max_x, inside_iteration=0, outside_iteration=0):
        super().__init__(centroid_x, centroid_y, min_x, max_x)
        self.concentric_region = []
        self.triangular_tesselation = None
        self.counter = {"delta_inside": [0, 0], "delta_outside": [0, 0]}
        self.inside_iteration = inside_iteration
        self.outside_iteration = outside_iteration  # For statistical purposes
        self.ratio = 0

    def construct_region(self, convex_region, **kwargs):
        self.ratio = kwargs["ratio"]
        for x, y in convex_region:
            dist_x = abs(x-self.centroid_x)*self.ratio
            dist_y = abs(y-self.centroid_y)*self.ratio
            if x < self.centroid_x:
                new_x = self.centroid_x - dist_x
            else:
                new_x = self.centroid_x + dist_x
            if y < self.centroid_y:
                new_y = self.centroid_y - dist_y
            else:
                new_y = self.centroid_y + dist_y

            self.concentric_region.append([new_x, new_y])
        self.triangular_tesselation = Delaunay(self.concentric_region)  # n*log n where n is the number of vertices

    def do_iteration(self, new_points_inside: npt.NDArray, against=None):
        for i in range(new_points_inside.shape[0]):
            self.evaluate_new_point(new_points_inside[i, :])
        return self.sliced(against)

    def evaluate_new_point(self, new_point: np.ndarray(shape=(2,))):
        decision = self.triangular_tesselation.find_simplex(new_point) >= 0
        if decision:  # inside
            self.counter["delta_inside"][0] += 1
            self.counter["delta_inside"][1] = self.inside_iteration
        else:
            self.counter["delta_outside"][0] += 1
            self.counter["delta_outside"][1] = self.outside_iteration

    def sliced(self, against=None):
        shrink = self.can_shrink()
        self.inside_iteration += 1
        self.outside_iteration += 1
        return shrink

    def can_shrink(self):
        expected_gap_between_last_change = self.consult_iteration_consensus(self.inside_iteration)
        # expected_point_balance = self.consult_point_consensus(self.left_iteration)
        # if self.counter["delta_inside"][0] < expected_point_balance:
        #    return True

        # No >=. If so change inside iteration behaviour or new regions can be sliced right after merged with old ones
        return (self.inside_iteration - self.counter["delta_outside"][1]) > expected_gap_between_last_change

    def get_details(self):
        return {"counter": self.counter, "inside_iteration": self.inside_iteration, "ratio": self.ratio,
                "outside_iteration": self.outside_iteration}

    def __repr__(self):
        return "ratio: {0}; inside {1}; iteration: {2}".format(self.ratio, self.counter["inside_iteration"],
                                                               self.inside_iteration)
