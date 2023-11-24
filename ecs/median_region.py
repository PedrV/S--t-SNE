import numpy as np
import numpy.typing as npt

from ecs.median_utils import point_side, discover_keypoint2
from ecs.subregion import SubRegion


class MedianRegion(SubRegion):
    def __init__(self, point1: np.ndarray(shape=(2,)), centroid_x, centroid_y, min_x, max_x, convex_region,
                 counter=None, left_iterations=0, right_iterations=0):

        super().__init__(centroid_x, centroid_y, min_x, max_x)
        self.left_iteration = left_iterations
        self.right_iteration = right_iterations
        if counter is None:
            counter = {"delta_left": [0, 0], "delta_right": [0, 0], "delta_mid": [0, 0]}
        self.key_point1 = point1
        self.counter = counter
        self.key_point2 = None
        self.construct_region(convex_region)

    def construct_region(self, convex_region, **kwargs):
        discovered_point_x, discovered_point_y = discover_keypoint2(self.key_point1, self.centroid_x, self.centroid_y,
                                                                    self.min_x, self.max_x, convex_region)
        self.key_point2 = np.array([discovered_point_x, discovered_point_y]).reshape(2, )

    def do_iteration(self, new_points_inside: npt.NDArray, against=None):  # npt: (2,blob1) where blob1 is the number of points
        for i in range(new_points_inside.shape[0]):
            self.evaluate_new_point(new_points_inside[i, :])
        return self.sliced()

    def evaluate_new_point(self, new_point: np.ndarray(shape=(2,))):
        decision = point_side(self.key_point1, self.key_point2, new_point)
        if decision == 1:
            self.counter["delta_left"][0] += 1
            self.counter["delta_left"][1] = self.left_iteration
        elif decision == 2:
            self.counter["delta_right"][0] += 1
            self.counter["delta_right"][1] = self.right_iteration
        else:
            self.counter["delta_mid"][0] += 1
            self.counter["delta_mid"][1] = self.right_iteration

    def sliced(self, against=None):
        slice_left = self.can_slice_left()
        slice_right = self.can_slice_right()
        self.left_iteration += 1
        self.right_iteration += 1

        if slice_left:
            return 1
        elif slice_right:
            return 2
        return 0

    def can_slice_left(self):
        expected_gap_between_last_change = self.consult_iteration_consensus(self.left_iteration)
        # expected_point_balance = self.consult_point_consensus(self.left_iteration)
        # if self.counter["delta_left"][0] < expected_point_balance:
        #    return True

        # No >=. If so change inside iteration behaviour or new regions can be sliced right after merged with old ones
        return (self.left_iteration - self.counter["delta_left"][1]) > expected_gap_between_last_change

    def can_slice_right(self):
        expected_gap_between_last_change = self.consult_iteration_consensus(self.right_iteration)
        # expected_point_balance = self.consult_point_consensus(self.right_iteration)
        # if self.counter["delta_right"][0] < expected_point_balance:
        #    return True

        # No >=. If so change inside iteration behaviour or new regions can be sliced right after merged with old ones
        return (self.right_iteration - self.counter["delta_right"][1]) > expected_gap_between_last_change

    def get_details(self):
        return {"counter": self.counter, "left_iteration": self.left_iteration,
                "right_iteration": self.right_iteration, "keypoint": self.key_point1}

    def __repr__(self):
        return "k1: {0}; k2: {1}".format(self.key_point1, self.key_point2)
