from timeit import default_timer as timer

import numpy as np
from scipy.spatial import Delaunay

from ecs.median_region import MedianRegion
from ecs.median_utils import point_side
from ecs.concentric_region import ConcentricRegion


def reverse_index(arr_length, indexes):
    mask = np.ones(arr_length, dtype=bool)
    mask[indexes] = False
    return np.where(mask)


class ConvexPolygon:
    layers_distance_ratio = [1 / 3, 2 / 3]  # Used for concentric regions

    def __init__(self, convex_hull, id, counted_points=0):
        self.id = id  # to hell with default object identifiers
        self.points_inside = counted_points
        self.__started_from_scipy_convex = True

        try:  # retain points of interest in scipy counterclockwise order
            self.convex_region = convex_hull.points[convex_hull.vertices]
            self.points_inside += convex_hull.points.shape[0] - convex_hull.vertices.shape[0]
        except AttributeError:
            self.__started_from_scipy_convex = False
            self.convex_region = convex_hull

        self.__test_delaunay = Delaunay(self.convex_region)
        self.centroid_x = np.mean(self.convex_region[:, 0])
        self.centroid_y = np.mean(self.convex_region[:, 1])

        self.min_x = min(self.convex_region[:, 0])
        self.min_y = min(self.convex_region[:, 1])
        self.max_x = max(self.convex_region[:, 0])
        self.max_y = max(self.convex_region[:, 1])

        self.binary_sub_regions = []  # Legacy of median regions
        self.concentric_sub_regions = []
        self.initialize_sub_regions()

        self.cut_k1 = None
        self.cut_k2 = None
        self.cut_direction = None
        self.cut_ratio = None

        # Doesn't start with scipy meaning it's a "merge", hence everything was already calculated
        if self.__started_from_scipy_convex:  # Used mainly when new sections appear
            interior_points = convex_hull.points[reverse_index(convex_hull.points.shape[0], convex_hull.vertices)]
            self.verify_for_slices(0, interior_points)
            self.verify_for_slices(1, interior_points)

    def initialize_sub_regions(self):
        for i in range(len(self.convex_region)):
            self.binary_sub_regions.append(MedianRegion(self.convex_region[i], self.centroid_x, self.centroid_y,
                                                        self.min_x, self.max_x, self.convex_region))

        for ratio in ConvexPolygon.layers_distance_ratio:
            c_reg = ConcentricRegion(self.centroid_x, self.centroid_y, self.min_x, self.max_x)
            c_reg.construct_region(self.convex_region, ratio=ratio)
            self.concentric_sub_regions.append(c_reg)

    def verify_for_slices(self, region_type, points=None):
        if region_type == 0:
            subregions = self.binary_sub_regions
        else:
            subregions = self.concentric_sub_regions

        for subregion in subregions:
            if points is not None:  # or region_type == 1
                proceed = subregion.do_iteration(points)
            else:
                proceed = subregion.sliced()

            if proceed == 1:  # Concentric can only activate this, meaning a shrink
                if region_type == 0:
                    self.cut_k1 = subregion.key_point1
                    self.cut_k2 = subregion.key_point2
                    self.cut_direction = 1
                else:
                    self.cut_ratio = subregion.ratio
                break  # (bugs_improvements 2.2)
            elif proceed == 2:
                self.cut_k1 = subregion.key_point1
                self.cut_k2 = subregion.key_point2
                self.cut_direction = 2
                break

    def update_concentric_regions(self):
        if self.cut_ratio is None:
            return None
        print("Cutting at ratio {0}.".format(self.cut_ratio))

        for concentric_region in self.concentric_sub_regions:
            if concentric_region.ratio == self.cut_ratio:
                return np.array(concentric_region.concentric_region).reshape(-1, 2)

    def update_median_regions(self):
        if self.cut_direction is None:
            return None
        print("Cutting the {0} part of {1}".format("left" if self.cut_direction == 1 else "right", self.cut_k1))

        new_convex_region = []
        for point in self.convex_region:
            if (point == self.cut_k1).all():
                new_convex_region.append(point)
                continue
            side = point_side(self.cut_k1, self.cut_k2, point)
            if side != self.cut_direction:
                new_convex_region.append(point)
        new_convex_region.append(self.cut_k2)
        return np.array(new_convex_region).reshape(-1, 2)

    def get_binary_subregions_details(self, key_point1):
        for i, s in zip(range(len(self.binary_sub_regions)), self.binary_sub_regions):
            if (s.key_point1 == key_point1).all() or (s.key_point2 == key_point1).all():
                return i, s.get_details()

    def __eq__(self, other):
        """Overrides the default implementation"""
        if isinstance(other, ConvexPolygon):
            return self.id == other.id
        return False

    def test_generate_dummy_points(self):
        """
        For testing purposes only! Generates points uniformly at random inside the convex polygon.
        """
        points_to_generate = max(self.points_inside, 200)  # Perhaps always use 200
        artificial_points = np.zeros((points_to_generate, 2))
        for i in range(0, points_to_generate):
            while True:
                p = [np.random.uniform(self.min_x, self.max_x), np.random.uniform(self.min_y, self.max_y)]
                if self.__test_delaunay.find_simplex(p) >= 0 and all([(e != p).any() for e in self.convex_region]):
                    artificial_points[i, :] = p
                    break
        return artificial_points
