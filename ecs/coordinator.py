from collections import defaultdict, Counter
from operator import itemgetter
from typing import Final

from scipy.spatial import ConvexHull

from ecs.coordinator_utils import calculate_common_points, merge_details_median, merge_details_concentric
from ecs.convex_polygon import ConvexPolygon


def median_region_handler(old_convex_polygon, new_convex_polygon, matched_old_point_index,
                          matched_new_point_index, total_iterations_of_old_polygon):
    _, old_details = old_convex_polygon.get_binary_subregions_details(
        old_convex_polygon.convex_region[matched_old_point_index])
    i, new_details = new_convex_polygon.get_binary_subregions_details(
        new_convex_polygon.convex_region[matched_new_point_index])

    # Counter dictionary is passed by reference, however, the left/right iter are not
    details = merge_details_median(old_details, new_details, total_iterations_of_old_polygon)
    new_convex_polygon.binary_sub_regions[i].counter = details['counter']  # meaning this useless :)
    new_convex_polygon.binary_sub_regions[i].left_iteration = details['left_iteration']  # this is not
    new_convex_polygon.binary_sub_regions[i].right_iteration = details['right_iteration']  # also this

    return new_convex_polygon


def concentric_region_handler(old_concentric_region, new_concentric_region, total_iterations_of_old_polygon):
    old_details = old_concentric_region.get_details()
    new_details = new_concentric_region.get_details()
    details = merge_details_concentric(old_details, new_details, total_iterations_of_old_polygon)

    new_concentric_region.counter = details['counter']
    new_concentric_region.outside_iteration = details['outside_iteration']
    new_concentric_region.inside_iteration = details['inside_iteration']

    return new_concentric_region


class Coordinator:
    EPSILON: Final = 0.001

    def __init__(self):
        self.total_iterations = Counter()
        self.convex_polygons = {}  # dict of SubRegions
        self._new_convex_polygons = {}
        self.current_id = 0

    def new_batch(self, polygons: list):
        cut_regions = []
        for polygon in polygons:
            self._new_convex_polygons[self.current_id] = ConvexPolygon(polygon, self.current_id)
            self.total_iterations[self.current_id] = 0
            self.current_id += 1

        if self.convex_polygons.__len__() == 0:
            self.convex_polygons = self._new_convex_polygons
        else:
            matched, strong_matched = self.polygon_followup_calculator()
            self.polygon_handler(matched, strong_matched)
            for region_id in self.convex_polygons.keys():
                new_region = self.convex_polygons[region_id].update_median_regions()
                if new_region is not None:
                    self.convex_polygons[region_id] = ConvexPolygon(ConvexHull(new_region), region_id)
                    print("[MEDIAN] Cut on iteration {0}\n".format(self.total_iterations[region_id]))
                    self.total_iterations[region_id] = 0
                    cut_regions.append(region_id)
                    continue

                new_region = self.convex_polygons[region_id].update_concentric_regions()
                if new_region is not None:
                    self.convex_polygons[region_id] = ConvexPolygon(ConvexHull(new_region), region_id)
                    print("[CONCENTRIC] Cut on iteration {0}\n".format(self.total_iterations[region_id]))
                    self.total_iterations[region_id] = 0
                    cut_regions.append(region_id)

        self._new_convex_polygons = {}
        self.total_iterations.update((k for k, v in self.total_iterations.items() if v not in cut_regions))

    def polygon_followup_calculator(self):
        """
        Compute matching between new regions (probably obtained by clustering) and the old regions.
        If half the points of a new region M are within Coordinator.EPSILON of the old region O than
        M is candidate to be the previously observed O.
        The candidate with more points within Coordinator.EPSILON of O gets identified as O.

        Technically allows multiple M to be O, but that is unlikely in the context this is used.
        """
        matched_polygons = {}
        strong_matched_polygons = {}
        is_a_match = (lambda occurrences, region_length: occurrences > region_length // 2)
        is_strong_match = (lambda occurrences, region_length: occurrences == region_length)

        for old_key in self.convex_polygons.keys():
            _candidates = {}
            for new_key in self._new_convex_polygons.keys():
                _index_common_points = calculate_common_points(self._new_convex_polygons[new_key].convex_region,
                                                               self.convex_polygons[old_key].convex_region,
                                                               Coordinator.EPSILON)

                if is_strong_match(len(_index_common_points), len(self.convex_polygons[old_key].convex_region)):
                    _candidates[new_key] = _index_common_points
                    strong_matched_polygons[old_key] = new_key
                elif is_a_match(len(_index_common_points), len(self.convex_polygons[old_key].convex_region)):
                    _candidates[new_key] = _index_common_points

            if _candidates.__len__() != 0:
                matched_polygons[old_key] = max(_candidates.items(), key=itemgetter(1))

        return matched_polygons, strong_matched_polygons

    def polygon_handler(self, matched_polygons, strong_matched_polygons):
        new_polygons_handled = defaultdict(lambda: 0)

        for old_key in list(self.convex_polygons):  # Force copy of keys to be iterated safely
            if old_key not in matched_polygons:
                self.convex_polygons.pop(old_key)
                self.total_iterations.pop(old_key)
            else:  # Update an old_polygon to the new polygon but retain the details of the old_polygon
                new_polygons_handled[matched_polygons[old_key][0]] = 1
                new_convex_polygon = self._new_convex_polygons[matched_polygons[old_key][0]]
                _old_convex_polygon = self.convex_polygons[old_key]

                self.total_iterations.pop(new_convex_polygon.id)
                for matched_new_point_index, matched_old_point_index in matched_polygons[old_key][1]:
                    new_convex_polygon = median_region_handler(_old_convex_polygon, new_convex_polygon,
                                                               matched_old_point_index, matched_new_point_index,
                                                               self.total_iterations[old_key])

                new_convex_polygon.id = old_key
                new_convex_polygon.points_inside = _old_convex_polygon.points_inside + new_convex_polygon.points_inside
                self.convex_polygons[old_key] = new_convex_polygon
                self.convex_polygons[old_key].verify_for_slices(0)

                if self.convex_polygons[old_key].cut_direction is None and old_key in strong_matched_polygons:
                    for i in range(len(_old_convex_polygon.concentric_sub_regions)):
                        for j in range(len(new_convex_polygon.concentric_sub_regions)):
                            old_ratio = _old_convex_polygon.concentric_sub_regions[i].ratio
                            new_ratio = new_convex_polygon.concentric_sub_regions[j].ratio
                            if old_ratio != new_ratio:
                                continue
                            nr = concentric_region_handler(_old_convex_polygon.concentric_sub_regions[i],
                                                           new_convex_polygon.concentric_sub_regions[j],
                                                           self.total_iterations[old_key])

                            self.convex_polygons[old_key].concentric_sub_regions[i] = nr
                    self.convex_polygons[old_key].verify_for_slices(1)

        for new_key in self._new_convex_polygons.keys():
            if not new_polygons_handled[new_key]:  # Completely new regions got verified for cuts at creation
                self.convex_polygons[new_key] = self._new_convex_polygons[new_key]
