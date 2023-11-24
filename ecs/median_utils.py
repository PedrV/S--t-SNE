# https://stackoverflow.com/questions/563198/how-do-you-detect-where-two-line-segments-intersect/565282#565282
def get_line_intersection(p0_x: float, p0_y: float, p1_x: float, p1_y: float, p2_x: float, p2_y: float, p3_x: float,
                          p3_y: float):
    s1_x = p1_x - p0_x
    s1_y = p1_y - p0_y
    s2_x = p3_x - p2_x
    s2_y = p3_y - p2_y

    # should not work for collinear and parallel cases
    s = (-s1_y * (p0_x - p2_x) + s1_x * (p0_y - p2_y)) / (-s2_x * s1_y + s1_x * s2_y)
    t = (s2_x * (p0_y - p2_y) - s2_y * (p0_x - p2_x)) / (-s2_x * s1_y + s1_x * s2_y)

    if s >= 0 and s <= 1 and t >= 0 and t <= 1:
        i_x = p0_x + (t * s1_x)
        i_y = p0_y + (t * s1_y)
        return i_x, i_y
    return None, None


def discover_keypoint2(keypoint1, centroid_x, centroid_y, min_x, max_x, hull_keypoints):
    """
    To form all binary regions it takes O(2n^2) where n is the number of points in ht convex hull
    Update  binary regions with new points is Theta(1)
    Update binary regions by deformation is **hopefully** O(2n)
    first test all points to see if any is collinear up to an epsilon, if not, test intersection
    """

    m = (keypoint1[1] - centroid_y) / (keypoint1[0] - centroid_x)
    b = (keypoint1[0] * centroid_y - centroid_x * keypoint1[1]) / (keypoint1[0] - centroid_x)

    if keypoint1[0] <= centroid_x:
        # print("MAX")
        m_point_x = max_x
    else:
        # print("MIN")
        m_point_x = min_x

    discovered_point_x, discovered_point_y = None, None
    for i in range(len(hull_keypoints)):
        p1 = hull_keypoints[i]
        if i == len(hull_keypoints) - 1:
            p2 = hull_keypoints[0]
        else:
            p2 = hull_keypoints[i + 1]

        if (p1 == keypoint1).all() or (p2 == keypoint1).all():
            continue

        cross_x, cross_y = get_line_intersection(keypoint1[0], keypoint1[1], m_point_x, m * m_point_x + b,
                                                 p2[0], p2[1], p1[0], p1[1])
        if cross_x is not None:
            discovered_point_x = cross_x
            discovered_point_y = cross_y
            # print(discovered_point_x, discovered_point_y)
            break

    if discovered_point_x is None and discovered_point_y is None:
        raise ValueError('Intersection not found for {0}.'.format(keypoint1),
                         'Consider review m_point orientation',
                         'Check collinearity, or check if the points given by [({0}, {1}*{2}+{3}), keypoint]'.format(
                             m_point_x, m, m_point_x, b), ' are parallel.')

    return discovered_point_x, discovered_point_y


def point_side(key_point1, key_point2, new_point):
    decision = (new_point[0] - key_point1[0]) * (key_point2[1] - key_point1[1]) - \
               (new_point[1] - key_point1[1]) * (key_point2[0] - key_point1[0])

    signal_of_left_side = ((key_point1[0] - 1) - key_point1[0]) * (key_point2[1] - key_point1[1]) - \
                          (key_point1[1] - key_point1[1]) * (key_point2[0] - key_point1[0])

    if decision < 0 and signal_of_left_side < 0 or decision > 0 and signal_of_left_side > 0:
        return 1
    elif decision < 0 and signal_of_left_side > 0 or decision > 0 and signal_of_left_side < 0:
        return 2
    return 0
