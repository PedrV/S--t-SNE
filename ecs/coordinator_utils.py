def calculate_common_points(new_region, old_region, epsilon):
    _common_points = []
    for i in range(len(new_region)):
        for j in range(len(old_region)):
            if (abs(old_region[j] - new_region[i]) < epsilon).all():
                _common_points.append((i, j))
    return _common_points


def merge_details_median(old_details, new_details, total_iteration):
    new_details['left_iteration'] = total_iteration
    new_details['right_iteration'] = total_iteration

    for reg in old_details['counter'].keys():
        if new_details['counter'][reg][0] != 0:
            new_details['counter'][reg][1] = total_iteration
        else:
            new_details['counter'][reg][1] = old_details['counter'][reg][1]

        new_details['counter'][reg][0] += old_details['counter'][reg][0]

    return new_details


def merge_details_concentric(old_details, new_details, total_iteration):
    new_details['outside_iteration'] = total_iteration
    new_details['inside_iteration'] = total_iteration

    for reg in old_details['counter'].keys():
        if new_details['counter'][reg][0] != 0:
            new_details['counter'][reg][1] = total_iteration
        else:
            new_details['counter'][reg][1] = old_details['counter'][reg][1]

        new_details['counter'][reg][0] += old_details['counter'][reg][0]

    return new_details
