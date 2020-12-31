import math

import numpy as np
from loguru import logger


def calculate_angle(coor1, coor2):
    """
    coor1: left shoulder (x, y)
    coor2: left ear / left eye / nose (x, y)
    """
    # a vector starting from shoulder to any other point
    x1, y1 = coor1
    x2, y2 = coor2
    cos = (x2 - x1) / math.sqrt((x2 - x1) ** 2 + (y1 - y2) ** 2)

    return cos


def dist(coor1, coor2):
    """
    coor1, coor2: (x, y) 
    """
    x1, y1 = coor1
    x2, y2 = coor2
    distance = math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

    return distance


def head_shoulder(data):
    """
    data: the (2 * 18) numpy array that stores all 18 key points' coordinates
    (if any key point is missing, its coordinate is (-1, -1))
    use any one from the nose - shoulder, left ear - shoulder, left eye - shoulder pairs coordinates to evaluate head-shoulder posture
    """
    # key point indices:
    # left ear: 16
    # left eye: 14
    # nose: 0
    # left shoulder: 2

    l_ear = get_coords(16, data)
    l_eye = get_coords(14, data)
    nose = get_coords(0, data)
    l_shoulder = get_coords(2, data)

    # ind_part: indicator of whether needs posture correction, so True = bad pose
    cos_ear = 2
    cos_eye = 2
    cos_nose = 2

    ind_ear = False
    ind_eye = False
    ind_nose = False
    if l_ear != (-1, -1):
        cos_ear = calculate_angle(l_shoulder, l_ear)
        ind_ear = cos_ear < -0.8

    if l_eye != (-1, -1):
        cos_eye = calculate_angle(l_shoulder, l_eye)
        ind_eye = cos_eye < -0.6

    if nose != (-1, -1):
        cos_nose = calculate_angle(l_shoulder, nose)
        ind_nose = cos_nose < -0.73

    if any([ind_ear, ind_eye, ind_nose]):
        logger.info(f'WARNING: wrong head shoulder posture')
        return True
    else:
        return False


def get_coords(idx, data):
    x = data[0, :]
    y = data[1, :]
    return (x[idx], y[idx])


def safe_append(point1, point2, lst):
    if point1 != (-1, -1) and point2 != (-1, -1):
        lst.append(dist(point1, point2))


def hand_on_face(data, threshold):
    """
    returns True if hand on face
    """
    nose = get_coords(0, data)
    l_hand = get_coords(4, data)
    r_hand = get_coords(7, data)
    l_ear = get_coords(16, data)
    r_ear = get_coords(17, data)
    l_eye = get_coords(14, data)
    r_eye = get_coords(15, data)

    lst_dist = []

    for hand in [l_hand, r_hand]:
        for point2 in [l_eye, r_eye, l_ear, r_ear, nose]:
            safe_append(hand, point2, lst_dist)
    # print(lst_dist)
    if min(lst_dist) <= threshold:
        logger.info(f'WARNING: hand on face')
        return True
    else:
        return False
