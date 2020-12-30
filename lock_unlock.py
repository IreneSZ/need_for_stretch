import datetime
import math
import os
import sqlite3
import time
from pathlib import Path
from sqlite3 import Error

import cv2
import numpy as np
import pyautogui
from loguru import logger
from op import model, util
from op.body import Body
from op.hand import Hand
from scipy import spatial
from scipy.spatial import distance

from record_key_points import (get_points_webcam, process_candidate,
                               record_points)

body_estimation = Body('body_pose_model.pth')

# model = baseline(36, 20, 10, 8, 5, 2)
# model.load_state_dict(torch.load('detect_position/baseline.pt'))
# model.eval()

if not os.path.isdir('./squat_img'):
    os.mkdir('./squat_img')


# the function to retrieve the last N records of the database
def get_position_records(db_path, records_start_time):
    """
    num_records: number of records to get
    """

    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    #lst_positions = cur.execute(f'SELECT position FROM records ORDER BY timestamp DESC LIMIT {earliest_timestamp}').fetchall()
    lst_positions = cur.execute(
        f'SELECT position FROM records where time(timestamp) >= "{records_start_time}"').fetchall()
    conn.close()
    position_records = [x[0] for x in lst_positions]
    logger.info(f'position records {position_records}')
    logger.info(f'last unlocked time {records_start_time}')
    return position_records


def lock_pc(last_n_records, num_records, pct_sitting):
    """
    if sitting for over % pct_sitting in the last n records, lock screen
    """
    if last_n_records.count(1) >= math.floor(num_records * pct_sitting):
        logger.info(
            f'sitting for too long, will lock pc {last_n_records.count(1)}.')
        pyautogui.hotkey('winleft', 'l')
    return True


def unlock_pc():
    """
    type in the password to unlock pc
    """
    pyautogui.write('justtrAIit', interval=0.2)
    pyautogui.typewrite(['enter'])


def get_points_oneshot(img_path):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (320, 240))
    candidate, subset = body_estimation(img)
    lst_points, data = process_candidate(candidate)

    return data


def three_point_angle(data, idx_start, idx_mid, idx_end):
    """
    calculate the cosine of the angle created by the three points, with ind_mid being the vertex
    """
    start = [data[0, idx_start], data[1, idx_start]]
    mid = [data[0, idx_mid], data[1, idx_mid]]
    end = [data[0, idx_end], data[1, idx_end]]

    # both vectors point to the direction away from the mid point
    vec1 = [start[0] - mid[0], mid[1] - start[1]]
    vec2 = [end[0] - mid[0], mid[1] - end[1]]

    cos = (vec1[0]*vec2[0] + vec1[1]*vec2[1]) / (math.sqrt(vec1[0]
                                                           ** 2 + vec1[1]**2) * math.sqrt(vec2[0]**2 + vec2[1]**2))
    angle = math.acos(cos)

    return angle


def get_strech_metrics(img_folder):
    lst_elbow_angle = []
    lst_shoulder_angle = []
    lst_hip_angle = []
    lst_knee_angle = []

    lst_stretch_img = os.listdir(img_folder)

    for img in lst_stretch_img:
        img_path = img_folder + img
        data = get_points_oneshot(img_path)

        elbow_angle = three_point_angle(data, 4, 3, 2)
        lst_elbow_angle.append(elbow_angle)

        shoulder_angle = three_point_angle(data, 3, 2, 8)
        lst_shoulder_angle.append(shoulder_angle)

        hip_angle = three_point_angle(data, 2, 8, 9)
        lst_hip_angle.append(hip_angle)

        knee_angle = three_point_angle(data, 8, 9, 10)
        lst_knee_angle.append(knee_angle)

    min_elbow = min(lst_elbow_angle)
    max_elbow = max(lst_elbow_angle)
    min_shoulder = min(lst_shoulder_angle)
    max_shoulder = max(lst_shoulder_angle)
    min_hip = min(lst_hip_angle)
    max_hip = max(lst_hip_angle)
    min_knee = min(lst_knee_angle)
    max_knee = max(lst_knee_angle)

    return min_elbow, max_elbow, min_shoulder, max_shoulder, min_hip, max_hip, min_knee, max_knee


def show_readable_angle(part: str, angle: float):
    angle = angle * 180 / 3.1415926
    print(f'{part} angle {angle:.1f}.')


def is_squat(data, min_elbow, max_elbow, min_shoulder, max_shoulder, min_hip, max_hip, min_knee, max_knee):
    elbow = three_point_angle(data, 4, 3, 2)
    shoulder = three_point_angle(data, 3, 2, 8)
    hip = three_point_angle(data, 2, 8, 9)
    knee = three_point_angle(data, 8, 9, 10)
    show_readable_angle('elbow', elbow)
    show_readable_angle('shoulder', shoulder)
    show_readable_angle('hip', hip)
    show_readable_angle('knee', knee)

    if min_elbow <= elbow <= max_elbow and min_shoulder <= shoulder <= max_shoulder and min_hip <= hip <= max_hip and min_knee <= knee <= max_knee:
        logger.info(f'squat: {elbow}, {shoulder}, {hip}, {knee}')
        return True
    else:
        logger.info(f'not squat: {elbow}, {shoulder}, {hip}, {knee}')
        return False


# numpy array to store the min and max of the cosine values of the three key angle points
def continuous_stretch(num_seconds, reader, min_elbow, max_elbow, min_shoulder, max_shoulder, min_hip, max_hip, min_knee, max_knee):
    """
    need to hold the position for num_seconds seconds, continuously
    """
    num_stretch = 0
    start_time = time.time()
    while num_stretch < num_seconds:
        if time.time() - start_time >= 1:
            _, data = get_points_webcam(reader)
            if is_squat(data, min_elbow, max_elbow, min_shoulder, max_shoulder, min_hip, max_hip, min_knee, max_knee):
                num_stretch += 1
            else:
                num_stretch = 0
            logger.info(f'continuous seconds of stretch: {num_stretch}')
            start_time = time.time()

    return True


# if __name__ == '__main__':


# print("elbow", lst_elbow_angle)
# print(lst_hip_angle)
# print(lst_knee_angle)


# def calculate_point_distance(point1, point2):
#     """
#     both inputs are 2*18 arrays
#     only use key points 1 - 13 because the rest of the keypoints do not really matter in this task
#     """
#     point1 = point1[:, 1:14]
#     point2 = point2[:, 1:14]

#     point1 = point1.flatten()
#     point2 = point2.flatten()
#     return(distance.euclidean(point1, point2))


# def squat_or_not(capture, data_sit, data_stand, data_squat):
#     data_new =
#     dist2sit =


# point1 = get_points_oneshot('./oneshot_sitting.jpg')
# point2 = get_points_oneshot('./oneshot_standing.jpg')
# point3 = get_points_oneshot('./oneshot_squat.jpg')


# last_n_records = [1, 1, 1, 1, 1, 1]
# num_records = 5
# pct_sigging = 0.95

# lock_pc(last_n_records, num_records, pct_sigging)
