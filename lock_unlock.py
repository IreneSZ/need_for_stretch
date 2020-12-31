import datetime
import math
import os
import sqlite3
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from sqlite3 import Error
from typing import Dict, Tuple

import cv2
import numpy as np
import pyautogui
from loguru import logger
from op import model, util
from op.body import Body
from op.hand import Hand
from scipy import spatial
from scipy.spatial import distance

from model import Model
from record_key_points import (get_points_webcam, process_candidate,
                               record_points)


def get_position_records(db_path, records_start_time):
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


def type_password(password: str):
    """
    type in the password to unlock pc
    """
    pyautogui.write(password, interval=0.2)
    pyautogui.typewrite(['enter'])


def get_points_img(img_path, model: Model):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (320, 240))
    candidate, subset = model.estimate_body(img)
    _, data = process_candidate(candidate)
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


@dataclass
class StretchMetrics:
    elbow: Tuple[float, float]
    shoulder: Tuple[float, float]
    hip: Tuple[float, float]
    knee: Tuple[float, float]

    def in_range(self, values: Dict[str, float]) -> bool:
        attr_dict = asdict(self)
        for part, v in values.items():
            min_v, max_v = attr_dict[part]
            if v > max_v or v < min_v:
                return False
        return True


def get_strech_metrics(img_folder, model: Model) -> StretchMetrics:
    lst_elbow_angle = []
    lst_shoulder_angle = []
    lst_hip_angle = []
    lst_knee_angle = []

    lst_stretch_img = os.listdir(img_folder)

    for img in lst_stretch_img:
        img_path = img_folder + img
        data = get_points_img(img_path, model)

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

    return StretchMetrics(
        (min_elbow, max_elbow),
        (min_shoulder, max_shoulder),
        (min_hip, max_hip),
        (min_knee, max_knee))


def show_readable_angle(part: str, angle: float):
    angle = angle * 180 / 3.1415926
    print(f'{part} angle {angle:.1f}.')


def is_squat(data, stretch_metrics: StretchMetrics) -> bool:
    elbow = three_point_angle(data, 4, 3, 2)
    shoulder = three_point_angle(data, 3, 2, 8)
    hip = three_point_angle(data, 2, 8, 9)
    knee = three_point_angle(data, 8, 9, 10)
    values = {
        'elbow': elbow,
        'shoulder': shoulder,
        'hip': hip,
        'knee': knee
    }
    for k, v in values.items():
        show_readable_angle(k, v)

    is_in_range = stretch_metrics.in_range(values)
    prefix = '' if is_in_range else 'not '
    logger.info(prefix + f'squat: {elbow}, {shoulder}, {hip}, {knee}')
    return is_in_range


def continuous_stretch(stretch_time, reader, stretch_metrics: StretchMetrics, model: Model) -> bool:
    """
    need to hold the position for stretch_time seconds, continuously
    """
    num_stretch = 0
    start_time = time.time()
    while num_stretch < stretch_time:
        if time.time() - start_time >= 1:
            _, data = get_points_webcam(reader, model)
            if is_squat(data, stretch_metrics):
                num_stretch += 1
            else:
                num_stretch = 0
            logger.info(f'continuous seconds of stretch: {num_stretch}')
            start_time = time.time()

    return True
