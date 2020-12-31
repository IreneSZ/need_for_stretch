import copy
import os
import sqlite3
import threading
import time
from collections import deque
from datetime import datetime, timedelta
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import math
import pyautogui
import torch
from loguru import logger
from op import model, util
from op.body import Body
from op.hand import Hand
from playsound import playsound

from detect_position.position_detection import baseline
from lock_unlock import (continuous_stretch, get_position_records,
                         get_strech_metrics, lock_pc, type_password)
from posture_correction import hand_on_face, head_shoulder
from record_key_points import (get_points_webcam, process_candidate,
                               record_points, sit_or_stand)
from reader import DebugReader, VideoReader, Reader
from argparse import ArgumentParser


#########################################
# every 30 seconds:
# 1. record new key points
# 2. if person detected, check posture
# 3. if PC unlocked, check sitting/standing for the past 35 min, screentime for past 45 min
# 4. if PC locked, count off-screen time
# 5. if PC locked and off-screen time >= 5 min, set status to "unlockable", check for stretch


def capture_and_openpose(reader: Reader, last_unlocked_time, screen_time, openpose_interval, pct_sitting) -> bool:
    lst_points, data = get_points_webcam(reader)
    lst_points = sit_or_stand(lst_points, data)
    record_points(db_path, lst_points)

    # posture correction, only execute when person is detected
    if lst_points[-1] != -1:
        wrong_position = head_shoulder(data)
        wrong_hand = hand_on_face(data, 30)
        if wrong_position or wrong_hand:
            playsound('./sound_effects/alarm.m4a')

    # lock if sitting for too long
    last_position_records = get_position_records(db_path, last_unlocked_time)

    locked = False
    if last_position_records.count(1) > math.floor(pct_sitting * screen_time / openpose_interval):
        locked = True
        logger.info('Sitting for too long, will lock PC.')
        pyautogui.hotkey('winleft', 'l')
        time.sleep(5)

    return locked


def count_offscreen(reader: Reader, last_locked_time, offscreen_time, openpose_interval) -> bool:
    offscreen_enough = False
    num_offscreen_records = math.floor(offscreen_time / openpose_interval)
    lst_points, data = get_points_webcam(reader)
    lst_points = sit_or_stand(lst_points, data)
    record_points(db_path, lst_points)
    last_position_records = get_position_records(db_path, last_locked_time)
    if last_position_records.count(-1) >= num_offscreen_records:
        offscreen_enough = True
    return offscreen_enough


def unlock(reader: Reader, stretch_time) -> bool:
    if continuous_stretch(stretch_time, reader, min_elbow, max_elbow, min_shoulder, max_shoulder, min_hip, max_hip, min_knee, max_knee):
        type_password()
        return True
    return False


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--debug', action='store_true', help='Toggle debug mode.')
    parser.add_argument('--screen_time', default=30, help='number of seconds in front of the screen')
    parser.add_argument('--pct_sitting', default=0.9, help='max percent of sitting time in front of the screen')
    parser.add_argument('--openpose_interval', default=3, help='num of seconds between each pose estimation run')
    parser.add_argument('--dir_stretch', default='./squat_img/', help='the dir to store self-defined stretch pose images')
    parser.add_argument('--offscreen_time', default=5, help='number of seconds needed before unlock process can start')
    parser.add_argument('--stretch_time', default=3, help='number of seconds needed for the stretch poses')

    args = parser.parse_args()

    # openpose model and position classification model
    body_estimation = Body('body_pose_model.pth')
    model = baseline(36, 20, 10, 8, 5, 2)
    model.load_state_dict(torch.load('detect_position/baseline.pt'))
    model.eval()

    # create database to store the timestamp, key points coordinates and position status 
    db_path = Path('daily_log.db')
    try:
        conn = sqlite3.connect(f'file:{db_path}?mode=rw', uri=True)
    except sqlite3.OperationalError:
        # Path exists but not a database.
        if db_path.exists():
            raise ValueError(
                f'{db_path} exists, but it is not not a valid database.')

        # Create a database here.
        col_names = ['timestamp', 'nose', 'neck', 'l_shoulder', 'l_elbow', 'l_hand', 'r_shoulder', 'r_elbow', 'r_hand', 'l_hip', 'l_knee', 'l_foot', 
                    'r_hip', 'r_knee', 'r_foot', 'l_eye', 'r_eye', 'l_ear', 'r_ear', 'position']

        col_types = ['datetime']
        col_types += ['float NOT NULL'] * 18
        col_types += ['int NOT NULL']

        col_command = 'timestamp text, '
        N = len(col_names)
        for i in range(1, N-1):
            name1 = col_names[i] + '_x'
            new_item1 = name1 + ' ' + col_types[i] + ', '
            name2 = col_names[i] + '_y'
            new_item2 = name2 + ' ' + col_types[i] + ', '

            col_command += new_item1
            col_command += new_item2

        col_command += 'position int NOT NULL'

        conn = sqlite3.connect(f'file:{db_path}?mode=rwc', uri=True)
        cur = conn.cursor()
        create_query = f"""
            CREATE TABLE records (
            {col_command}
            );
        """
        cur.execute(create_query)
        conn.commit()
        conn.close()
        logger.info(f'Database created at {db_path}.')

    # get the data for stretch positions
    min_elbow, max_elbow, min_shoulder, max_shoulder, min_hip, max_hip, min_knee, max_knee = get_strech_metrics(img_folder=args.dir_stretch)

    cam_reader = VideoReader(debug=args.debug)
    # if args.debug:
    #     unlock_reader = DebugReader('./squat_test')
    # else:
    unlock_reader = cam_reader


    last_unlocked_time = str(datetime.now().time())
    while True:
        locked = False
        while not locked:
            time.sleep(args.openpose_interval)
            locked = capture_and_openpose(cam_reader, last_unlocked_time, args.screen_time, args.openpose_interval, args.pct_sitting)

        last_locked_time = str(datetime.now().time())
        # off screen count for 5 min
        offscreen_enough = False
        while not offscreen_enough:
            offscreen_enough = count_offscreen(cam_reader, last_locked_time, args.offscreen_time, args.openpose_interval)
            time.sleep(args.openpose_interval)
        logger.info('Enough off screen time, will start detecting stretch')

        unlocked = False
        while not unlocked:
            unlocked = unlock(unlock_reader, args.stretch_time)
        last_unlocked_time = str(datetime.now().time())
        logger.info('Stretch detected, will unlock screen')
