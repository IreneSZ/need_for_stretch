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
import pyautogui
import schedule
import torch
from loguru import logger
from op import model, util
from op.body import Body
from op.hand import Hand
from playsound import playsound

from detect_position.position_detection import baseline
from lock_unlock import (continuous_stretch, get_last_n_positions,
                         get_strech_metrics, lock_pc, unlock_pc)
from posture_correction import hand_on_face, head_shoulder
from record_key_points import (get_points_webcam, process_candidate,
                               record_points)
from reader import DebugReader, VideoReader, Reader
from argparse import ArgumentParser

# every 30 seconds, run this script to 
# (1) run openpose, get key points coordinates
# (2) run the model to detect sitting/standing position
# (3) store the timestamp, key points coordinates, position indicators into a csv (one csv for each day)



#########################################
# every 30 seconds: 
# 1. record new key points
# 2. if person detected, check posture
# 3. if PC unlocked, check sitting/standing for the past 35 min, screentime for past 45 min
# 4. if PC locked, count off-screen time
# 5. if PC locked and off-screen time >= 5 min, set status to "unlockable", check for stretch 

# def capture_and_openpose():
#     lst_points, data = get_points_webcam(capture)
#     record_points(db_path, lst_points)

def capture_and_openpose(reader: Reader) -> bool:
    #print('start capturing')
    # capture = cv2.VideoCapture(0)
    # capture.set(3, 320)
    # capture.set(4, 240)
    lst_points, data = get_points_webcam(reader)

    # only record points and do posture correction when enough key points are detected
    locked = False
    if (data[0] == -1).sum() <= 5: 
        record_points(db_path, lst_points)

        # posture correction
        
        wrong_position = head_shoulder(data)
        wrong_hand = hand_on_face(data, 30)
        if wrong_position or wrong_hand:
            playsound('./sound_effects/alarm.m4a')

        # lock
        last_n_records = get_last_n_positions(db_path, num_records=3)
        if last_n_records.count(1) > 3 * 0.5:
            locked = True
            logger.info('Sitting for too long.')
            pyautogui.hotkey('winleft', 'l')
            time.sleep(5)
    
    return locked

def unlock(reader: Reader) -> bool:
    # capture = cv2.VideoCapture(0)
    # capture.set(3, 320)
    # capture.set(4, 240)
    lst_points, data = get_points_webcam(reader)
    # decide whether to lock pc
    # when PC is locked, recording is paused
    if continuous_stretch(2, data, min_elbow, max_elbow, min_hip, max_hip, min_knee, max_knee):
        unlock_pc()
        return True
    return False


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--debug', action='store_true', help='Toggle debug mode.')
    args = parser.parse_args()

    num_records = 60

    body_estimation = Body('body_pose_model.pth')

    model = baseline(36, 20, 10, 8, 5, 2)
    model.load_state_dict(torch.load('detect_position/baseline.pt'))
    model.eval()

    col_names = ['timestamp', 'nose', 'neck', 'l_shoulder', 'l_elbow', 'l_hand', 'r_shoulder','r_elbow', 'r_hand', 'l_hip', 'l_knee', 'l_foot', 
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


    db_path = Path('daily_log.db')

    try:
        conn = sqlite3.connect(f'file:{db_path}?mode=rw', uri=True)
    except sqlite3.OperationalError:
        # Path exists but not a database.
        if db_path.exists():
            raise ValueError(f'{db_path} exists, but it is not not a valid database.')

        # Create a database here.
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


    # lst_points, data = get_points_webcam(capture)

    min_elbow, max_elbow, min_hip, max_hip, min_knee, max_knee = get_strech_metrics('./squat_img/')
    # print(continuous_stretch(2, data, min_elbow, max_elbow, min_hip, max_hip, min_knee, max_knee))

    cam_reader = VideoReader(debug=args.debug)
    if args.debug:
        unlock_reader = DebugReader('./squat_test')
    else:
        unlock_reader = cam_reader
# 
    # schedule.every(0.1).minutes.do(capture_and_openpose, reader=cam_reader)
    # schedule.every(0.1).minutes.do(lock_and_unlock, reader=unlock_reader)

    # while True:

        # schedule.run_pending()
        # time.sleep(1)

    while True:
        locked = capture_and_openpose(cam_reader)

        if locked:
            logger.info('Locked.')
            unlocked = unlock(unlock_reader)
            while not unlocked:
                time.sleep(1)
            logger.info('Unlocked.')

        time.sleep(1)
