import cv2
import matplotlib.pyplot as plt
import copy
import numpy as np
import torch

import time, threading
from datetime import datetime, timedelta
import argparse
import os
import schedule
from loguru import logger
from playsound import playsound
import pyautogui





from pathlib import Path
from collections import deque
import sqlite3

from op import model
from op import util
from op.body import Body
from op.hand import Hand


from detect_position.position_detection import baseline
from posture_correction import head_shoulder, hand_on_face
from record_key_points import process_candidate, record_points, get_points_webcam
from lock_unlock import lock_pc, unlock_pc, get_last_n_positions, continuous_stretch, get_strech_metrics



# every 30 seconds, run this script to 
# (1) run openpose, get key points coordinates
# (2) run the model to detect sitting/standing position
# (3) store the timestamp, key points coordinates, position indicators into a csv (one csv for each day)


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






def capture_and_openpose():
    print('start capturing')
    capture = cv2.VideoCapture(-1)
    capture.set(3, 320)
    capture.set(4, 240)
    lst_points, data = get_points_webcam(capture)

    # only record points and do posture correction when enough key points are detected
    if (data[0] == -1).sum() <= 5: 
        record_points(db_path, lst_points)

        # posture correction
        
        wrong_position = head_shoulder(data)
        wrong_hand = hand_on_face(data, 30)
        if wrong_position or wrong_hand:
            playsound('./sound_effects/alarm.m4a')
        
        # decide whether to lock pc
        # when PC is locked, recording is paused
        last_n_records = get_last_n_positions(db_path, num_records=3)
        print(last_n_records)
        if last_n_records.count(1) > 3 * 0.5:
            pyautogui.hotkey('winleft', 'l')
            time.sleep(10)
            unlock_pc()
    


#def unlock():


    # calculate cumulative standing time, to determine if need lock pc


if __name__ == '__main__':
    # capture = cv2.VideoCapture(-1)
    # capture.set(3, 320)
    # capture.set(4, 240)
    min_elbow, max_elbow, min_hip, max_hip, min_knee, max_knee = get_strech_metrics('./squat_img/')

    schedule.every(0.06).minutes.do(capture_and_openpose)

    while True:
        schedule.run_pending()
        time.sleep(1)



# if __name__ == '__main__':
    






# while True:
#     lst_points, data = get_points_webcam(capture)
#     print('______________')















































# # take a pic every 30 sec
# capture = cv2.VideoCapture(0)
# capture.set(3, 640)
# capture.set(4, 480)

# frame_set = []
# start_time = datetime.datetime.now()

# while True:
#     if time.time() - start_time = 30:
#         start_time = datetime.datetime.now()
#         ret, frame = capture.read()
        




#     ret, frame = capture.read()
#     cv2.imshow('frame', frame)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
#     if time.time() - start_time >= 30: #<---- Check if 60 sec passed
#         now = datetime.datetime.now()

#         img_name = datetime.datetime.strftime(now, '%Y-%m-%d-%H-%M-%S') + '.jpg'
#         img_name = dir_name / img_name
#         cv2.imwrite(str(img_name), frame)
#         print("{} written!".format(img_name))
#         start_time = time.time()













#from detect_sitting import *

# records = defaultdict()
# start_min = 0
# end_time = 30


# s = sched.scheduler(time.time, time.sleep)
# def do_something(sc): 
#     print("Doing stuff...")
#     # do your stuff
#     s.enter(60, 1, do_something, (sc,))

# s.enter(60, 1, do_something, (s,))
# s.run()