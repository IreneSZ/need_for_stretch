import cv2
import argparse
import os
import time
from pathlib import Path
from utils import get_img_name


dir_name = Path('./data/unlabeled_data/')
# if not os.path.exists(dir_name):
#     os.mkdir(dir_name)

dir_name.mkdir(parents=True,  exist_ok=True)




capture = cv2.VideoCapture(-1)
capture.set(3, 640)
capture.set(4, 480)
#img_counter = 0
frame_set = []
start_time = time.time()

while True:
    ret, frame = capture.read()
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    if time.time() - start_time >= 2: #<---- Check if 60 sec passed
        img_name = dir_name / get_img_name()
        cv2.imwrite(str(img_name), frame)
        print("{} written!".format(img_name))
        start_time = time.time()
