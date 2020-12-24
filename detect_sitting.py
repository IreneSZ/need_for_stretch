import cv2
import matplotlib.pyplot as plt
import copy
import numpy as np
import torch

import timeit

from op import model
from op import util
from op.body import Body
from op.hand import Hand


from utils import *


# if there is no person detected (aka no body parts detected), count as stand


def detect_sit(candidate):

    ############### try whole legs #######################################################################
    if leg_detected(candidate, 8, 9, 10):
        left_leg = hip_knee_foot_sit(candidate, 8, 9, 10)
    else:
        left_leg = None
    if leg_detected(candidate, 11, 12, 13):
        right_leg = hip_knee_foot_sit(candidate, 11, 12, 13)
    else:
        right_leg = None
    
    if left_leg or right_leg:
        print('whole leg')
        return True     ####################### sitting position detected using the whole leg

    
    ########################### try upper legs ###################################
    if upper_leg_detected(candidate, 8, 9, 10):
        left_upper_leg = upper_leg_sit(candidate, 8, 9)

    else:
        left_upper_leg = None
    if upper_leg_detected(candidate, 11, 12, 13):
        right_upper_leg = upper_leg_sit(candidate, 11, 12)
    else:
        right_upper_leg = None

    if left_upper_leg or right_upper_leg: 
        print('upper leg')
        return True

    ############################## try lower legs ##########################

    if lower_leg_detected(candidate, 8, 9, 10):
        left_lower_leg = lower_leg_sit(candidate, 9, 10)
    else:
        left_lower_leg = None
    if lower_leg_detected(candidate, 11, 12, 13):
        right_lower_leg = lower_leg_sit(candidate, 12, 13)
    else:
        right_lower_leg = None
    
    if left_lower_leg or right_lower_leg:
        print('lower leg')
        return True

    # ############ if no whole, upper, or lower leg is detected, check lower arm #########
    # if lower_arm_detected(candidate, 3, 4):
    #     left_lower_arm = lower_arm_sit(candidate, 3, 4)
    # else:
    #     left_lower_arm = None
    # if lower_arm_detected(candidate, 6, 7):
    #     right_lower_arm = lower_arm_sit(candidate, 6, 7)
    # else:
    #     right_lower_arm = None
    
    # if left_lower_arm or right_lower_arm:
    #     return True

    ############## if no leg is detected, raise error, else, detects "stand" ##################
    all_segments = [left_leg, right_leg, left_upper_leg, right_upper_leg, left_lower_leg, right_lower_leg]
    if all(x is None for x in all_segments):
        return None
    else:
        return False





body_estimation = Body('body_pose_model.pth')

print(f"Torch device: {torch.cuda.get_device_name()}")





def detect_once():
    cap = cv2.VideoCapture(0)
    cap.set(3, 640)
    cap.set(4, 480)
    while True:
        ret, oriImg = cap.read()
        ### start timer
        start = timeit.default_timer()

        candidate, subset = body_estimation(oriImg)
        canvas = copy.deepcopy(oriImg)
        canvas = util.draw_bodypose(canvas, candidate, subset)

        candidate = dict_index(candidate)

        if candidate:
            position = detect_sit(candidate)
            if position is True:
                print('person is sitting')
            elif position is False:
                print('person is NOT sitting')
            else:
                raise RuntimeError('adjust camera')



        if not candidate:
            print('no person detected')
        ### end timer
        stop = timeit.default_timer()
        print('inference time :', stop - start)


        cv2.imshow('detect', canvas)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

detect_once()




# candidate1 = defaultdict()
# candidate1[8] = (50, 50)
# candidate1[9] = (50, 100)
# #candidate1[10] = (130, 100)

# candidate2 = defaultdict()
# candidate2[8] = (50, 50)




# print(detect_sit(candidate2))

    
    