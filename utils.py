import datetime
from collections import defaultdict
import numpy as np

import math


# determine whether the person is in sitting position: if any of the following conditions is met
# 1. if the x distance between hip and knee significantly greater than the x distance between knee and foot or the y distance btw foot and hip signifiantly larger
# 2. if the y distance between foot and hip less than the distance between hip and head
# 3. if foot or knee undetected, check if hand-elbow-sholder approx 90 degree

def get_img_name() -> str:
    now = datetime.datetime.now()
    img_name = datetime.datetime.strftime(now, '%Y-%m-%d-%H-%M-%S') + '.jpg'
    return img_name


#################### combine the three?
def leg_detected(candidate, hip, knee, foot):
    # returns whether a whole leg is detected (hip + knee + foot), either left leg or right leg
    if hip in candidate and knee in candidate and foot in candidate:
        return True
    else:
        return False

def lower_leg_detected(candidate, hip, knee, foot):
    # returns true if hip not detected, but knee and foot both detected
    if hip not in candidate:
        if knee in candidate and foot in candidate:
            return True
    return False

def upper_leg_detected(candidate, hip, knee, foot):
    # returns true if foot not detected but 
    if foot not in candidate:
        if hip in candidate and knee in candidate:
            return True
    return False

def lower_arm_detected(candidate, elbow, wrist):
    if elbow in candidate and wrist in candidate:
        return True
    return False



    





    return angle 


def line_angle(candidate, index1, index2):
    # calculate the angle of the line segment created by index1 and index2, relative to the x axis
    vec1 = [candidate[index1][0] - candidate[index2][0], candidate[index1][1] - candidate[index2][1]]
    norm1 = np.linalg.norm(vec1)

    vec2 = [1, 0]
    norm2 = np.linalg.norm(vec2)

    cos = np.dot(vec1, vec2) / (norm1 * norm2)
    angle = np.arccos(cos) * 180 / math.pi

    return angle 



def hip_knee_foot_sit(candidate, hip, knee, foot):
    # use this function only when leg_detected returns True
    # if the angle of hip-knee-foot less than 150 degree, sit=True
    # else, if foot y position same or higher than hip position, sit=True (this is for the wierd and unhealthy sitting position of feet on desk)
    # calcualte the tan of the foot-hip, if angle between -20 and 20 degree, returns True
    knee_angle = vertex_angle(candidate, hip, knee, foot)
    if knee_angle < 150:
        print(knee_angle)
        return True # if leg is not straight, then sit
    else: # leg is straight, but foot on desk or bed, hence the angle of hip-knee to the floor is small (-30 degree to 30 degree)
        angle = line_angle(candidate, hip, knee)
        if angle < 30 or angle > 150:
            return True
    return False
    
def upper_leg_sit(candidate, hip, knee):
    angle = line_angle(candidate, hip, knee)
    if angle < 30:
        return True
    return False

def lower_leg_sit(candidate, knee, foot):  # lower leg is a hacky way, assuming vertical postion while standing
    # using 70 degree as of now
    angle = line_angle(candidate, knee, foot)
    if angle < 70 or angle > 110:
        return True
    return False
    
         
def record_video():
    pass




num2name = defaultdict()
num_points = list(range(18))
name_points = ['nose', 'neck', 'lshoulder', 'lelbow', 'lhand', 'rshoulder',
                'relbow', 'rhand', 'lhip', 'lknee', 'lfoot', 'rhip', 'rknee',
                'rfoot', 'leye', 'reye', 'lear', 'rear']
for i in range(18):
    num2name[num_points[i]] = name_points[i]

# convert candidate (openpose result, [x, y, confidence, body part index]) into a dictionary
def dict_index(candidate):
    ans = defaultdict()
    for row in candidate: # candidate is a list of list
        key = int(row[-1])
        x = row[0]
        y = row[1]
        ans[key] = (x, y)
    return ans




# def convert2polar(data):
#     """
#     use neck as the origin, convert all other points to polar coordinates
#     ignoring ears/eyes/nose, i.e. use data[1:14]
#     """
#     neck_x = data[0, 1]
#     neck_y = data[1, 1]
    
#     relative_data = np.zeros((2, 18))
#     for col in range(18):
#         relative_data[0, col] = data[0, col] - neck_x
#         relative_data[1, col] = neck_y - data[1, col] 

#     # only need to keep the [1:14], now neck is the first 
#     relative_data = relative_data[:, 1:14]

#     # array of polar coordinates: array[0, :] is theta, array[1, :] is r
#     polar_data = np.zeros((2, 13))
#     for i in range(1, 13): # excluding neck, otherwise it produces 0 in the denominator
#         cart_x = relative_data[0, i]
#         cart_y = relative_data[1, i]
#         polar_data[0, i] = cart_x / (math.sqrt(cart_x ** 2 + cart_y ** 2))
#         polar_data[1, i] = math.sqrt(cart_x ** 2 + cart_y ** 2)
    
#     polar_data = polar_data[:, 1:13]
#     return polar_data


# def squat_or_not(curr_data_raw, polar_sit, polar_stand, polar_squat):
#     polar_curr = convert2polar(curr_data_raw)[0, :]
#     dist2stand = distance.euclidean(polar_curr, polar_stand[0, :])
#     dist2sit = distance.euclidean(polar_curr, polar_sit[0, :])
#     dist2squat = distance.euclidean(polar_curr, polar_squat[0, :])
#     actions = ['stand', 'sit', 'squat']
#     dists = [dist2stand, dist2sit, dist2squat]
#     closest_action = actions[dists.index(min(dists))]
#     return closest_action




# create the csv file to store today's data
# if not os.path.isdir('./daily_data'):
#     os.mkdir('./daily_data')

# date = datetime.today().strftime('%Y-%m-%d')
# fname = date + '.csv'
# col_names = ['timestamp', 'nose', 'neck', 'l_shoulder', 'l_elbow', 'l_hand', 'r_shoulder','r_elbow', 'r_hand', 'l_hip', 'l_knee', 'l_foot', 
#             'r_hip', 'r_knee', 'r_foot', 'l_eye', 'r_eye', 'l_ear', 'r_ear', 'position']

# fpath = './daily_data/' + fname
# if not os.path.exists(fpath):
#     with open(fpath, 'w') as fout:
#         csvwriter = csv.writer(fout)
#         csvwriter.writerow(col_names)