import cv2
import numpy as np
from time import sleep
import argparse

from environment import Environment, Parking1
from pathplanning import PathPlanning, ParkPathPlanning
from bicycle_control import Car_Dynamics
from utils import angle_of_line, DataLogger


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--x_start', type=int, default=0, help='X of start')
    parser.add_argument('--y_start', type=int, default=90, help='Y of start')

    args = parser.parse_args()
    logger = DataLogger()

    ########################## default variables ################################################

    start = np.array([args.x_start, args.y_start])

    #########################################################################################

    ########################### initialization ##############################################
    env = Environment(obstacles=None)
    my_car = Car_Dynamics(start[0], start[1], 3.0, 0, 0, length=2.54, dt=0.17, a=1.14)

    res = env.render(my_car.x, my_car.y, my_car.psi, 0)
    cv2.imshow('environment', res)
    key = cv2.waitKey(1)
    #############################################################################################

    ################################## open loop command array ##################################################
    # acc_arr = np.random.rand(100) # TODO: check the possible value of acc
    acc_arr = np.ones(200) * 0.3 # TODO: check the possible value of acc
    acc_arr[50:] = -0.1
    acc_arr[-1] = 0.0
    # delta_arr = np.random.rand(100)
    # delta_arr[-1] = 0.0
    delta_dot_arr = np.zeros_like(acc_arr)
    delta_dot_arr[:50] = 0.0174
    delta_dot_arr[50:75] = -0.0174 * 3
    delta_dot_arr[75:] = 0.0
    command_len = int(np.prod(acc_arr.shape))
    #############################################################################################

    ################################## open loop control ##################################################

    # x = t
    # y = 50

    size = (700, 700)
    # fourcc = cv2.VideoWriter_fourcc(*'XVID')
    # fourcc = cv2.VideoWriter_fourcc(*'MJPG') # Encoder form!!
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    # fourcc = cv2.VideoWriter_fourcc(*'mpv4')
    # fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    # fourcc = cv2.VideoWriter_fourcc('U', '2', '6', '3')
    videowrite = cv2.VideoWriter('/home/zlj/Documents/ROB599-Autonomous Vehicles/project/Automatic-Parking/CAR kinematic model/log results/cartoon.mp4', fourcc, 20, size)

    print('driving in random path ...')
    for i in range(command_len):
        acc = acc_arr[i]
        delta_dot = delta_dot_arr[i]
        my_car.update_state(my_car.move(acc,  delta_dot))
        res = env.render(my_car.x, my_car.y, my_car.psi, my_car.delta)
        point = np.array([my_car.x, my_car.y]) # TODO: check the point definition
        logger.log(point, my_car, acc, my_car.delta)
        cv2.imshow('environment', res)
        key = cv2.waitKey(1)
        videowrite.write((res*255).astype(np.uint8))
        if key == ord('s'):
            cv2.imwrite('res.png', res*255)

    # zeroing car steer
    res = env.render(my_car.x, my_car.y, my_car.psi, 0)
    # cv2.imwrite('res.png', res*255)
    logger.save_data()
    cv2.imshow('environment', res)
    key = cv2.waitKey(1)
    videowrite.write((res*255).astype(np.uint8))
    videowrite.release()

    sleep(10)

    #############################################################################################

    cv2.destroyAllWindows()

