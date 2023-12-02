import cv2
import numpy as np
from time import sleep
import argparse

from environment import Environment, Parking1
from pathplanning import PathPlanning, ParkPathPlanning, interpolate_path
from bicycle_control import Car_Dynamics, MPC_Controller, Linear_MPC_Controller
from utils import angle_of_line, make_square, DataLogger

import matplotlib.pyplot as plt

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--x_start', type=int, default=0, help='X of start')
    parser.add_argument('--y_start', type=int, default=90, help='Y of start')
    parser.add_argument('--psi_start', type=int, default=0, help='psi of start')
    parser.add_argument('--x_end', type=int, default=90, help='X of end')
    parser.add_argument('--y_end', type=int, default=80, help='Y of end')
    parser.add_argument('--parking', type=int, default=1, help='park position in parking1 out of 24')

    args = parser.parse_args()
    logger = DataLogger()

    parking1 = Parking1(args.parking)
    end, obs = parking1.generate_obstacles()
    env = Environment(obstacles=obs)
    
    data_root = "result_analysis/"
    # Read from npy file
    target_path = np.load(data_root + 'final_path.npy')
    print(target_path)
    his_x = np.load(data_root + 'his_x.npy')
    his_y = np.load(data_root + 'his_y.npy')
    his_psi = np.load(data_root + 'his_psi.npy')
    ref_x = np.load(data_root + 'ref_x.npy')
    ref_y = np.load(data_root + 'ref_y.npy')
    ref_psi = np.load(data_root + 'ref_psi.npy')

    # Combine his_x and his_y to get the path
    color_blue = np.array([255,0, 0])
    color_red = np.array([0,0, 255])
    his_path = np.vstack([his_x, his_y]).T
    env.draw_path(target_path, color_red)
    res = env.render(his_x[0], his_y[0], his_psi[0], 0)
    cv2.imshow('environment', res)
    cv2.imwrite(data_root + "target_path.png", res * 255)
    key = cv2.waitKey(1)
    
    env.draw_path(his_path, color_blue)
    env.draw_path(target_path, color_red)
    res = env.render(his_x[-1], his_y[-1], his_psi[-1], 0)
    cv2.imshow('environment', res)
    cv2.imwrite(data_root + "target_and_real_path.png", res * 255)
    key = cv2.waitKey(1)

    # Plot the ref_x vs index and hit_x vs index

    plt.figure()
    plt.plot(ref_x, label="reference_x")
    plt.plot(his_x, label="real_x")
    plt.title("Comparison of Reference and Real x")
    plt.legend()
    plt.savefig(data_root + "x.png")

    # Plot the ref_y vs index and hit_y vs index

    plt.figure()
    plt.plot(ref_y, label="reference_y")
    plt.plot(his_y, label="real_y")
    plt.title("Comparison of Reference and Real y")
    plt.legend()
    plt.savefig(data_root + "y.png")

    # Plot the ref_psi vs index and hit_psi vs index

    plt.figure()
    plt.plot(ref_psi, label="reference_psi")
    plt.plot(his_psi, label="real_psi")
    plt.title("Comparison of Reference and Real psi")
    plt.legend()
    plt.savefig(data_root + "psi.png")

    key = cv2.waitKey()

    size = (700, 700)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    videowrite = cv2.VideoWriter(data_root + 'mpc.mp4', fourcc, 20, size)

    env_video = Environment(obstacles=obs)
    for i in range(len(his_x)):
        env_video.draw_path(target_path, color_red)
        res = env_video.render(his_x[i], his_y[i], his_psi[i], 0)
        videowrite.write((res*255).astype(np.uint8))

    
    videowrite.write((res*255).astype(np.uint8))
    videowrite.release()



    #############################################################################################

    cv2.destroyAllWindows()

