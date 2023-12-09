import cv2
import numpy as np
from time import sleep
import argparse

from environment import Environment, Parking1
from pathplanning_h_astar import PathPlanning, interpolate_path
from bicycle_control import Car_Dynamics, MPC_Controller, Linear_MPC_Controller
from utils import angle_of_line, make_square, DataLogger

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

    ########################## default variables ################################################
    start = np.array([args.x_start, args.y_start])
    end   = np.array([args.x_end, args.y_end])
    #############################################################################################

    # environment margin  : 5
    # pathplanning margin : 5

    ########################## defining obstacles ###############################################
    parking1 = Parking1(args.parking)
    end, obs = parking1.generate_obstacles()
    # print("end: ", end)
    # obs = None

    # add squares
    # square1 = make_square(10,65,20)
    # square2 = make_square(15,30,20)
    # square3 = make_square(50,50,10)
    # obs = np.vstack([obs,square1,square2,square3])

    # Rahneshan logo
    # start = np.array([50,5])
    # end = np.array([35,67])
    # rah = np.flip(cv2.imread('READ_ME/rahneshan_obstacle.png',0), axis=0)
    # obs = np.vstack([np.where(rah<100)[1],np.where(rah<100)[0]]).T

    # new_obs = np.array([[78,78],[79,79],[78,79]])
    # obs = np.vstack([obs,new_obs])
    #############################################################################################

    ########################### initialization ##################################################
    # env = Environment(obstacles=None)
    env = Environment(obstacles=obs)
    # path = np.array([[20.5, 20.5],
    #                  [10.5, 40.5],
    #                  [30.5, 60.5],
    #                  [20.5, 80.5],
    #                  [50.5, 80.5]])
    
    # start = path[0]
    # #append 20 goal state to the end of path
    # end = path[-1]
    # for i in range(5):
    #     path = np.vstack([path, end])


    
    MPC_HORIZON = 5
    controller = MPC_Controller()
    # controller = Linear_MPC_Controller()

    
    #############################################################################################

    ############################# path planning #################################################
    # park_path_planner = ParkPathPlanning(obs)
    path_planner = PathPlanning(obs)

    # print('planning park scenario ...')
    # new_end, park_path, ensure_path1, ensure_path2 = park_path_planner.generate_park_scenario(int(start[0]),int(start[1]),int(end[0]),int(end[1]))
    
    print('routing to destination ...')
    path = path_planner.plan_path(start[0],start[1],0,0,41,32,np.pi/2,0,1)
    # path = path[:,0:2]
    # path = np.vstack([path, ensure_path1])
    # end = path[-1]
    # for i in range(5):
    #     path = np.vstack([path, end])

    interpolated_path = path


    # print('interpolating ...')
    # interpolated_path = interpolate_path(path, sample_rate = 3)
    # interpolated_path = np.hstack([interpolated_path, np.zeros((len(interpolated_path),1))])
    # # Add a column at the end of the path for the car's orientation
    # # Set the orientation such that the current orientation heads toward next point
    # def angle_of_line(point1, point2):
    #     return np.arctan2(point2[1]-point1[1], point2[0]-point1[0])
    
    # for i in range(len(interpolated_path)-1):
    #     interpolated_path[i,2] = angle_of_line(interpolated_path[i], interpolated_path[i+1])
    
    # # print("interpolated_path: ", interpolated_path)
    # # Set the orientation of the last point to be the same as the second last point
    # interpolated_path[-1,2] = interpolated_path[-2,2]
    final_path = interpolated_path
    print("final_point:", final_path[-1,:])
    # # print("final_path: ", final_path)

    # Save the final_path to a npy
    # np.save('final_path.npy', final_path)
    # for i in range(final_path.shape[0]):
    #     print("final_path x: ", final_path[i][0])
    #     print("final_path y: ", final_path[i][1])
    #     print("final_path psi: ", final_path[i][2])

    env.draw_path(final_path)
    # env.draw_path(interpolated_park_path)

    # final_path = np.vstack([interpolated_path, interpolated_park_path, ensure_path2])

    #############################################################################################

    ################################## control ##################################################
    my_car = Car_Dynamics(start[0], start[1], 0, 0, length=4, dt=0.2, a=1.14)
    res = env.render(my_car.x, my_car.y, my_car.psi, 0)
    cv2.imshow('environment', res)
    key = cv2.waitKey(1)

    print('driving to destination ...')
    acc_path_arr = list()
    delta_path_arr = list()

    ref_x = []
    ref_y = []
    ref_psi = []

    his_x = []
    his_y = []
    his_psi = []
    print("final path shape: ", final_path.shape)
    while len(final_path) > 0:
        point = final_path[0]
        acc, delta = controller.optimize(my_car, final_path[0:MPC_HORIZON])
        # print("acc: ", acc)
        # print("delta: ", delta)
        my_car.update_state(my_car.move(acc,  delta))
        acc_path_arr.append(acc)
        delta_path_arr.append(delta)
        res = env.render(my_car.x, my_car.y, my_car.psi, delta)

        # Save history to array
        his_x.append(my_car.x)
        his_y.append(my_car.y)
        his_psi.append(my_car.psi)

        ref_x.append(point[0])
        ref_y.append(point[1])
        ref_psi.append(point[2])

        logger.log(point, my_car, acc, delta)
        cv2.imshow('environment', res)
        key = cv2.waitKey(1)
        if key == ord('s'):
            cv2.imwrite('res.png', res*255)
        if my_car.goal_state_reached(point):
            final_path = final_path[1:]
            # print("final_path: ", final_path)

    # Save history and ref to npy
    # np.save('his_x.npy', his_x)
    # np.save('his_y.npy', his_y)
    # np.save('his_psi.npy', his_psi)
    # np.save('ref_x.npy', ref_x)
    # np.save('ref_y.npy', ref_y)
    # np.save('ref_psi.npy', ref_psi)

    # for i,point in enumerate(final_path):
        
    #         acc, delta = controller.optimize(my_car, final_path[i:i+MPC_HORIZON])
    #         # print("acc: ", acc)
    #         # print("delta: ", delta)
    #         my_car.update_state(my_car.move(acc,  delta))
    #         acc_path_arr.append(acc)
    #         delta_path_arr.append(delta)
    #         res = env.render(my_car.x, my_car.y, my_car.psi, delta)
    #         logger.log(point, my_car, acc, delta)
    #         cv2.imshow('environment', res)
    #         key = cv2.waitKey(1)
    #         if key == ord('s'):
    #             cv2.imwrite('res.png', res*255)

    print("acc_path_arr: ", acc_path_arr)
    print("delta_path_arr: ", delta_path_arr)

    print("len of acc_path_arr: ", len(acc_path_arr))
    print("len of delta_path_arr: ", len(delta_path_arr))
    # zeroing car steer
    res = env.render(my_car.x, my_car.y, my_car.psi, 0)
    logger.save_data()
    cv2.imshow('environment', res)
    key = cv2.waitKey()
    #############################################################################################

    cv2.destroyAllWindows()

