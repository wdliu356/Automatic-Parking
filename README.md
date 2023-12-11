# Automatic Parking

This repository contains a Python implementation of an automatic parking system including Car Dynamics & Kinematics Model, Linear/non-Linear MPC Controller, and Hybrid A* path planning. 
 The agent navigates its route through the environment and is directed to the assigned park location. This repo is built and extended based on original repository from Pandas-Team: ```https://github.com/Pandas-Team/Automatic-Parking```. 

## Envroinment
conda environment requirements are listed in ```requirements.txt```. 
An extra dependency for the Reeds-Shepp curve in Hybrid A* can be found in the repository: ```https://github.com/zhm-real/CurvesGenerator```

## Parallel Parking
### Running command:
```
$ python ours_control_test.py --x_start 0 --y_start 90 --psi_start 0 --parking 7
```
### Input Argument
--x_start : x start position 
--y_start : y start position
--psi_start : start car orientation
--x_end : goal x position
--y_end : goal y position
--parking : park position in parking1 out of 24


![automatic_parking_process](https://user-images.githubusercontent.com/56114938/128083454-60f8ba82-00a8-43a2-b8ad-8d4ad09cc762.gif)

## Path Planning
#### Hybrid A* Algorithm
Hybrid A* is implemented to find a path from the start to the park location, extending from regular A* which allows the robot to navigate in continuous state spaces where the movement of a robot is not limited to a grid but can take any position in a continuous space. Therefore, based on our car kinematic model, the resultant planned path has x, y, and yaw so the MPC controller follows as a reference. 

## Path Tracking
**The kinematic model** of the car is:
```math
\left\{\begin{matrix}
\dot{x} = v . cos(ψ+\beta)\\
\dot{y} = v . sin(ψ+\beta)\\
\dot{v} = a\\
\dot{ψ} = v . sin(\beta)/L_r
\beta = \arctan(\frac{L_r}{L_r + L_f} \arctan \delta
\end{matrix}\right.
```
```a: acceleration, δ: steering angle, ψ: yaw angle, L: wheelbase, x: x-position, y: y-position, v: velocity```

**The state vector** is:
```math
z=[x,y,v,ψ]
```
```x: x-position, y: y-position, v: velocity, ψ: yaw angle```

**The input vector** is:
```math
u=[a,δ]
```
```a: acceleration, δ: steering angle```

#### Control
The MPC controller controls vehicle speed and steering based on the model and the car is directed through the path. There is an option for using the linearized model for MPC. Our linearized MPC controller is tuned for small angles and slow turning. 

