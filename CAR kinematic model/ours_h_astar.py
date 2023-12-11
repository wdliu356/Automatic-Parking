import numpy as np
import math
import scipy.interpolate as scipy_interpolate
from utils import angle_of_line
from queue import PriorityQueue
import os
from bicycle_control import *
from heapdict import heapdict
import scipy.spatial.kdtree as kd
import CurvesGenerator.reeds_shepp as rsCurve
import heapq
import matplotlib.pyplot as plt

class Cost:
    reverse = 10
    directionChange = 10
    steerAngle = 1
    steerAngleChange = 1
    hybridCost = 50

class HolonomicNode:
    def __init__(self, gridIndex, cost, parentIndex):
        self.gridIndex = gridIndex
        self.cost = cost
        self.parent = parentIndex


class Hybrid_AStarPlanner:

    def __init__(self, ox, oy, resolution, rr, length, a):
        """
        Initialize grid map for a star planning

        ox: x position list of Obstacles [m]
        oy: y position list of Obstacles [m]
        resolution: grid resolution [m]
        rr: robot radius[m]
        """

        self.resolution = resolution
        self.yaw_resolution = np.deg2rad(5.0)
        self.rr = rr
        self.min_x, self.min_y = 0, 0
        self.max_x, self.max_y = 0, 0
        self.obstacle_map = None
        self.x_width, self.y_width = 0, 0
        self.motion = self.get_motion_model()
        self.calc_obstacle_map(ox, oy)
        self.length = length
        self.a = a
        self.x_res = 1
        self.y_res = 1
        self.psi_res = 0.08*np.pi
        self.v_res = 0.1
        self.maxSteerAngle = np.deg2rad(60)


    class Node:
        def __init__(self, x, y, psi, v, cost, parent, length, a, traj, gridIndex):
            self.x = x  # x position, can be used to calculate its position in the grid map
            self.y = y  # y position, can be used to calculate its position in the grid map
            self.psi = psi          # steering angle
            self.v = v
            self.cost = cost
            self.parent= parent
            self.length = length
            self.a = a
            self.traj = traj                   # trajectory x, y  of a simulated node
            self.gridIndex = gridIndex         # grid block x, y, yaw index
            

        def __str__(self):
            return str(self.x) + "," + str(self.y) + "," + str(
                self.cost) + "," + str(self.parent)
        
        # def update(self,dt,acc,delta):
        #     b = self.length - self.a
        #     beta = np.arctan((b * np.tan(delta)) / (self.length))
        #     x_dot = self.v*np.cos(self.psi + beta)
        #     y_dot = self.v*np.sin(self.psi + beta)
        #     v_dot = acc
        #     psi_dot = self.v*np.tan(delta)*np.cos(beta)/self.length
        #     x = self.x + dt*x_dot
        #     y = self.y + dt*y_dot
        #     psi = self.psi + dt*psi_dot
        #     if psi > np.pi:
        #         psi = psi - 2*np.pi
        #     elif psi < -np.pi:
        #         psi = psi + 2*np.pi
        #     v = self.v + dt*v_dot
        #     if v > 2:
        #         v = 2
        #     elif v < -2:
        #         v = -2
        #     if v == 2 or v == -2:
        #         acc = 0
        #         v_dot = 0
        #         #  + 0.2*psi_dot**2 + 0.1*v_dot**2
        #         #  + 0.1*delta**2 + 0.1*acc**2
        #     # action cost
        #     # + np.sqrt(dt**2 * (0.5*x_dot**2 + 0.5*y_dot**2))
        #     new_traj = 
        #     action_cost = self.simulatedPathCost([delta,acc], dt)
        #     return [x,y,psi,v,action_cost,self.length,self.a]
        
        def calc_new_state(self, curr, dt, acc, delta):
            x_0, y_0, psi_0 = curr[0], curr[1], curr[2]
            b = self.length - self.a
            v_dot = acc
            beta = np.arctan((b * np.tan(delta)) / (self.length))
            x_dot = self.v*np.cos(psi_0 + beta)
            y_dot = self.v*np.sin(psi_0 + beta)
            psi_dot = self.v*np.tan(delta)*np.cos(beta)/self.length
            x = x_0 + dt*x_dot
            y = y_0 + dt*y_dot
            psi = psi_0 + dt*psi_dot
            if psi > np.pi:
                psi = psi - 2*np.pi
            elif psi < -np.pi:
                psi = psi + 2*np.pi
            v = self.v + dt*v_dot
            if v > 2:
                v = 2
            elif v < -2:
                v = -2
            if v == 2 or v == -2:
                acc = 0
                v_dot = 0
            new_wp = [] 
            new_wp.append(x)
            new_wp.append(y)
            new_wp.append(psi)
            return new_wp, v

    def hybrid_update(self, curr_node, dt, acc, delta, sim_length = 4):
        # Simulate node using given current Node and Motion Commands

        traj = []
        traj_new_start, _ = curr_node.calc_new_state(curr_node.traj[-1], dt, acc, delta)
        traj.append(traj_new_start)
        # angle = rsCurve.pi_2_pi(currentNode.traj[-1][2] + motionCommand[1] * step / Car.wheelBase * math.tan(motionCommand[0]))
        # traj.append([currentNode.traj[-1][0] + motionCommand[1] * step * math.cos(angle),
        #             currentNode.traj[-1][1] + motionCommand[1] * step * math.sin(angle),
        #             rsCurve.pi_2_pi(angle + motionCommand[1] * step / Car.wheelBase * math.tan(motionCommand[0]))])
        for i in range(int((sim_length/dt))-1):
            new_state,v = curr_node.calc_new_state(traj[i], dt, acc, delta)
            traj.append(new_state)

        # Find grid index
        gridIndex = [round(traj[-1][0]/self.resolution), \
                    round(traj[-1][1]/self.resolution), \
                    round(traj[-1][2]/self.yaw_resolution), \
                    v/0.1]

        # Check if node is valid
        if not self.verify_xy(traj):
            return None

        # Calculate Cost of the node
        cost = self.simulatedPathCost(curr_node, [delta,acc], sim_length)
        return self.Node(traj[-1][0],traj[-1][1], traj[-1][2], v, cost, self.reg_index(curr_node), curr_node.length, curr_node.a, traj, gridIndex)

    def simulatedPathCost(self, curr_node, motionCommand, simulationLength):
        
        # Previos Node Cost
        cost = curr_node.cost

        # Distance cost
        if motionCommand[1] == 1:
            distance_cost = simulationLength
            cost += distance_cost 
        else:
            distance_cost = simulationLength
            cost += distance_cost * Cost.reverse  # Cost for reversing

        direction_cost = 0
        # Direction change cost
        if np.sign(curr_node.v) != np.sign(motionCommand[1]):
            direction_cost = Cost.directionChange
            cost += direction_cost

        # Steering Angle Cost
        steering_cost = abs(motionCommand[0]) * Cost.steerAngle
        cost += steering_cost

        # Steering Angle change cost
        steering_delta_cost = abs(motionCommand[0] - curr_node.psi) * Cost.steerAngleChange
        cost += steering_delta_cost
        # file_path = "/home/lidonghao/rob599proj/Automatic-Parking/log/action_cost.txt"

        # with open(file_path, "a") as file:
        #     file.write("Distance Cost: " + str(distance_cost) + ", Direction Cost: " + str(direction_cost) + ", Steering Cost: " + str(steering_cost) + ", Steering Delta Cost: " + str(steering_delta_cost) + "\n")

        return cost
    def state2idx(self, state):
        # 4 dimensional grid x, y , psi, v
        index = [round(state[0] / self.resolution), \
                  round(state[1] / self.resolution), \
                  round(state[2]/ self.yaw_resolution), \
                    round(state[3]/0.1)]
        return index

    def reedsSheppNode(self, currentNode, goalNode):

        # Get x, y, yaw of currentNode and goalNode
        startX, startY, startYaw = currentNode.traj[-1][0], currentNode.traj[-1][1], currentNode.traj[-1][2]
        goalX, goalY, goalYaw = goalNode.traj[-1][0], goalNode.traj[-1][1], goalNode.traj[-1][2]

        # Instantaneous Radius of Curvature
        radius = math.tan(np.deg2rad(60))/3.5   # wheel base 3.5m

        #  Find all possible reeds-shepp paths between current and goal node
        reedsSheppPaths = rsCurve.calc_all_paths(startX, startY, startYaw, goalX, goalY, goalYaw, radius, 1)

        # Check if reedsSheppPaths is empty
        if not reedsSheppPaths:
            return None

        # Find path with lowest cost considering non-holonomic constraints
        costQueue = heapdict()
        for path in reedsSheppPaths:
            costQueue[path] = self.reedsSheppCost(currentNode, path)

        # Find first path in priority queue that is collision free
        while len(costQueue)!=0:
            path = costQueue.popitem()[0]
            traj=[]
            traj = [[path.x[k],path.y[k],path.yaw[k]] for k in range(len(path.x))]
            # TODO: Need to refine
            if self.verify_xy(traj):
                cost = self.reedsSheppCost(currentNode, path)
                # return self.Node(goalNode.gridIndex ,traj, None, None, cost, index(currentNode))
                return self.Node(goalX, goalY, goalYaw, goalNode.v, cost, self.reg_index(currentNode), self.length, self.a, traj, goalNode.gridIndex)
                
        return None


    def reedsSheppCost(self, currentNode, path):

        # Previos Node Cost
        cost = currentNode.cost

        # Distance cost
        for i in path.lengths:
            if i >= 0:
                cost += 1
            else:
                cost += abs(i) * Cost.reverse

        # Direction change cost
        for i in range(len(path.lengths)-1):
            if path.lengths[i] * path.lengths[i+1] < 0:
                cost += Cost.directionChange

        # Steering Angle Cost
        for i in path.ctypes:
            # Check types which are not straight line
            if i!="S":
                cost += self.maxSteerAngle * Cost.steerAngle

        # Steering Angle change cost
        turnAngle=[0.0 for _ in range(len(path.ctypes))]
        for i in range(len(path.ctypes)):
            if path.ctypes[i] == "R":
                turnAngle[i] = - self.maxSteerAngle
            if path.ctypes[i] == "WB":
                turnAngle[i] = self.maxSteerAngle

        for i in range(len(path.lengths)-1):
            cost += abs(turnAngle[i+1] - turnAngle[i]) * Cost.steerAngleChange

        return cost

    def planning(self, sx, sy, spsi, sv, gx, gy, gpsi, gv,dt):
        """
        A star path search

        input:
            s_x: start x position [m]
            s_y: start y position [m]
            gx: goal x position [m]
            gy: goal y position [m]

        output:
            rx: x position list of the final path
            ry: y position list of the final path
        """
        s = [sx, sy, spsi, 0]
        g = [gx, gy, gpsi, 0]
        sGridIndex = self.state2idx(s)
        gGridIndex = self.state2idx(g)
        
        start_node = self.Node(sx, sy, spsi, sv, 0.0, tuple(sGridIndex), self.length, self.a, [s], sGridIndex)
        goal_node = self.Node(gx, gy, gpsi, gv, 0.0, tuple(gGridIndex), self.length, self.a, [g], gGridIndex)
        # Find Holonomic Heuristric
        holonomicHeuristics = self.holonomicCostsWithObstacles(goal_node)

        # Add start node to open Set
        openSet = {self.reg_index(start_node):start_node}
        closedSet = {}
        # Create a priority queue for acquiring nodes based on their cost's
        costQueue = heapdict()

        costQueue[self.reg_index(start_node)] = max(start_node.cost, Cost.hybridCost * holonomicHeuristics[start_node.gridIndex[0]][start_node.gridIndex[1]])

        counter = 0

        # Run loop while path is found or open set is empty
        while True:
            counter +=1
            # Check if openSet is empty, if empty no solution available
            if not openSet:
                return None

            # Get first node in the priority queue
            currentNodeIndex = costQueue.popitem()[0]
            currentNode = openSet[currentNodeIndex]

            # Revove currentNode from openSet and add it to closedSet
            openSet.pop(currentNodeIndex)
            # path = "/home/lidonghao/rob599proj/Automatic-Parking/log/output.txt"
            # with open(path, "a") as file:
            #     file.write("current x: " + str(currentNode.x) + ", current y: " + str(currentNode.y)+ "\n")

            closedSet[currentNodeIndex] = currentNode


            # Get Reed-Shepp Node if available
            rSNode = self.reedsSheppNode(currentNode, goal_node)

            # Id Reeds-Shepp Path is found exit
            if rSNode:
                closedSet[self.reg_index(rSNode)] = rSNode
                break

            # USED ONLY WHEN WE DONT USE REEDS-SHEPP EXPANSION OR WHEN START = GOAL
            if currentNodeIndex == self.reg_index(goal_node):
                print("Path Found")
                print(currentNode.traj[-1])
                break

            # Get all simulated Nodes from current node
            # dt ====== 0.8
            for i in range(len(self.motion)):
                simulatedNode = self.hybrid_update(currentNode, dt, self.motion[i][0], self.motion[i][1], 4)

                # Check if path is within map bounds and is collision free
                if not simulatedNode:
                    continue

                # Draw Simulated Node
                x,y,z =zip(*simulatedNode.traj)
                # plt.plot(x, y, linewidth=0.3, color='g')

                # Check if simulated node is already in closed set
                simulatedNodeIndex = self.reg_index(simulatedNode)
                if simulatedNodeIndex not in closedSet: 

                    # Check if simulated node is already in open set, if not add it open set as well as in priority queue
                    if simulatedNodeIndex not in openSet:
                        openSet[simulatedNodeIndex] = simulatedNode
                        costQueue[simulatedNodeIndex] = max(simulatedNode.cost , Cost.hybridCost * holonomicHeuristics[simulatedNode.gridIndex[0]][simulatedNode.gridIndex[1]])
                    else:
                        if simulatedNode.cost < openSet[simulatedNodeIndex].cost:
                            openSet[simulatedNodeIndex] = simulatedNode
                            costQueue[simulatedNodeIndex] = max(simulatedNode.cost , Cost.hybridCost * holonomicHeuristics[simulatedNode.gridIndex[0]][simulatedNode.gridIndex[1]])
        
        # Backtrack
        x, y, yaw = self.backtrack(start_node, goal_node, closedSet)

        return x, y, yaw
    
    def backtrack(self, startNode, goalNode, closedSet):

        # Goal Node data
        startNodeIndex= self.reg_index(startNode)
        currentNodeIndex = goalNode.parent
        print("current Node Index", currentNodeIndex)
        currentNode = closedSet[currentNodeIndex]
        x=[]
        y=[]
        yaw=[]

        # Iterate till we reach start node from goal node
        while currentNodeIndex != startNodeIndex:
            a, b, c = zip(*currentNode.traj)
            x += a[::-1] 
            y += b[::-1] 
            yaw += c[::-1]
            currentNodeIndex = currentNode.parent
            currentNode = closedSet[currentNodeIndex]
        return x[::-1], y[::-1], yaw[::-1]
    
    def holonomicNodeIndex(self, HolonomicNode):
        # Index is a tuple consisting grid index, used for checking if two nodes are near/same
        return tuple([HolonomicNode.gridIndex[0], HolonomicNode.gridIndex[1]])
    
    def reg_index(self, node):
        return tuple([node.gridIndex[0], node.gridIndex[1], node.gridIndex[2], node.gridIndex[3]])
    
    def eucledianCost(self, holonomicMotionCommand):
        # Compute Eucledian Distance between two nodes
        return math.hypot(holonomicMotionCommand[0], holonomicMotionCommand[1])
    
    def holonomicCostsWithObstacles(self, goalNode):

        gridIndex = [round(goalNode.traj[-1][0]/self.resolution), round(goalNode.traj[-1][1]/self.resolution)]
        gNode =HolonomicNode(gridIndex, 0, tuple(gridIndex))

        # obstacles = obstaclesMap(mapParameters.obstacleX, mapParameters.obstacleY, mapParameters.xyResolution)

        holonomicMotionCommand = self.get_holo_motion_model()

        openSet = {self.holonomicNodeIndex(gNode): gNode}
        closedSet = {}

        priorityQueue =[]
        heapq.heappush(priorityQueue, (gNode.cost, self.holonomicNodeIndex(gNode)))

        while True:
            if not openSet:
                break

            _, currentNodeIndex = heapq.heappop(priorityQueue)
            currentNode = openSet[currentNodeIndex]
            openSet.pop(currentNodeIndex)
            closedSet[currentNodeIndex] = currentNode

            for i in range(len(holonomicMotionCommand)):
                neighbourNode = HolonomicNode([currentNode.gridIndex[0] + holonomicMotionCommand[i][0],\
                                        currentNode.gridIndex[1] + holonomicMotionCommand[i][1]],\
                                        currentNode.cost + self.eucledianCost(holonomicMotionCommand[i]), currentNodeIndex)

                if not self.verify_node(neighbourNode):
                    continue

                neighbourNodeIndex = self.holonomicNodeIndex(neighbourNode)

                if neighbourNodeIndex not in closedSet:            
                    if neighbourNodeIndex in openSet:
                        if neighbourNode.cost < openSet[neighbourNodeIndex].cost:
                            openSet[neighbourNodeIndex].cost = neighbourNode.cost
                            openSet[neighbourNodeIndex].parent = neighbourNode.parent
                            # heapq.heappush(priorityQueue, (neighbourNode.cost, neighbourNodeIndex))
                    else:
                        openSet[neighbourNodeIndex] = neighbourNode
                        heapq.heappush(priorityQueue, (neighbourNode.cost, neighbourNodeIndex))

        holonomicCost = [[np.inf for i in range(self.calc_xy_index(self.max_y, self.min_y))]for i in range(self.calc_xy_index(self.max_x, self.min_x))]

        for nodes in closedSet.values():
            holonomicCost[nodes.gridIndex[0]][nodes.gridIndex[1]]=nodes.cost

        return holonomicCost

    def calc_grid_position(self, index, min_position):
        """
        calc grid position

        :param index:
        :param min_position:
        :return:
        """
        pos = index * self.resolution + min_position
        return pos

    def calc_xy_index(self, position, min_pos):
        return int(round((position - min_pos) / self.resolution))

    # def calc_grid_index(self, node):
    #     return (node.y - self.min_y) * self.x_width + (node.x - self.min_x)
    def verify_xy(self, traj):
        for i in traj:
            x, y = i[0], i[1]
            px = self.calc_grid_position(self.calc_xy_index(x, self.min_x), self.min_x)
            py = self.calc_grid_position(self.calc_xy_index(y, self.min_y), self.min_y)

            if px < self.min_x:
                return False
            elif py < self.min_y:
                return False
            elif px >= self.max_x:
                return False
            elif py >= self.max_y:
                return False

            # collision check
            if self.obstacle_map[self.calc_xy_index(x, self.min_x)][self.calc_xy_index(y, self.min_y)]:
                return False

        return True
    
    def verify_node(self, node):
        px = self.calc_grid_position(node.gridIndex[0], self.min_x)
        py = self.calc_grid_position(node.gridIndex[1], self.min_y)

        if px < self.min_x:
            return False
        elif py < self.min_y:
            return False
        elif px >= self.max_x:
            return False
        elif py >= self.max_y:
            return False

        # collision check
        if self.obstacle_map[node.gridIndex[0]][node.gridIndex[1]]:
            return False

        return True

    def calc_obstacle_map(self, ox, oy):

        self.min_x = round(min(ox))
        self.min_y = round(min(oy))
        self.max_x = round(max(ox))
        self.max_y = round(max(oy))

        self.x_width = round((self.max_x - self.min_x) / self.resolution)
        self.y_width = round((self.max_y - self.min_y) / self.resolution)

        # obstacle map generation
        self.obstacle_map = [[False for _ in range(self.y_width)]
                             for _ in range(self.x_width)]
        for ix in range(self.x_width):
            x = self.calc_grid_position(ix, self.min_x)
            for iy in range(self.y_width):
                y = self.calc_grid_position(iy, self.min_y)
                for iox, ioy in zip(ox, oy):
                    d = math.hypot(iox - x, ioy - y)
                    if d < self.rr:
                        self.obstacle_map[ix][iy] = True
                        break

    @staticmethod
    def get_motion_model():

        # Motion commands for a Non-Holonomic Robot like a Car or Bicycle (Trajectories using Steer Angle and Direction)
        direction = 1
        motionCommand = []
        max_steer_angle = np.deg2rad(60)
        steerPresion = 20
        for i in np.arange(max_steer_angle, -(max_steer_angle + max_steer_angle/steerPresion), -max_steer_angle/steerPresion):
            motionCommand.append([direction, i])
            motionCommand.append([direction, -i])
        return motionCommand

    def get_holo_motion_model(self):
        # Action set for a Point/Omni-Directional/Holonomic Robot (8-Directions)
        holonomicMotionCommand = [[-1, 0], [-1, 1], [0, 1], [1, 1], [1, 0], [1, -1], [0, -1], [-1, -1]]
        return holonomicMotionCommand

class PathPlanning:
    def __init__(self,obstacles):
        self.margin = 5
        # self.margin = 10
        #sacale obstacles from env margin to pathplanning margin
        obstacles = obstacles + np.array([self.margin,self.margin])
        obstacles = obstacles[(obstacles[:,0]>=0) & (obstacles[:,1]>=0)]

        self.obs = np.concatenate([np.array([[0,i] for i in range(100+self.margin)]),
                                  np.array([[100+2*self.margin,i] for i in range(100+2*self.margin)]),
                                  np.array([[i,0] for i in range(100+self.margin)]),
                                  np.array([[i,100+2*self.margin] for i in range(100+2*self.margin)]),
                                  obstacles])

        self.ox = [int(item) for item in self.obs[:,0]]
        self.oy = [int(item) for item in self.obs[:,1]]
        self.grid_size = 1
        # self.robot_radius = 4
        self.robot_radius = 5
        self.length = 4
        self.a = 1.14
        self.a_star = Hybrid_AStarPlanner(self.ox, self.oy, self.grid_size, self.robot_radius, self.length, self.a)

    def plan_path(self,sx, sy,spsi,sv, gx, gy,gpsi,gv,dt):    
        rx, ry, rpsi = self.a_star.planning(sx+self.margin, sy+self.margin,spsi,sv, gx+self.margin, gy+self.margin,gpsi,gv,dt)
        rx = np.array(rx)-self.margin+0.5
        ry = np.array(ry)-self.margin+0.5
        path = np.vstack([rx,ry,rpsi]).T
        return path