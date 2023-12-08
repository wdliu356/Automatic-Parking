import numpy as np
import math
import scipy.interpolate as scipy_interpolate
from utils import angle_of_line
from queue import PriorityQueue

############################################## Functions ######################################################

def interpolate_b_spline_path(x, y, n_path_points, degree=3):
    ipl_t = np.linspace(0.0, len(x) - 1, len(x))
    spl_i_x = scipy_interpolate.make_interp_spline(ipl_t, x, k=degree)
    spl_i_y = scipy_interpolate.make_interp_spline(ipl_t, y, k=degree)
    travel = np.linspace(0.0, len(x) - 1, n_path_points)
    return spl_i_x(travel), spl_i_y(travel)

def interpolate_path(path, sample_rate):
    choices = np.arange(0,len(path),sample_rate)
    if len(path)-1 not in choices:
            choices =  np.append(choices , len(path)-1)
    way_point_x = path[choices,0]
    way_point_y = path[choices,1]
    n_course_point = len(path)*3
    rix, riy = interpolate_b_spline_path(way_point_x, way_point_y, n_course_point)
    new_path = np.vstack([rix,riy]).T
    # new_path[new_path<0] = 0
    return new_path

################################################ Path Planner ################################################

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
        self.rr = rr
        self.min_x, self.min_y = 0, 0
        self.max_x, self.max_y = 0, 0
        self.obstacle_map = None
        self.x_width, self.y_width = 0, 0
        self.motion = self.get_motion_model()
        self.calc_obstacle_map(ox, oy)
        self.length = length
        self.a = a

    class Node:
        def __init__(self, x, y, psi, v, cost, parent, length, a):
            self.x = x  # x position, can be used to calculate its position in the grid map
            self.y = y  # y position, can be used to calculate its position in the grid map
            self.psi = psi
            self.v = v
            self.cost = cost
            self.parent= parent
            self.length = length
            self.a = a
            

        def __str__(self):
            return str(self.x) + "," + str(self.y) + "," + str(
                self.cost) + "," + str(self.parent)
        
        def update(self,dt,acc,delta):
            b = self.length - self.a
            beta = np.arctan((b * np.tan(delta)) / (self.length))
            x_dot = self.v*np.cos(self.psi + beta)
            y_dot = self.v*np.sin(self.psi + beta)
            v_dot = acc
            psi_dot = self.v*np.tan(delta)*np.cos(beta)/self.length
            x = self.x + dt*x_dot
            y = self.y + dt*y_dot
            psi = self.psi + dt*psi_dot
            v = self.v + dt*v_dot
            cost = self.cost + np.sqrt(0.35*x_dot**2 + 0.35*y_dot**2 + 0.2*psi_dot**2 + 0.1*v_dot**2 + 0.1*delta**2 + 0.1*acc**2)
            return [x,y,psi,v,cost,self.length,self.a]

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

        start_node = self.Node(sx, sy, spsi, sv, 0.0, None, self.length, self.a)
        goal_node = self.Node(gx, gy, gpsi, gv, 0.0, None, self.length, self.a)
        test_set = set()
        pq = PriorityQueue()
        pq.put((self.calc_heuristic(start_node,goal_node),0, start_node))
        test_set.add((start_node.x, start_node.y, start_node.psi, start_node.v))
        id = 0
        while 1:
            if pq.empty():
                print("Nothing found, pq is empty")
                break

            current = pq.get()[2]### get the top of the queue and remove it
            dis = self.calc_heuristic(current, goal_node)
            print("current: ", current.x, current.y, current.psi, current.v)
            print("dis: ", dis)
            if dis <= 0.5:
                print("Find goal")
                goal_node.parent = current
                goal_node.cost = current.cost + self.calc_heuristic(current, goal_node)
                break


            # Add it to the closed set
            test_set.add((current.x, current.y, current.psi, current.v))

            # expand_grid search grid based on motion model
            for i, _ in enumerate(self.motion):
                node_info = current.update(dt,self.motion[i][0],self.motion[i][1])
                # print("node_info: ", node_info)
                node = self.Node(node_info[0], node_info[1], node_info[2], node_info[3], 0, current, node_info[5], node_info[6])
                # n_id = self.calc_grid_index(node)

                # If the node is not safe, do nothing
                if (node.x, node.y, node.psi, node.v) in test_set:
                    continue
                if not self.verify_node(node):
                    continue
                id+=1
                pq.put((node.cost + self.calc_heuristic(node, goal_node),id, node))
                # print(i)

        rx, ry = self.calc_final_path(goal_node,start_node)

        return rx, ry

    def calc_final_path(self, goal_node,start_node):
        # generate final course
        node_list = [goal_node]
        current = goal_node
        while current.parent!=None:
            current = current.parent
            node_list.append(current)
        node_list.append(start_node)
        node_list.reverse()
        rx, ry = [], []
        for node in node_list:
            rx.append(node.x)
            ry.append(node.y)
        return rx, ry

    @staticmethod
    def calc_heuristic(n1, n2):
        weight = [0.35, 0.35, 0.2, 0.1]  # x, y, psi, v
        d = np.sqrt(weight[0]*(n1.x - n2.x)**2 + weight[1]*(n1.y - n2.y)**2 + weight[2]*(n1.psi - n2.psi)**2 + weight[3]*(n1.v - n2.v)**2)
        return d

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

    def calc_grid_index(self, node):
        return (node.y - self.min_y) * self.x_width + (node.x - self.min_x)

    def verify_node(self, node):
        px = self.calc_grid_position(self.calc_xy_index(node.x, self.min_x), self.min_x)
        py = self.calc_grid_position(self.calc_xy_index(node.y, self.min_y), self.min_y)

        if px < self.min_x:
            return False
        elif py < self.min_y:
            return False
        elif px >= self.max_x:
            return False
        elif py >= self.max_y:
            return False

        # collision check
        if self.obstacle_map[self.calc_xy_index(node.x, self.min_x)][self.calc_xy_index(node.y, self.min_y)]:
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
        # acc, delta
        # motion = [[0.1,0],[0.1,np.deg2rad(60)], [0.1,np.deg2rad(-60)], [0.1,np.deg2rad(30)], [0.1,np.deg2rad(-30)],
        #            [-0.1,0], [-0.1,np.deg2rad(60)], [-0.1,np.deg2rad(-60)], [-0.1,np.deg2rad(30)], [-0.1,np.deg2rad(-30)],
        #            [0,0],[0,np.deg2rad(60)], [0,np.deg2rad(-60)], [0,np.deg2rad(30)], [0,np.deg2rad(-30)]]

        acc = 10
        delta = 60
        motion = [[acc, 0], [-acc, 0], [0, 0], 
                  [acc, np.deg2rad(delta)], [-acc, np.deg2rad(delta)],
                  [acc, np.deg2rad(-delta)], [-acc, np.deg2rad(-delta)]]

        return motion


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
        rx, ry = self.a_star.planning(sx+self.margin, sy+self.margin,spsi,sv, gx+self.margin, gy+self.margin,gpsi,gv,dt)
        rx = np.array(rx)-self.margin+0.5
        ry = np.array(ry)-self.margin+0.5
        path = np.vstack([rx,ry]).T
        return path[::-1]

############################################### Park Path Planner #################################################

# class ParkPathPlanning:
#     def __init__(self,obstacles):
#         self.margin = 5
#         #sacale obstacles from env margin to pathplanning margin
#         obstacles = obstacles + np.array([self.margin,self.margin])
#         obstacles = obstacles[(obstacles[:,0]>=0) & (obstacles[:,1]>=0)]

#         self.obs = np.concatenate([np.array([[0,i] for i in range(100+self.margin)]),
#                                   np.array([[100+2*self.margin,i] for i in range(100+2*self.margin)]),
#                                   np.array([[i,0] for i in range(100+self.margin)]),
#                                   np.array([[i,100+2*self.margin] for i in range(100+2*self.margin)]),
#                                   obstacles])

#         self.ox = [int(item) for item in self.obs[:,0]]
#         self.oy = [int(item) for item in self.obs[:,1]]
#         self.grid_size = 1
#         self.robot_radius = 4
#         self.a_star = AStarPlanner(self.ox, self.oy, self.grid_size, self.robot_radius)

#     def generate_park_scenario(self,sx, sy, gx, gy):    
#         rx, ry = self.a_star.planning(sx+self.margin, sy+self.margin, gx+self.margin, gy+self.margin)
#         rx = np.array(rx)-self.margin+0.5
#         ry = np.array(ry)-self.margin+0.5
#         path = np.vstack([rx,ry]).T
#         path = path[::-1]
#         computed_angle = angle_of_line(path[-10][0],path[-10][1],path[-1][0],path[-1][1])

#         s = 4
#         l = 8
#         d = 2
#         w = 4

#         if -math.atan2(0,-1) < computed_angle <= math.atan2(-1,0):
#             x_ensure2 = gx
#             y_ensure2 = gy
#             x_ensure1 = x_ensure2 + d + w
#             y_ensure1 = y_ensure2 - l - s
#             ensure_path1 = np.vstack([np.repeat(x_ensure1,3/0.25), np.arange(y_ensure1-3,y_ensure1,0.25)[::-1]]).T
#             ensure_path2 = np.vstack([np.repeat(x_ensure2,3/0.25), np.arange(y_ensure2,y_ensure2+3,0.25)[::-1]]).T
#             park_path = self.plan_park_down_right(x_ensure2, y_ensure2)

#         elif math.atan2(-1,0) <= computed_angle <= math.atan2(0,1):
#             x_ensure2 = gx
#             y_ensure2 = gy
#             x_ensure1 = x_ensure2 - d - w
#             y_ensure1 = y_ensure2 - l - s 
#             ensure_path1 = np.vstack([np.repeat(x_ensure1,3/0.25), np.arange(y_ensure1-3,y_ensure1,0.25)[::-1]]).T
#             ensure_path2 = np.vstack([np.repeat(x_ensure2,3/0.25), np.arange(y_ensure2,y_ensure2+3,0.25)[::-1]]).T
#             park_path = self.plan_park_down_left(x_ensure2, y_ensure2)

#         elif math.atan2(0,1) < computed_angle <= math.atan2(1,0):
#             x_ensure2 = gx
#             y_ensure2 = gy
#             x_ensure1 = x_ensure2 - d - w
#             y_ensure1 = y_ensure2 + l + s
#             ensure_path1 = np.vstack([np.repeat(x_ensure1,3/0.25), np.arange(y_ensure1,y_ensure1+3,0.25)]).T
#             ensure_path2 = np.vstack([np.repeat(x_ensure2,3/0.25), np.arange(y_ensure2-3,y_ensure2,0.25)]).T
#             park_path = self.plan_park_up_left(x_ensure2, y_ensure2)

#         elif math.atan2(1,0) < computed_angle <= math.atan2(0,-1):
#             x_ensure2 = gx
#             y_ensure2 = gy
#             x_ensure1 = x_ensure2 + d + w
#             y_ensure1 = y_ensure2 + l + s
#             ensure_path1 = np.vstack([np.repeat(x_ensure1,3/0.25), np.arange(y_ensure1,y_ensure1+3,0.25)]).T
#             ensure_path2 = np.vstack([np.repeat(x_ensure2,3/0.25), np.arange(y_ensure2-3,y_ensure2,0.25)]).T
#             park_path = self.plan_park_up_right(x_ensure2, y_ensure2)

#         return np.array([x_ensure1, y_ensure1]), park_path, ensure_path1, ensure_path2


#     def plan_park_up_right(self, x1, y1):       
#             s = 4
#             l = 8
#             d = 2
#             w = 4

#             x0 = x1 + d + w
#             y0 = y1 + l + s
            
#             curve_x = np.array([])
#             curve_y = np.array([])
#             y = np.arange(y1,y0+1)
#             circle_fun = (6.9**2 - (y-y0)**2)
#             x = (np.sqrt(circle_fun[circle_fun>=0]) + x0-6.9)
#             y = y[circle_fun>=0]
#             choices = x>x0-6.9/2
#             x=x[choices]
#             y=y[choices]
#             curve_x = np.append(curve_x, x[::-1])
#             curve_y = np.append(curve_y, y[::-1])
            
#             y = np.arange(y1,y0+1)
#             circle_fun = (6.9**2 - (y-y1)**2)
#             x = (np.sqrt(circle_fun[circle_fun>=0]) + x1+6.9)
#             y = y[circle_fun>=0]
#             x = (x - 2*(x-(x1+6.9)))
#             choices = x<x1+6.9/2
#             x=x[choices]
#             y=y[choices]
#             curve_x = np.append(curve_x, x[::-1])
#             curve_y = np.append(curve_y, y[::-1])

#             park_path = np.vstack([curve_x, curve_y]).T
#             return park_path

#     def plan_park_up_left(self, x1, y1):       
#             s = 4
#             l = 8
#             d = 2
#             w = 4

#             x0 = x1 - d - w
#             y0 = y1 + l + s
            
#             curve_x = np.array([])
#             curve_y = np.array([])
#             y = np.arange(y1,y0+1)
#             circle_fun = (6.9**2 - (y-y0)**2)
#             x = (np.sqrt(circle_fun[circle_fun>=0]) + x0+6.9)
#             y = y[circle_fun>=0]
#             x = (x - 2*(x-(x0+6.9)))
#             choices = x<x0+6.9/2
#             x=x[choices]
#             y=y[choices]
#             curve_x = np.append(curve_x, x[::-1])
#             curve_y = np.append(curve_y, y[::-1])
            
#             y = np.arange(y1,y0+1)
#             circle_fun = (6.9**2 - (y-y1)**2)
#             x = (np.sqrt(circle_fun[circle_fun>=0]) + x1-6.9)
#             y = y[circle_fun>=0]
#             choices = x>x1-6.9/2
#             x=x[choices]
#             y=y[choices]
#             curve_x = np.append(curve_x, x[::-1])
#             curve_y = np.append(curve_y, y[::-1])

#             park_path = np.vstack([curve_x, curve_y]).T
#             return park_path


#     def plan_park_down_right(self, x1,y1):
#             s = 4
#             l = 8
#             d = 2
#             w = 4

#             x0 = x1 + d + w
#             y0 = y1 - l - s
            
#             curve_x = np.array([])
#             curve_y = np.array([])
#             y = np.arange(y0,y1+1)
#             circle_fun = (6.9**2 - (y-y0)**2)
#             x = (np.sqrt(circle_fun[circle_fun>=0]) + x0-6.9)
#             y = y[circle_fun>=0]
#             choices = x>x0-6.9/2
#             x=x[choices]
#             y=y[choices]
            
#             curve_x = np.append(curve_x, x)
#             curve_y = np.append(curve_y, y)
            
#             y = np.arange(y0,y1+1)
#             circle_fun = (6.9**2 - (y-y1)**2)
#             x = (np.sqrt(circle_fun[circle_fun>=0]) + x1+6.9)
#             x = (x - 2*(x-(x1+6.9)))
#             y = y[circle_fun>=0]
#             choices = x<x1+6.9/2
#             x=x[choices]
#             y=y[choices]
#             curve_x = np.append(curve_x, x)
#             curve_y = np.append(curve_y, y)
            
#             park_path = np.vstack([curve_x, curve_y]).T
#             return park_path


#     def plan_park_down_left(self, x1,y1):
#             s = 4
#             l = 8
#             d = 2
#             w = 4

#             x0 = x1 - d - w
#             y0 = y1 - l - s
            
#             curve_x = np.array([])
#             curve_y = np.array([])
#             y = np.arange(y0,y1+1)
#             circle_fun = (6.9**2 - (y-y0)**2)
#             x = (np.sqrt(circle_fun[circle_fun>=0]) + x0+6.9)
#             y = y[circle_fun>=0]
#             x = (x - 2*(x-(x0+6.9)))
#             choices = x<x0+6.9/2
#             x=x[choices]
#             y=y[choices]
#             curve_x = np.append(curve_x, x)
#             curve_y = np.append(curve_y, y)
            
#             y = np.arange(y0,y1+1)
#             circle_fun = (6.9**2 - (y-y1)**2)
#             x = (np.sqrt(circle_fun[circle_fun>=0]) + x1-6.9)
#             y = y[circle_fun>=0]
#             choices = x>x1-6.9/2
#             x=x[choices]
#             y=y[choices]
#             curve_x = np.append(curve_x, x)
#             curve_y = np.append(curve_y, y)
            
#             park_path = np.vstack([curve_x, curve_y]).T
#             return park_path