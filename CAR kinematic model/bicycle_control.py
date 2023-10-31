import numpy as np
from scipy.optimize import minimize
import copy


class Car_Dynamics:
    def __init__(self, x_0, y_0, psi_0, v_0, length, dt, a):
        self.dt = dt             # sampling time
        self.L = length          # vehicle length
        self.x = x_0
        self.y = y_0
        self.v = v_0
        self.psi = psi_0
        self.state = np.array([[self.x, self.y, self.psi, self.v]]).T
        self.a = a
        self.b = length - a

    def move(self, accelerate, delta):
        self.beta = np.arctan((self.b * np.tan(delta)) / (self.L))
        x_dot = self.v*np.cos(self.psi + self.beta)
        y_dot = self.v*np.sin(self.psi + self.beta)
        v_dot = accelerate
        psi_dot = self.v*np.tan(delta)*np.cos(self.beta)/self.L
        return np.array([[x_dot, y_dot, psi_dot, v_dot]]).T
    
    # def fn(self, accelerate, delta):
    #     beta = np.arctan((self.b * np.tan(delta)) / (self.L))
    #     x_dot = self.v*np.cos(self.psi + self.beta)
    #     y_dot = self.v*np.sin(self.psi + self.beta)
    #     v_dot = accelerate
    #     psi_dot = self.v*np.tan(delta)*np.cos(self.beta)/self.L
    #     return np.array([[x_dot, y_dot, psi_dot, v_dot]]).T

    def update_state(self, state_dot):
        # self.u_k = command
        # self.z_k = state
        self.state = self.state + self.dt*state_dot
        self.x = self.state[0,0]
        self.y = self.state[1,0]
        self.psi = self.state[2,0]
        self.v = self.state[3,0]
    
    def update_with_new_state(self, state):
        self.state = state
        self.x = self.state[0,0]
        self.y = self.state[1,0]
        self.psi = self.state[2,0]
        self.v = self.state[3,0]

    def goal_state_reached(self, goal_state):
        curr_state = np.array([self.x, self.y, self.psi])
        state_diff = np.abs(curr_state - goal_state)
        state_diff[2] = np.min([state_diff[2], 2*np.pi - state_diff[2]])
        if np.linalg.norm(state_diff) < 0.5:
            return True
        else:
            return False


    
class MPC_Controller:
    def __init__(self):
        self.horiz = None
        self.R = np.diag([0.01, 0.01])                 # input cost matrix
        self.Rd = np.diag([0.01, 1.0])                 # input difference cost matrix
        self.Q = np.diag([1.0, 1.0, 1.0])                   # state cost matrix
        self.Qf = self.Q                               # state final matrix

    def mpc_cost(self, u_k, my_car, points):
        mpc_car = copy.copy(my_car)
        u_k = u_k.reshape(self.horiz, 2).T
        z_k = np.zeros((3, self.horiz+1))
    
        desired_state = points.T
        cost = 0.0

        for i in range(self.horiz):
            state_dot = mpc_car.move(u_k[0,i], u_k[1,i])
            mpc_car.update_state(state_dot)
        
            z_k[:,i] = [mpc_car.x, mpc_car.y, mpc_car.psi]
            cost += np.sum(self.R@(u_k[:,i]**2))
            cost += np.sum(self.Q@((desired_state[:,i]-z_k[:,i])**2))
            if i < (self.horiz-1):     
                cost += np.sum(self.Rd@((u_k[:,i+1] - u_k[:,i])**2))
        return cost

    def optimize(self, my_car, points):
        self.horiz = points.shape[0]
        bnd = [(-5, 5),(np.deg2rad(-60), np.deg2rad(60))]*self.horiz
        # cons = ({'type': 'ineq', 'fun': my_car.delta + np.deg2rad(60)},{'type': 'ineq', 'fun': - my_car.delta + np.deg2rad(60)})
        result = minimize(self.mpc_cost, args=(my_car, points), x0 = np.zeros((2*self.horiz)), method='SLSQP', bounds = bnd)
        return result.x[0],  result.x[1]



######################################################################################################################################################################

class Linear_MPC_Controller:
    def __init__(self):
        self.horiz = None
        self.R = np.diag([0.01, 0.01])                 # input cost matrix
        self.Rd = np.diag([0.01, 1.0])                 # input difference cost matrix
        self.Q = np.diag([1.0, 1.0, 1.0])                   # state cost matrix
        self.Qf = self.Q                               # state final matrix
        self.dt=0.01   
        self.L=4                          

    def make_model(self, my_car, u):        
        # matrices
        # 4*4
        b = my_car.b
        a = my_car.a
        delta = u[1]
        psi = my_car.psi
        v = my_car.v

        beta = np.arctan((b * np.tan(delta)) / (self.L))
        
        Ap = np.array([[0, 0, -v*np.sin(psi+beta), np.cos(psi+beta)],
                  [0, 0, v*np.cos(psi+beta), np.sin(psi+beta)],
                  [0, 0, 0, np.tan(beta)/b/np.sqrt(np.tan(beta)**2+1)],
                  [0, 0, 0, 0]])
        A = np.eye(4) + self.dt*Ap
        # A = np.array([[1, 0, -self.dt*v*np.sin(psi+beta), self.dt*np.cos(psi+beta)],
        #           [0, 1, self.dt*v*np.cos(psi+beta), self.dt*np.sin(psi+beta)],
        #           [0, 0, 1, self.dt*np.tan(beta)/self.b/np.sqrt(np.tan(beta)**2+1)],
        #           [0, 0, 0, 1]])
    
        # B = np.array([[0, -self.dt*self.b*v*np.sin(psi+beta)/((delta**2+1)*(np.tan(beta)**2+1)*(self.b+self.a))],
        #           [0, self.dt*self.b*v*np.cos(psi+beta)/((delta**2+1)*(np.tan(beta)**2+1)*(self.b+self.a))],
        #           [0, self.dt*v/((delta**2+1)*(np.tan(beta)**2+1)**(3/2)*(self.b+self.a))],
        #           [self.dt,0]])
        Bp = np.array([[0, -b*v*np.sin(psi+beta)/((delta**2+1)*(np.tan(beta)**2+1)*(b+a))],
                  [0, b*v*np.cos(psi+beta)/((delta**2+1)*(np.tan(beta)**2+1)*(b+a))],
                  [0, v/((delta**2+1)*(np.tan(beta)**2+1)**(3/2)*(b+a))],
                  [1,0]])
        
        # # 4*2 

        B = self.dt*Bp

        # 4*1
        s = my_car.move(u[0], u[1])
        # print('the shape inside model')
        # print(s.shape)
        # print((Ap@my_car.state).shape)
        # print((Bp@u).shape)
        C = self.dt*(s - Ap@my_car.state - (Bp@u).reshape(4,1))
        
        return A, B, C

    def mpc_cost(self, u_k, my_car, points):
        mpc_car = copy.copy(my_car)
        u_k = u_k.reshape(self.horiz, 2).T
        z_k = np.zeros((3, self.horiz+1))
        desired_state = points.T
        cost = 0.0
        old_state = np.array([my_car.x, my_car.y, my_car.psi, my_car.v]).reshape(4,1)

        for i in range(self.horiz):
            # delta = u_k[1,i]
            A,B,C = self.make_model(mpc_car, u_k[:,i])
            # Print all the shapes
            # print('********************************')
            # print("A: ", A.shape)
            # print("B: ", B.shape)
            # print("C: ", C.shape)
            # print("old_state: ", old_state.shape)
            # print("u_k: ", u_k.shape)

            new_state = A@old_state + B@u_k + C
        
            z_k[:,i] = [new_state[0,0], new_state[1,0], new_state[2,0]]
            cost += np.sum(self.R@(u_k[:,i]**2))
            cost += np.sum(self.Q@((desired_state[:,i]-z_k[:,i])**2))
            if i < (self.horiz-1):     
                cost += np.sum(self.Rd@((u_k[:,i+1] - u_k[:,i])**2))
            mpc_car.update_with_new_state(new_state)
            old_state = new_state
            print(cost)
        return cost

    def optimize(self, my_car, points):
        self.horiz = points.shape[0]
        bnd = [(-5, 5),(np.deg2rad(-60), np.deg2rad(60))]*self.horiz
        # cons = ({'type': 'ineq', 'fun': my_car.delta + np.deg2rad(60)},{'type': 'ineq', 'fun': - my_car.delta + np.deg2rad(60)})
        result = minimize(self.mpc_cost, args=(my_car, points), x0 = np.zeros((2*self.horiz)), method='SLSQP', bounds = bnd)
        return result.x[0],  result.x[1]