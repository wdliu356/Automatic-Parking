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
    
    # This batch move is used in mppi controller
    def batched_move(self, curr_state, accelerate, delta):
        # Accelerate is a vector of size (batch_size, 1)
        # delta is a vector of size (batch_size, 1)
        # curr_state is a vector of size (batch_size, 4)

        x = curr_state[:, 0]
        y = curr_state[:, 1]
        psi = curr_state[:, 2]
        v = curr_state[:, 3]
        beta = np.arctan((self.b * np.tan(delta)) / (self.L))
        x_dot = v*np.cos(psi + beta)
        y_dot = v*np.sin(psi + beta)
        v_dot = accelerate
        psi_dot = v*np.tan(delta)*np.cos(beta)/self.L
        return np.hstack([x_dot, y_dot, psi_dot, v_dot])
    
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
        state_diff[2] = np.min([state_diff[2], 2*np.pi - state_diff[2]]) * 0.1
        if np.linalg.norm(state_diff) < 3:
            return True
        else:
            return False
    
    def predict_state(self, u_k):
        return self.state + self.move(u_k[0], u_k[1])*self.dt
    
    def batch_predict_state(self, curr_state, u_k):
        return curr_state + self.batched_move(curr_state, u_k[:,0], u_k[:,1])*self.dt


    
class MPC_Controller:
    def __init__(self):
        self.horiz = None
        self.R = np.diag([0.01, 0.01])                 # input cost matrix
        self.Rd = np.diag([0.01, 1.0])                 # input difference cost matrix
        self.Q = np.diag([1.0, 1.0, 1.0])              # state cost matrix
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
        # self.Q = np.diag([1.0, 1.0, 1.0])
        self.Q = np.diag([5.0, 5.0, 1.0])                   # state cost matrix
        self.Qf = self.Q                               # state final matrix
        self.dt=0.01   
        self.L=4      
        self.v_cost = 0.05                    

    def make_model(self, my_car, u):        
        # matrices
        # 4*4
        b = my_car.b
        a = my_car.a
        delta = u[1]
        psi = my_car.psi
        v = my_car.v

        beta = np.arctan((b * np.tan(delta)) / (self.L))
        
        # Ap = np.array([[0, 0, -v*np.sin(psi+beta), np.cos(psi+beta)],
        #           [0, 0, v*np.cos(psi+beta), np.sin(psi+beta)],
        #           [0, 0, 0, np.tan(beta)/b/np.sqrt(np.tan(beta)**2+1)],
        #           [0, 0, 0, 0]])
        Ap = np.array([[0, 0, -v*np.sin(psi+beta), np.cos(psi+beta)],
            [0, 0, v*np.cos(psi+beta), np.sin(psi+beta)],
            [0, 0, 0, np.tan(delta)*np.cos(beta)/self.L],
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
        # Bp = np.array([[0, -b*v*np.sin(psi+beta)/((delta**2+1)*(np.tan(beta)**2+1)*(b+a))],
        #           [0, b*v*np.cos(psi+beta)/((delta**2+1)*(np.tan(beta)**2+1)*(b+a))],
        #           [0, v/((delta**2+1)*(np.tan(beta)**2+1)**(3/2)*(b+a))],
        #           [1,0]])
        L = self.L
        Q = b**2*L*np.tan(delta)**2 + L
        b11 = -b*v*np.sin(psi+beta)/(Q*np.cos(delta)**2)
        b21 = b*v*np.cos(psi+beta)/(Q*np.cos(delta)**2)
        b31 = v*(np.cos(beta) - b*np.sin(beta)*np.tan(delta)/Q)*(np.tan(delta)**2+1)/L
        Bp = np.array([[0, b11],
                       [0, b21],
                       [0, b31],
                       [1,0]])
        
        # # 4*2 

        B = self.dt*Bp

        # 4*1
        s = my_car.move(u[0], u[1])
        # print('the shape inside model')
        # print(s.shape)
        # print((Ap@my_car.state).shape)
        # print((Bp@u).shape)
        # C = self.dt*(s - Ap@my_car.state - (Bp@u).reshape(4,1))
        C = np.array([[v* np.sin(psi+beta)*psi - b11*delta],
                      [-v*np.cos(psi+beta)*psi - b21*delta],
                      [-v*delta*np.cos(beta)/(self.L*np.cos(delta)**2)+delta*b*v*np.sin(beta)*np.tan(delta)/(Q*L*np.cos(delta)**2)],
                      [0]]) * self.dt
        
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
            # print("u_k: ", u_k[:, i].shape)

            new_state = A@old_state + B@(u_k[:,i].reshape([2,1])) + C
            # predict_state = mpc_car.predict_state(u_k[:,i])
            # See if the predict state is the same as the new state
            # print("new_state: ", new_state)
            # print("predict_state: ", predict_state)
            # print("difference: ", new_state - predict_state)
            # print("New_state: ", new_state.shape)
            z_k[:,i] = np.array([new_state[0], new_state[1], new_state[2]]).reshape(3)
            cost += np.sum(self.R@(u_k[:,i]**2))
            cost += np.sum(self.Q@((desired_state[:,i]-z_k[:,i])**2))
            cost += new_state[3]**2*self.v_cost
            if i < (self.horiz-1):     
                cost += np.sum(self.Rd@((u_k[:,i+1] - u_k[:,i])**2))
            mpc_car.update_with_new_state(new_state)
            old_state = new_state
            # print(cost)
        return cost

    def optimize(self, my_car, points):
        self.horiz = points.shape[0]
        bnd = [(-5, 0.1),(np.deg2rad(-60), np.deg2rad(60))]*self.horiz
        # cons = ({'type': 'ineq', 'fun': my_car.delta + np.deg2rad(60)},{'type': 'ineq', 'fun': - my_car.delta + np.deg2rad(60)})
        result = minimize(self.mpc_cost, args=(my_car, points), x0 = np.zeros((2*self.horiz)), method='SLSQP', bounds = bnd)
        return result.x[0],  result.x[1]
    

######################################################################################################################################################################
class MPPIController(object):

    def __init__(self, model, num_samples, horizon, hyperparams):
        """

        :param env: Simulation environment. Must have an action_space and a state_space.
        :param num_samples: <int> Number of perturbed trajectories to sample
        :param horizon: <int> Number of control steps into the future
        :param hyperparams: <dic> containing the MPPI hyperparameters
        """
        self.model = model
        self.T = horizon
        self.K = num_samples
        self.lambda_ = hyperparams['lambda']
        self.action_size = 2
        self.state_size = 4
        self.goal_state = np.zeros(self.state_size)  # This is just a container for later use
        self.Q = hyperparams['Q'] # Quadratic Cost Matrix (state_size, state_size)
        self.noise_mu = np.zeros(self.action_size)
        self.noise_sigma = hyperparams['noise_sigma']  # Noise Covariance matrix shape (action_size, action_size)
        self.noise_sigma_inv = np.linalg.inv(self.noise_sigma)
        self.U = np.zeros((self.T, self.action_size)) # nominal action sequence (T, action_size)
        self.u_init = np.zeros(self.action_size)
        self.noise_dist = np.random.multivariate_normal(self.noise_mu, self.noise_sigma)

    def reset(self):
        """
        Resets the nominal action sequence
        :return:
        """
        self.U = np.zeros((self.T, self.action_size))# nominal action sequence (T, action_size)

    def command(self, state):
        """
        Run a MPPI step and return the optimal action.
        :param state: torch tensor of shape (state_size,)
        :return:
        """
        action = None
        # Need to modify perturbations
        perturbations = self.noise_dist.sample((self.K, self.T))    # shape (K, T, action_size)
        perturbed_actions = self.U + perturbations      # shape (K, T, action_size)
        trajectory = self._rollout_dynamics(state, actions=perturbed_actions)
        trajectory_cost = self._compute_trajectory_cost(trajectory, perturbations)
        self._nominal_trajectory_update(trajectory_cost, perturbations)
        # select optimal action
        action = self.U[0]
        # final update nominal trajectory
        self.U = torch.roll(self.U, -1, dims=0) # move u_t to u_{t-1}
        self.U[-1] = self.u_init # Initialize new end action
        return action

    def _rollout_dynamics(self, state_0, actions):
        """
        Roll out the environment dynamics from state_0 and taking the control actions given by actions
        :param state_0: torch tensor of shape (state_size,)
        :param actions: torch tensor of shape (K, T, action_size)
        :return:
         * trajectory: torch tensor of shape (K, T, state_size) containing the states along the trajectories given by
                       starting at state_0 and taking actions.
                       This tensor contains K trajectories of T length.
         TIP 1: You may need to call the self._dynamics method.
         TIP 2: At most you need only 1 for loop.
        """
        state = np.expand_dims(state_0, axis=0).repeat(self.K, 1) # transform it to (K, state_size)
        trajectory = None
        # --- Your code here
        trajectory = np.zeros(self.K, self.T, self.state_size)
        trajectory[:,0,:] = self._dynamics(state, actions[:,0,:])
        for t in range(1, self.T):
          trajectory[:,t,:] = self._dynamics(trajectory[:,t-1,:], actions[:,t,:])
        # ---
        return trajectory

    def _compute_trajectory_cost(self, goal_state, trajectory, perturbations):
        """
        Compute the costs for the K different trajectories
        :param trajectory: torch tensor of shape (K, T, state_size)
        :param perturbations: torch tensor of shape (K, T, action_size)
        :return:
         - total_trajectory_cost: torch tensor of shape (K,) containing the total trajectory costs for the K trajectories
        Observations:
        * The trajectory cost be the sum of the state costs and action costs along the trajectories
        * State cost should be quadratic as (state_i-goal_state)^T Q (state_i-goal_state)
        * Action costs should be given by (non_perturbed_action_i)^T noise_sigma^{-1} (perturbation_i)

        TIP 1: the nominal actions (without perturbation) are stored in self.U
        TIP 2: Check Algorithm 2 in https://ieeexplore.ieee.org/document/7989202 for more references.
        """
        total_trajectory_cost = None

        non_bias_traj = trajectory - goal_state # (K, T, state_size)
        # state_cost = torch.bmm(non_bias_traj @ self.Q, non_bias_traj.transpose(1,2)) # (K, T, T)
        # state_cost = torch.einsum("ijj -> i", state_cost)
        state_cost = np.einsum(non_bias_traj @ self.Q, non_bias_traj.transpose(1,2)) # (K, T, T)
        state_cost = np.einsum("ijj -> i", state_cost)

        action_cost = self.lambda_ * self.U @ self.noise_sigma_inv @ perturbations.transpose(1,2)
        action_cost = torch.einsum("ijj -> i", action_cost)
        total_trajectory_cost = action_cost + state_cost

        return total_trajectory_cost

    def _nominal_trajectory_update(self, trajectory_costs, perturbations):
        """
        Update the nominal action sequence (self.U) given the trajectory costs and perturbations
        :param trajectory_costs: torch tensor of shape (K,)
        :param perturbations: torch tensor of shape (K, T, action_size)
        :return: No return, you just need to update self.U

        TIP: Check Algorithm 2 in https://ieeexplore.ieee.org/document/7989202 for more references about the action update.
        """
        # --- Your code here
        beta = torch.min(trajectory_costs)
        gamma = torch.exp(- 1 / self.lambda_ * (trajectory_costs - beta))
        eta = torch.sum(gamma)
        omega = 1 / eta * gamma

        self.U += torch.einsum("i,ijk -> jk", omega, perturbations)
        # ---

    def _dynamics(self, state, accelrate, delta):
        """
        Query the environment dynamics to obtain the next_state in a batched format.
        :param state: torch tensor of size (...., state_size)
        :param action: torch tensor of size (..., action_size)
        :return: next_state: torch tensor of size (..., state_size)
        """
        # next_state = self.env.batched_dynamics(state.cpu().detach().numpy(), action.cpu().detach().numpy())
        # next_state = torch.tensor(next_state, dtype=state.dtype)
        state_dot = self.model.batch_move(state, accelrate, delta)
        next_state = self.model.batch_predict_state(state, state_dot)
        return next_state