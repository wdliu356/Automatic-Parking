import numpy as np
import math
from scipy.optimize import minimize
import copy

class Car_Dynamics:
    def __init__(self, x_0, y_0, psi_0, u_0, v_0, r_0, length, dt, Gama, w_0 = 0):
        self.dt = dt             # sampling time
        self.L = length          # vehicle length
        self.Gama = Gama         # Slope Angle of Earth
        self.x = x_0
        self.y = y_0
        self.psi = psi_0
        self.u = u_0
        self.v = v_0
        self.r = r_0
        self.w = w_0
        self.state = np.array([[self.x, self.y, self.psi, self.u, self.v, self.r, self.w]]).T
        self.Parameters = self.generate_parameters()

    def move(self, T, df):
        par = self.Parameters
        Fzf = par['m']*par['g']*par['b']/par['L']
        Fzr = par['m']*par['g']*par['a']/par['L']
        Fz = Fzr/2
        # dydx = np.zeros((7,1))
        c1 = df
        c2 = T
        # lambda = ( y( 3 ) * rw - y( 2 ) )./( y( 3 ) * rw );
        # alphaf = rad2deg( c1 - atan( ( y( 5 ) + a * y( 7 ) )/y( 2 ) ) ); %% convert from
        # rad to degrees
        # alphar = rad2deg( atan( ( y( 5 ) - b * y( 7 ) )/y( 2 ) ) ); %% convert from rad to
        # degrees
        # phix = ( 1 - Ex ) * ( lambda + Shx ) + Ex/Bx * atan( Bx * ( lambda + Shx ) );
        # phiyf = ( 1 - Ey ) * ( alphaf + Shy ) + Ey/By * atan( By * ( alphaf + Shy ) );
        # phiyr = ( 1 - Ey ) * ( alphar + Shy ) + Ey/By * atan( By * ( alphar + Shy ) );
        # %driving force per driven wheel
        # Fx = Dx*Fz* sin( Cx * atan( Bx * phix ) ) + Svx;
        # %front lateral force per 2 front wheels
        # Fyf = Dy*Fzf * sin( Cy * atan( By * phiyf ) ) + Svy;
        # %the rear lateral force is negated because of how we calculate the slip angle in
        # %the SAE coordinate frame this is also per 2 rear wheels
        # Fyr = -Dy*Fzr * sin( Cy * atan( By * phiyr ) ) + Svy;
        # %check for tire saturation
        # Frmax=mu*Fzr;
        # Frtotal=sqrt((2*Fx)^2+Fyr^2);
        # if Frtotal>Frmax
        # Fx=Frmax/Frtotal*(Fx);
        # Fyr=Frmax/Frtotal*(Fyr);
        # end
        # dydt( 1 ) = y( 2 ) .* cos( y( 6 ) ) - y( 5 ) .* sin( y( 6 ) );
        # dydt( 2 ) = (-0.5*rho*Cd*A*(y(2)+uw^2) -f * m * g + Nw * Fx )/m;
        # dydt( 3 ) = ( c2 - rw * Fx - fw * Fz - bw * y( 3 ) )/J;%wheel rpm
        # dydt( 4 ) = y( 2 ) .* sin( y( 6 ) ) - y( 5) .* cos( y( 6 ) );
        # dydt( 5 ) = ( Fyf + Fyr )/m - y( 2 ) .* y( 7 );
        # dydt( 6 ) = y( 7 );
        # dydt( 7 ) = ( Fyf * a - Fyr * b )/Iz;
        #Translate all these comments above into Python
        print("---------------------------")
        print("before self.u: ", self.u)
        print("before self.psi: ", self.psi)
        print("before self.w: ", self.w)
        if abs(self.w) <= 0.001 and abs(self.u) <= 0.001:
            l = 0
        else:
            if abs(self.u) <= 0.00001:
                if self.w>0:
                    l = (self.w*par['rw'] - self.u)/(self.w*par['rw'])
                else:
                    l = -(self.w*par['rw'] - self.u)/(self.w*par['rw'])

            elif T * self.u > 0:
                # print("w speed: ", self.w*par['rw'])
                if self.u > 0:
                    l = (self.w*par['rw'] - self.u)/(self.w*par['rw'])
                else:
                    l = - (self.w*par['rw'] - self.u)/(self.w*par['rw'])
            else:
                if self.u > 0:
                    l = (self.w*par['rw'] - self.u)/self.u
                else:
                    l = - (self.w*par['rw'] - self.u)/self.u
        if self.u == 0:
            alphaf = 0.0
            alphar = 0.0
        else:
            if self.u > 0.0:
                alphaf = np.rad2deg(c1 - math.atan((self.v + par['a']*self.r)/self.u))
                alphar = np.rad2deg(-math.atan((self.v - par['b']*self.r)/self.u))
            else:
                alphaf = -np.rad2deg(c1 - math.atan((self.v + par['a']*self.r)/self.u))
                alphar = -np.rad2deg(-math.atan((self.v - par['b']*self.r)/self.u))
        print("self.r: ", self.r)
        print("self.v: ", self.v)
        print("alphaf: ", alphaf)
        print("alphar: ", alphar)
        phix = (1 - par['Ex'])*(l + par['Shx']) + par['Ex']/par['Bx']*math.atan(par['Bx']*(l + par['Shx']))
        phiyf = (1 - par['Ey'])*(alphaf + par['Shy']) + par['Ey']/par['By']*math.atan(par['By']*(alphaf + par['Shy']))
        phiyr = (1 - par['Ey'])*(alphar + par['Shy']) + par['Ey']/par['By']*math.atan(par['By']*(alphar + par['Shy']))
        # print("l: ", l)
        # print("phix: ", phix)
        print("phiyf: ", phiyf)
        print("phiyr: ", phiyr)
        Fx = par['Dx']*Fz*math.sin(par['Cx']*math.atan(par['Bx']*phix)) + par['Svx']
        Fyf = par['Dy']*Fzf*math.sin(par['Cy']*math.atan(par['By']*phiyf)) + par['Svy']
        Fyr = - par['Dy']*Fzr*math.sin(par['Cy']*math.atan(par['By']*phiyr)) + par['Svy']
        print("Fyf: ", Fyf)
        print("Fyr: ", Fyr)
        Frmax = par['mu']*Fzr

        Frtotal = math.sqrt((2*Fx)**2 + Fyr**2)
        if Frtotal > Frmax:
            Fx = Frmax/Frtotal*Fx
            Fyr = Frmax/Frtotal*Fyr
        print("Fx: ", Fx)

        x_dot = self.u*math.cos(self.psi) - self.v*math.sin(self.psi)
        y_dot = self.u*math.sin(self.psi) + self.v*math.cos(self.psi)
        psi_dot = self.r
        if self.u > 0.001:
            u_dot = (-0.5*par['rho']*par['Cd']*par['A']*(self.u + par['uw'])**2 - par['f']*par['m']*par['g'] + par['Nw']*Fx)/par['m']
            print(1)
        elif self.u < -0.001:
            u_dot = (0.5*par['rho']*par['Cd']*par['A']*(self.u + par['uw'])**2 + par['f']*par['m']*par['g'] + par['Nw']*Fx)/par['m']
            print(2)

        else:
            # if par['f']*par['m']*par['g'] > abs(par['Nw']*Fx):
            if c2/par['rw'] <= par['mu']*par['m']*par['g']:
                u_dot = 0.0
            else:
                if Fx > 0.0:
                    u_dot = (-par['f']*par['m']*par['g'] + par['Nw']*Fx)/par['m']
                    print(3)
                else:
                    u_dot = (par['f']*par['m']*par['g'] + par['Nw']*Fx)/par['m']
                    print(4)
        print("u_dot: ", u_dot)
        w_dot = (c2 - par['rw']*Fx - par['fw']*Fz - par['bw']*self.w)/par['J']
        print("w_dot: ", w_dot)
        v_dot = (Fyf + Fyr)/par['m'] - self.u*self.r
        print("v_dot: ", v_dot)
        r_dot = (Fyf*par['a'] + Fyr*par['b'])/par['Iz']
        print("r_dot: ", r_dot)


        return np.array([[x_dot, y_dot, psi_dot, u_dot, v_dot, r_dot, w_dot]]).T

    def update_state(self, state_dot):
        self.state = self.state + self.dt*state_dot
        self.x = self.state[0,0]
        self.y = self.state[1,0]
        self.psi = self.state[2,0]
        self.u = self.state[3,0]
        self.v = self.state[4,0]
        self.r = self.state[5,0]
        self.w = self.state[6,0]

    def generate_parameters(self):
        # Parameters
        # Vehicle Model: Renault Mégane Coupé 16V 150 HP 
        Parameters ={}

        # General data
        Parameters['g'] = 9.8
        Parameters['m'] = 1400
        Parameters['f'] = 0.01
        Parameters['Cd'] = 0.3
        Parameters['A'] = 2.5
        Parameters['rho'] = 1.225
        Parameters['uw'] = 0.0     
        Parameters['Iz'] = 2420
        Parameters['bw'] = 0
        Parameters['fw'] = 0
        Parameters['Nw'] = 2
        Parameters['rw'] = 0.31
        Parameters['J'] = 2.65
        Parameters['a'] = 1.14
        Parameters['L'] = 2.54
        Parameters['b'] = Parameters['L'] - Parameters['a']      
        Parameters['mu'] = 0.8 ## set as constant. Related to the condition of the road surface
        Parameters['Dx'] = Parameters['mu']
        Parameters['Dy'] = Parameters['mu']
        Parameters['Bx'] = 0 if Parameters['mu'] == 0 else 25/Parameters['mu']
        Parameters['By'] = 0 if Parameters['mu'] == 0 else 0.27/Parameters['mu'] 
        Parameters['Cx'] = 1.35
        Parameters['Shx'] = 0
        Parameters['Svx'] = 0
        Parameters['Ex'] = -2.9
        Parameters['Cy'] = 1.2
        Parameters['Ey'] = -1.6
        Parameters['Shy'] = 0
        Parameters['Svy'] = 0
        Parameters['g'] = 9.86055

        return Parameters



    
class MPC_Controller:
    def __init__(self):
        self.horiz = None
        self.R = np.diag([0.01, 0.01])                 # input cost matrix
        self.Rd = np.diag([0.01, 1.0])                 # input difference cost matrix
        self.Q = np.diag([1.0, 1.0])         # state cost matrix
        self.Qf = self.Q       

    def mpc_cost(self, u_k, my_car, points):
        mpc_car = copy.copy(my_car)
        u_k = u_k.reshape(self.horiz, 2).T
        z_k = np.zeros((2, self.horiz+1))
    
        desired_state = points.T
        cost = 0.0

        for i in range(self.horiz):
            state_dot = mpc_car.move(u_k[0,i], u_k[1,i])
            mpc_car.update_state(state_dot)
        
            z_k[:,i] = [mpc_car.x, mpc_car.y]
            cost += np.sum(self.R@(u_k[:,i]**2))
            cost += np.sum(self.Q@((desired_state[:,i]-z_k[:,i])**2))
            if i < (self.horiz-1):     
                cost += np.sum(self.Rd@((u_k[:,i+1] - u_k[:,i])**2))
        return cost

    def optimize(self, my_car, points):
        self.horiz = points.shape[0]
        bnd = [(0.1, 1),(np.deg2rad(-60), np.deg2rad(60))]*self.horiz
        result = minimize(self.mpc_cost, args=(my_car, points), x0 = np.zeros((2*self.horiz)), method='SLSQP', bounds = bnd)
        return result.x[0],  result.x[1]



######################################################################################################################################################################

    # def make_model(self, v, phi, delta):        
    #     # matrices
    #     # 4*4
    #     A = np.array([[1, 0, self.dt*np.cos(phi)         , -self.dt*v*np.sin(phi)],
    #                   [0, 1, self.dt*np.sin(phi)         , self.dt*v*np.cos(phi) ],
    #                   [0, 0, 1                           , 0                     ],
    #                   [0, 0, self.dt*np.tan(delta)/self.L, 1                     ]])
    #     # 4*2 
    #     B = np.array([[0      , 0                                  ],
    #                   [0      , 0                                  ],
    #                   [self.dt, 0                                  ],
    #                   [0      , self.dt*v/(self.L*np.cos(delta)**2)]])

    #     # 4*1
    #     C = np.array([[self.dt*v* np.sin(phi)*phi                ],
    #                   [-self.dt*v*np.cos(phi)*phi                ],
    #                   [0                                         ],
    #                   [-self.dt*v*delta/(self.L*np.cos(delta)**2)]])
        
    #     return A, B, C

    # def move(self, accelerate, steer):
    #     delta = np.deg2rad(steer)
    #     u_k = np.array([[accelerate, delta]]).T
    #     A,B,C = self.make_model(self.v, self.phi, delta)
    #     z_k1 = A@self.z_k + B@u_k + C
    #     return u_k, z_k1