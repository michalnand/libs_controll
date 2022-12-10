'''
Euler solver, 1st order
'''
class ODESolverEuler:

    def __init__(self, dynamical_system):
        self.dynamical_system = dynamical_system

    def step(self, x, u, dt = 0.001):
        dx, y  = self.dynamical_system(x, u)
        return x + dx*dt, y

'''
Runge-Kuta solver, 4th order
'''
class ODESolverRK4: 

    def __init__(self, dynamical_system):
        self.dynamical_system = dynamical_system

    def step(self, x, u, dt = 0.001):

        y1, k1  = self.dynamical_system(x, u)
        k1      = k1*dt

        y2, k2  = self.dynamical_system(x + 0.5*k1, u + 0.5*dt)
        k2      = k2*dt

        y3, k3  = self.dynamical_system(x + 0.5*k2, u + 0.5*dt)
        k3      = k3*dt

        y4, k4  = self.dynamical_system(x + k3    , u + dt)
        k4      = k4*dt

        y       = (1.0/6.0)*(1.0*y1 + 2.0*y2 + 2.0*y3 + 1.0*y4)
        x_new   = x + (1.0/6.0)*(1.0*k1 + 2.0*k2 + 2.0*k3 + 1.0*k4)
        
        return y, x_new
