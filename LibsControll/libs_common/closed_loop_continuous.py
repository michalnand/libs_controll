import torch

from .ode_solver import *

class ClosedLoopResponseContinuous:

    def __init__(self, plant, controller, dt=1.0/200.0):
        self.plant      = plant
        self.controller = controller

        self.dt = dt

        self.solver      = ODESolverRK4(plant)

    def step(self, y_required, y_output, x_state, h_state = None):

        if hasattr(self.controller, "hidden_dim"):
            u, h_state = self.controller(y_required, y_output, h_state) 
        else:
            u = self.controller(y_required, x_state)  

        y, x_new = self.solver.step(x_state, u, self.dt)
 
        return y, x_new, h_state
 
        