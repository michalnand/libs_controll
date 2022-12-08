import torch

from .ode_solver import *

class ClosedLoopResponseContinuous:

    def __init__(self, plant, controller, dt=1.0/200.0):
        self.plant      = plant
        self.controller = controller

        self.dt = dt

        self.solver      = ODESolverRK4(plant)

    def step(self, required, plant_state_x):
        u = self.controller(required, plant_state_x) 
        x_new, y = self.solver.step(plant_state_x, u, self.dt)

        return y, x_new
 
        