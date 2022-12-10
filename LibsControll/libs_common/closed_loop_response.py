import torch

from .ode_solver import *

class ClosedLoopResponse:

    def __init__(self, plant, controller, dt=1.0/200.0):
        self.plant      = plant
        self.controller = controller

        self.dt = dt

        self.solver      = ODESolverRK4(plant)

    def step(self, input, state_noise = 0.0, observation_noise = 0.0, random_initial_state = 0.0):

        trajectory_length = input.shape[0]
        batch_size        = input.shape[1]

        plant_order        = self.plant.mat_a.shape[1]
        plant_inputs_count = self.plant.mat_b.shape[1]
        plant_outputs_count= self.plant.mat_c.shape[1]

        #plant state, zero initial conditions
        x_state      = torch.zeros((batch_size, plant_order)).float()
        x_state+= random_initial_state*torch.randn_like(x_state)
        
        #storage for controller output
        u_trajectory  = torch.zeros((trajectory_length, batch_size, plant_inputs_count)).float()

        #storage for plant output
        y_trajectory  = torch.zeros((trajectory_length, batch_size, plant_outputs_count)).float()

        #controller hidden state, if any
        if hasattr(self.controller, "hidden_dim"):
            h_state  = torch.zeros((batch_size, self.controller.hidden_dim)).float()

        y = torch.zeros((batch_size, plant_outputs_count)).float()


        for t in range(trajectory_length):
            y_noise = observation_noise*torch.randn((batch_size, plant_outputs_count)).float() 
            x_noise = state_noise*torch.randn((batch_size, plant_order)).float() 

            if hasattr(self.controller, "hidden_dim"):
                h_state, u = self.controller(input[t], y + y_noise, h_state) 
            else:
                u = self.controller(input[t], y + y_noise) 

            x_new, y = self.solver.step(x_state, u, self.dt)
 
            x_state         = x_new + x_noise

            y_trajectory[t] = y
            u_trajectory[t] = u 

        return u_trajectory, y_trajectory
