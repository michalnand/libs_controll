import torch
from .signal_source         import *
from .closed_loop_response  import *


def step_response_closed_loop(controller, plant, amplitudes, trajectory_length = 200, dt = 1.0/200.0):

   
    required_inputs_dim     = plant.mat_c.shape[1]
    
    testing_source          = SignalUnitStep(trajectory_length, required_inputs_dim, amplitudes=amplitudes)

    y_req_trajectory        = testing_source.sample_batch(1)
    y_req_trajectory        = torch.from_numpy(y_req_trajectory)
        
    clr = ClosedLoopResponse(plant, controller, dt)

    u_trajectory, y_trajectory = clr.step(y_req_trajectory)

    t_trajectory = dt*torch.arange(0, u_trajectory.shape[0])

    y_req_trajectory = y_req_trajectory.squeeze(1)
    u_trajectory = u_trajectory.squeeze(1)
    y_trajectory = y_trajectory.squeeze(1)

   

    return t_trajectory, y_req_trajectory, u_trajectory, y_trajectory
          