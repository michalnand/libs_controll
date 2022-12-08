import torch
from .signal_source         import *
from .closed_loop_response  import *

def fit_controller(ControllerC, plant, steps = 200, amplitudes = [0.0, 1.0, 0.0, 0.0], loss_weight = [1.0, 1.0, 1.0, 1.0], x_noise = 0.1, y_noise = 0.0, lr = 1.0):

    required_inputs_dim     = plant.mat_c.shape[0]
    system_order_dim        = plant.mat_a.shape[0]
    plant_inputs_count      = plant.mat_b.shape[1]
    plant_outputs_count     = plant.mat_c.shape[0]

    controller  = ControllerC(required_inputs_dim, system_order_dim, plant_inputs_count)

    optimizer   = torch.optim.Adam(controller.parameters(), lr = lr)


    trajectory_length   = 400
    batch_size          = 64
    
    required_source = SignalSquare(trajectory_length, required_inputs_dim, amplitudes=amplitudes)
    
    clr = ClosedLoopResponse(plant, controller)

    for step in range(steps):
        #sample random units-steps
        y_req_trajectory  = required_source.sample_batch(batch_size)

        y_req_trajectory  = torch.from_numpy(y_req_trajectory)

        #compute plant output
        u_trajectory, y_trajectory = clr.step(y_req_trajectory, x_noise, y_noise)

        loss_trajectory = 0 

        for i in range(len(loss_weight)):
            loss_trajectory+= loss_weight[i]*((y_req_trajectory[:,:,i] - y_trajectory[:,:,i])**2).mean()

        loss_controll   = (u_trajectory**2).mean()

        optimizer.zero_grad()
        loss = loss_trajectory + 0.001*loss_controll
        loss.backward()
        optimizer.step()
        
        if step%10 == 0:
            print("loss = ", loss_trajectory.detach().to("cpu").numpy(), loss_controll.detach().to("cpu").numpy())

  

    return controller
          