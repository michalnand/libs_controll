import torch
from .signal_source         import *
from .closed_loop_response  import *

def fit_controller(ControllerC, plant, steps = 200, amplitudes = [1.0, 1.0, 1.0, 1.0], loss_weight = [1.0, 1.0, 1.0, 1.0], x_noise = 0.01, y_noise = 0.0, batch_size = 64, lr = 0.1, dt=0.01):

    y_required_dim          = plant.mat_c.shape[1]
    x_state_dim             = plant.mat_a.shape[0]
    u_dim                   = plant.mat_b.shape[1]
    y_output_dim            = plant.mat_c.shape[1]

    controller  = ControllerC(y_required_dim, x_state_dim, u_dim, y_output_dim)

    optimizer   = torch.optim.AdamW(controller.parameters(), lr = lr)

    trajectory_length   = 400
    
    #required_source = SignalSquare(trajectory_length, y_required_dim, amplitudes=amplitudes)
    required_source     = SignalUnitStep(trajectory_length, y_required_dim, amplitudes=amplitudes)


    clr = ClosedLoopResponse(plant, controller, dt)

    for step in range(steps):
        #sample random units-steps
        y_req_trajectory  = required_source.sample_batch(batch_size)

        y_req_trajectory  = torch.from_numpy(y_req_trajectory)

        #compute plant output
        y_trajectory, u_trajectory = clr.step(y_req_trajectory, x_noise, y_noise, random_initial_state = 0.1)

        loss_controll   = (u_trajectory**2).mean()

        loss_trajectory = 0  
        decay = 0.995**(y_req_trajectory.shape[0] - torch.arange(y_req_trajectory.shape[0]))
        decay = decay.unsqueeze(1)
        

        for i in range(len(loss_weight)):
            loss_trajectory+= loss_weight[i]*(decay*((y_req_trajectory[:,:,i] - y_trajectory[:,:,i])**2)).mean()


        optimizer.zero_grad()
        loss = loss_trajectory + 0.001*loss_controll
        loss.backward()
        optimizer.step()
        
        if step%10 == 0:
            print("loss = ", loss_trajectory.detach().to("cpu").numpy(), loss_controll.detach().to("cpu").numpy())
            print(str(controller))

    return controller
          