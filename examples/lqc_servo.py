import torch
import LibsControllnew

from plot_response import *


alpha   = 0.9
beta    = 5.3

mat_a = torch.zeros((2, 2))
mat_b = torch.zeros((2, 1))
mat_c = torch.zeros((2, 2))

mat_a[0][0] = -alpha
mat_a[1][0] =  1.0



mat_b[0][0] = beta

mat_c[0][0] = 1.0
mat_c[1][1] = 1.0

required_inputs_dim     = 2
system_order_dim        = mat_a.shape[0]
plant_inputs_count      = mat_b.shape[1]
plant_outputs_count     = 2



print("mat_a = ")
print(mat_a)
print("\n\n")

print("mat_b = ")
print(mat_b)
print("\n\n")

print("mat_c = ")
print(mat_c)
print("\n\n") 

print("poles = ")
poles, _ = torch.linalg.eig(mat_a)
print(poles)


plant       = LibsControllnew.DynamicalSystem(mat_a, mat_b, mat_c)

controller  = LibsControllnew.LQC(required_inputs_dim, system_order_dim, plant_inputs_count)
#controller  = LibsControll.Nonlinear(required_inputs_dim, system_order_dim, plant_inputs_count)


optimizer   = torch.optim.Adam(controller.parameters(), lr = 1.0)


trajectory_length   = 200
batch_size          = 64
 

required_source = LibsControllnew.SignalSquare(trajectory_length, required_inputs_dim, amplitudes=[0.0, 1.0])
testing_source  = LibsControllnew.SignalUnitStep(trajectory_length, required_inputs_dim, amplitudes=[0.0, 1.0])
 

clr = LibsControllnew.ClosedLoopResponse(plant, controller)

for step in range(200):

    #zero initial state
    x_state       = torch.zeros(batch_size, system_order_dim)
    
    y             = torch.zeros(batch_size, plant_outputs_count)

    #sample random units-steps
    y_req_trajectory        = required_source.sample_batch(batch_size)

    y_req_trajectory  = torch.from_numpy(y_req_trajectory)


    #compute plant output
    u_trajectory, y_trajectory = clr.step(y_req_trajectory, 0.1, 0.1)


    loss_trajectory = ((y_req_trajectory[:,:,1] - y_trajectory[:,:,1])**2).mean()
    loss_controll   = (u_trajectory**2).mean()

    optimizer.zero_grad()
    loss = loss_trajectory + 0.001*loss_controll
    loss.backward()
    optimizer.step()
    

    if step%10 == 0:
        print(loss_trajectory, loss_controll)

        print(controller.controll_mat)

        t_trajectory = torch.arange(0, trajectory_length)

        y_req_trajectory        = testing_source.sample_batch(batch_size)
        y_req_trajectory  = torch.from_numpy(y_req_trajectory)
        

        u_trajectory, y_trajectory = clr.step(y_req_trajectory)

        
        plot_controll_output(t_trajectory, u_trajectory[:,0,:], y_req_trajectory[:,0,:], y_trajectory[:,0,:], ["voltage"], ["speed", "angle"], "images/" + str(step) + ".png")