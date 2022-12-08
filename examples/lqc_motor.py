import torch
import LibsControllnew

from plot_response import *


alpha   = -0.9
beta    = 3.7


mat_a = torch.zeros((1, 1))
mat_b = torch.zeros((1, 1))
mat_c = torch.zeros((1, 1))

mat_a[0][0] = alpha

mat_b[0][0] = beta

mat_c[0][0] = 1.0


required_inputs_dim     = 1
system_order_dim        = mat_a.shape[0]
plant_inputs_count      = mat_b.shape[1]
plant_outputs_count     = 1

print("mat_a = ")
print(mat_a)
print("\n\n")

print("mat_b = ")
print(mat_b)
print("\n\n")

print("mat_c = ")
print(mat_c)
print("\n\n") 


plant       = LibsControllnew.DynamicalSystem(mat_a, mat_b, mat_c)

controller  = LibsControllnew.LQC(required_inputs_dim, system_order_dim, plant_inputs_count)
#controller  = LibsControll.Nonlinear(required_inputs_dim, system_order_dim, plant_inputs_count)

solver      = LibsControllnew.ODESolverRK4(plant)
#solver      = LibsControll.ODESolverEuler(plant)

optimizer   = torch.optim.Adam(controller.parameters(), lr = 0.25)

dt          = 1.0/200.0

trajectory_length   = 200
batch_size          = 64


required_source = LibsControllnew.SignalUnitStep(trajectory_length, required_inputs_dim, amplitudes=[1.0])


for step in range(100):

    #zero initial state
    x_state       = torch.randn(batch_size, system_order_dim)
    
    y             = torch.zeros(batch_size, plant_outputs_count)

    #sample random units-steps
    y_req_trajectory        = required_source.sample_batch(batch_size)

    y_req_trajectory  = torch.from_numpy(y_req_trajectory)


    #storage for plant output
    y_trajectory  = torch.zeros((trajectory_length, batch_size, plant_outputs_count)).float()

    #storage for controller output
    u_trajectory  = torch.zeros((trajectory_length, batch_size, plant_inputs_count)).float()


    for t in range(trajectory_length):
        u = controller(y_req_trajectory[t], y) 

        x_new, y = solver.step(x_state, u, dt)

        x_state = x_new

        y_trajectory[t] = y
        u_trajectory[t] = u

    loss_trajectory = ((y_req_trajectory[:,:,0] - y_trajectory[:,:,0])**2).mean()
    loss_controll   = (u_trajectory**2).mean()

    loss = loss_trajectory + 0.001*loss_controll
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    

    if step%10 == 0:
        print(controller.controll_mat)
        print(loss_trajectory, loss_controll, (u**2).mean())

        t_trajectory = dt*torch.arange(0, trajectory_length)
        
        plot_controll_output(t_trajectory, u_trajectory[:,0,:], y_req_trajectory[:,0,:], y_trajectory[:,0,:], ["voltage"], ["speed"], "images/" + str(step) + ".png")