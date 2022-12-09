import torch
import LibsControll 


dt = 1.0/200.0

'''
create example dynamical system, 4th order
state is : pos1, pos2, vel1, vel2
'''
plant       = LibsControll.BalancingRobot() 

print(str(plant))

'''
train controller, 400 steps trajectory length
care only for pos2, loss_weight=[0, 1, 0, 0]
'''
controller   = LibsControll.fit_controller(LibsControll.LQC, plant, steps = 400, amplitudes=[0, 1, 0, 0], loss_weight=[0, 1, 0, 0], dt=dt)

'''
plot result for step response
'''
t_trajectory, y_req_trajectory, u_trajectory, y_trajectory  = LibsControll.step_response(controller, plant, amplitudes=[0, 1, 0, 0], dt=dt)
LibsControll.plot_controll_output(t_trajectory, u_trajectory, y_req_trajectory, y_trajectory, ["force [N]"], ["position 1 [m]", "position 2 [m]", "speed 1 [m/s]", "speed 2 [m/s]"], "images/balancing_robot_step_response.png")

print(str(controller))

print("training done")



 
        

clr = LibsControll.ClosedLoopResponseContinuous(plant, controller, dt)

y_req       = torch.zeros((1, plant.mat_a.shape[0]))
y_req[:, :] = 1.0

x     = torch.zeros((1, plant.mat_a.shape[0]))

steps = 0

while True:
    y, x = clr.step(y_req, x)

    plant.render()

    steps+= 1
    if steps%400 == 0:
        y_req*= -1
    