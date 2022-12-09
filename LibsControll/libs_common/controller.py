import torch


'''
apply control law : 

y = xK

where 
x = [required_value, plant_state]
K = controller martrix, shape (required_inputs_dim+system_order_dim, system_inputs_dim)


required_inputs_dim - required value
system_inputs_dim   - controlled plant inputs count
system_order_dim    - fully observed system state

'''
class LQC(torch.nn.Module):
    def __init__(self, required_inputs_dim, system_order_dim, system_inputs_dim, init_range = 0.0):
        super().__init__()

        controll_mat        = init_range*torch.randn((required_inputs_dim + system_order_dim, system_inputs_dim)).float()

        '''
        for i in range(required_inputs_dim):
            controll_mat[i] = 1.0

        for i in range(system_order_dim):
            controll_mat[i + required_inputs_dim] = -1.0
        '''

        self.controll_mat   = torch.nn.parameter.Parameter(controll_mat, requires_grad=True)

    def __repr__(self):
        res = ""

        res+= "controll_mat=\n"
        for j in range(self.controll_mat.shape[0]):
            for i in range(self.controll_mat.shape[1]):
                v = round(float(self.controll_mat[j][i]), 4)
                res+= str("%8.4f"%v) + " "
            res+= "\n"
        res+= "\n"

        return res
    
    def forward(self, required_state, plant_state):
        x   = torch.cat([required_state, plant_state], dim=1)    
        y   = torch.mm(x.detach(), self.controll_mat) 
    
        return y



class Nonlinear(torch.nn.Module):
    def __init__(self, required_inputs_dim, plant_output_dim, system_inputs_dim, hidden_dim = 64):
        super().__init__()

        self.hidden_dim = hidden_dim

        inputs_count = required_inputs_dim + plant_output_dim

        self.gru    = torch.nn.GRU(inputs_count, self.hidden_dim, batch_first = True)
        self.lin    = torch.nn.Linear(self.hidden_dim, system_inputs_dim)


    
    def forward(self, required_state, plant_state, hidden_state):
        x  = torch.cat([required_state, plant_state], dim=1).unsqueeze(1)

        #rnn step
        _, hidden_new = self.gru(x, hidden_state.unsqueeze(0))

        #take final hidden state
        hidden_new = hidden_new.squeeze(0)

        y = self.lin(hidden_new)

        return hidden_new, y