import torch


'''
apply control law : 

y = xK

where 
x = [required_value, plant_state]
K = controller martrix, shape (y_required_dim+system_order_dim, system_inputs_dim)


y_required_dim - required value
system_inputs_dim   - controlled plant inputs count
system_order_dim    - fully observed system state

'''
class LQC(torch.nn.Module):
    def __init__(self, y_required_dim, x_state_dim, u_dim, y_output_dim, init_range = 0.0):
        super().__init__()

        controll_mat        = init_range*torch.randn((y_required_dim + y_output_dim, u_dim)).float()

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
    
    def forward(self, y_required, y_output):
        x   = torch.cat([y_required, y_output], dim=1)    
        u   = torch.mm(x, self.controll_mat) 
    
        return u


class LQCHidden(torch.nn.Module):
    def __init__(self, y_required_dim, x_state_dim, u_dim, y_output_dim, order = 4, init_range = 0.0):
        super().__init__()

        self.hidden_dim     = y_output_dim*order
        controll_mat        = init_range*torch.randn((y_required_dim + y_output_dim*order, u_dim)).float()

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

    def forward(self, y_required, y_output, h_state):

        h_state_new = torch.roll(h_state, shifts=y_output.shape[1], dims=1)
        h_state_new[:,0:y_output.shape[1]] = y_output

        x   = torch.cat([y_required, h_state_new], dim=1)    
        u   = torch.mm(x, self.controll_mat) 
    
        return u, h_state_new  



'''
class Observer(torch.nn.Module):
    def __init__(self, x_state_dim, y_output_dim, init_range = 0.0001):
        super().__init__()

        mat_a        = init_range*torch.randn((system_order_dim, system_order_dim)).float()
        self.mat_a   = torch.nn.parameter.Parameter(mat_a, requires_grad=True)

        mat_c        = init_range*torch.randn((system_order_dim, system_outputs_dim)).float()
        self.mat_c   = torch.nn.parameter.Parameter(mat_c, requires_grad=True)

        mat_l        = init_range*torch.randn((system_outputs_dim, system_order_dim)).float()
        self.mat_l   = torch.nn.parameter.Parameter(mat_l, requires_grad=True)

    def __repr__(self):
        res = ""
        res+= "mat_a=\n"
        for j in range(self.mat_a.shape[0]):
            for i in range(self.mat_a.shape[1]):
                v = round(float(self.mat_a[j][i]), 4)
                res+= str("%8.4f"%v) + " "
            res+= "\n"
        res+= "\n"

        res+= "mat_c=\n"
        for j in range(self.mat_c.shape[0]):
            for i in range(self.mat_c.shape[1]):
                v = round(float(self.mat_c[j][i]), 4)
                res+= str("%8.4f"%v) + " "
            res+= "\n"
        res+= "\n"

        res+= "mat_l=\n"
        for j in range(self.mat_l.shape[0]):
            for i in range(self.mat_l.shape[1]):
                v = round(float(self.mat_l[j][i]), 4)
                res+= str("%8.4f"%v) + " "
            res+= "\n"
        res+= "\n"

        return res

    def forward(self, x_hat, y):
        y_hat     = torch.mm(x_hat, self.mat_c)
        error     = y - y_hat

        g         = torch.mm(error, self.mat_l)

        x_hat_new = torch.mm(x_hat, self.mat_a) + g
        return x_hat_new
        


class LQCHidden(torch.nn.Module):
    def __init__(self, required_inputs_dim, system_order_dim, system_inputs_dim, system_outputs_dim, init_range = 0.0):
        super().__init__()

        self.hidden_dim     = system_order_dim
        controll_mat        = init_range*torch.randn((required_inputs_dim  + system_outputs_dim + system_order_dim, system_inputs_dim)).float()

        self.observer       = Observer(system_order_dim, system_outputs_dim)
        self.controll_mat   = torch.nn.parameter.Parameter(controll_mat, requires_grad=True)

    def __repr__(self):
        res = ""

        res+= str(self.observer)
        res+= "\n"

        res+= "controll_mat=\n"
        for j in range(self.controll_mat.shape[0]):
            for i in range(self.controll_mat.shape[1]):
                v = round(float(self.controll_mat[j][i]), 4)
                res+= str("%8.4f"%v) + " "
            res+= "\n"
        res+= "\n"

        return res
    
    def forward(self, required_state, plant_output, hidden_state):
        hidden_new = self.observer(hidden_state, plant_output)

        x   = torch.cat([required_state, plant_output, hidden_new], dim=1)    
        y   = torch.mm(x, self.controll_mat) 
    
        return hidden_new, y
'''


class Nonlinear(torch.nn.Module):
    def __init__(self,  y_required_dim, x_state_dim, u_dim, y_output_dim, hidden_dim = 32):
        super().__init__()

        self.hidden_dim = hidden_dim

        inputs_count = y_required_dim + y_output_dim

        self.gru    = torch.nn.GRU(inputs_count, self.hidden_dim, batch_first = True)
        self.lin    = torch.nn.Linear(self.hidden_dim, u_dim)


    def forward(self, y_required, y_output, h_state):
        x  = torch.cat([y_required, y_output], dim=1).unsqueeze(1)

        #rnn step
        _, h_state_new = self.gru(x, h_state.unsqueeze(0))

        #take final hidden state
        h_state_new = h_state_new.squeeze(0)

        u = self.lin(h_state_new)

        return u, h_state_new