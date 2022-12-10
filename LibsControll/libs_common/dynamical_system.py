import torch


'''
dx = xA^T + uB^T
y  = xC


mat_a.shape = (system_order, system_order)
mat_b.shape = (system_order, inputs_count)
mat_c.shape = (system_order, outputs_count)

inputs : 
system prev. state  : x.shape = (batch, system_order)
system input        : u.shape = (batch, inputs_count)


returns:
state change dx, shape = (batch, system_order)
system output y, shape = (batch, outputs_count)
'''
class DynamicalSystem(torch.nn.Module):
    def __init__(self, mat_a, mat_b, mat_c):
        super().__init__()

        self.mat_a  = torch.nn.parameter.Parameter(mat_a, requires_grad=True)
        self.mat_b  = torch.nn.parameter.Parameter(mat_b, requires_grad=True)
        self.mat_c  = torch.nn.parameter.Parameter(mat_c, requires_grad=True)

    def forward(self, x, u):
        dx = torch.mm(x, self.mat_a.T) + torch.mm(u, self.mat_b.T)
        y  = torch.mm(x, self.mat_c)

        return y, dx


    def __repr__(self):
        res = ""

        res+= "mat_a=\n"
        for j in range(self.mat_a.shape[0]):
            for i in range(self.mat_a.shape[1]):
                v = round(float(self.mat_a[j][i]), 4)
                res+= str("%8.4f"%v) + " "
            res+= "\n"
        res+= "\n"

        res+= "mat_b=\n"
        for j in range(self.mat_b.shape[0]):
            for i in range(self.mat_b.shape[1]):
                v = round(float(self.mat_b[j][i]), 4)
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


        poles, _ = torch.linalg.eig(self.mat_a)
        poles = poles.detach().to("cpu").numpy()

        res+= "poles=\n"
        for j in range(poles.shape[0]):
            res+= str(round(poles[j], 5)) + "\n"
        res+= "\n\n"

        return res






class DynamicalSystemVar(torch.nn.Module):
    def __init__(self, batch_size, mat_a, mat_b, mat_c, var_a, var_b, var_c):
        super().__init__()

        self.initial_mat_a      = mat_a
        self.initial_var_a      = var_a

        self.initial_mat_b      = mat_b
        self.initial_var_b      = var_b

        self.initial_mat_c      = mat_c
        self.initial_var_c      = var_c

        self.batch_size         = batch_size
        
        self.reset()

    def reset(self):
        a_value = self.initial_mat_a.unsqueeze(0) + self.initial_var_a*torch.randn((self.batch_size, ) + self.initial_var_a.shape)
        b_value = self.initial_mat_b.unsqueeze(0) + self.initial_var_b*torch.randn((self.batch_size, ) + self.initial_var_b.shape)
        c_value = self.initial_mat_c.unsqueeze(0) + self.initial_var_c*torch.randn((self.batch_size, ) + self.initial_var_c.shape)

        self.mat_a  = torch.nn.parameter.Parameter(a_value, requires_grad=True)
        self.mat_b  = torch.nn.parameter.Parameter(b_value, requires_grad=True)
        self.mat_c  = torch.nn.parameter.Parameter(c_value, requires_grad=True)

    def forward(self, x, u):
        x_      = x.unsqueeze(2)
        u_      = u.unsqueeze(2)

        dx = torch.bmm(self.mat_a, x_) + torch.bmm(self.mat_b, u_)

        dx = dx.squeeze(2)

        y  = torch.bmm(self.mat_c, x_)
        y  = y.squeeze(2)

        return dx, y
    
    '''
    def forward(self, x, u):
        dx = torch.bmm(x, torch.transpose(self.mat_a, 1, 2)) + torch.bmm(u, torch.transpose(self.mat_b, 1, 2))
        y  = torch.bmm(x, self.mat_c)

        return dx, y
    '''

    def __repr__(self):

        mat_a = self.mat_a.mean(dim=0)
        mat_b = self.mat_b.mean(dim=0)
        mat_c = self.mat_c.mean(dim=0)

        res = ""

        res+= "mat_a=\n"
        for j in range(mat_a.shape[0]):
            for i in range(mat_a.shape[1]):
                v = round(float(mat_a[j][i]), 4)
                res+= str("%8.4f"%v) + " "
            res+= "\n"
        res+= "\n"

        res+= "mat_b=\n"
        for j in range(mat_b.shape[0]):
            for i in range(mat_b.shape[1]):
                v = round(float(mat_b[j][i]), 4)
                res+= str("%8.4f"%v) + " "
            res+= "\n"
        res+= "\n"

        res+= "mat_c=\n"
        for j in range(mat_c.shape[0]):
            for i in range(mat_c.shape[1]):
                v = round(float(mat_c[j][i]), 4)
                res+= str("%8.4f"%v) + " "
            res+= "\n"
        res+= "\n"

        poles, _ = torch.linalg.eig(mat_a)
        poles = poles.detach().to("cpu").numpy()

        res+= "poles=\n"
        for j in range(poles.shape[0]):
            res+= str(round(poles[j], 5)) + "\n"
        res+= "\n\n"

        return res
