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

        return dx, y


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


if __name__ == "__main__":

    batch_size      = 128
    
    system_order    = 7
    inputs_count    = 3
    outputs_count   = 2


    mat_a           = torch.randn((system_order, system_order))
    mat_b           = torch.randn((system_order, inputs_count))
    mat_c           = torch.randn((system_order, outputs_count))

    system          = DynamicalSystem(mat_a, mat_b, mat_c)

    x               = torch.randn((batch_size, system_order))
    input           = torch.randn((batch_size, inputs_count))


    for i in range(10):
        dx, y           = system(x, input)

        x = x + dx

        print("state_shape  = ", x.shape)
        print("input_shape  = ", input.shape)
        print("dx_shape     = ", dx.shape)
        print("output_shape = ", y.shape)
        print()