import torch.nn as nn
from Domain import *

class PINN(nn.Module):
    """
    Class to define the Physics Informed Neural Network (PINN) architecture.
    This class inherits from nn.Module and defines a feedforward neural network with
    a specified number of hidden layers and neurons per layer.
    It also includes methods for computing the loss based on boundary conditions,
    outlet conditions, and the PDE residuals.
    """
    # def __init__(self, layers):
    #     super(PINN, self).__init__()
    #     self.layers = layers
    #     self.activation = nn.Tanh()
    #     self.weights = nn.ParameterList([nn.Parameter(torch.randn(layers[i], layers[i + 1]) * np.sqrt(2 / (layers[i] + layers[i + 1]))) for i in range(len(layers) - 1)])
    #     self.biases = nn.ParameterList([nn.Parameter(torch.zeros(layers[i + 1])) for i in range(len(layers) - 1)])

    # def forward(self, x):
    #     for weight, bias in zip(self.weights, self.biases):
    #         x = self.activation(torch.matmul(x, weight) + bias)
    #     return x
    def __init__(self, num_hidden_layers: int =4, num_neurons: int =20,
                 rho: float=1, mu: float=0.01):
        """
        Initialize the PINN with a specified number of hidden layers and neurons per layer, and specify the physical properties of the flow.
        
        Parameters
        ----------
        num_hidden_layers : int
            The number of hidden layers in the neural network (between the input and output layers). Default is 4.
        num_neurons : int
            The number of neurons in each hidden layer. Default is 20.
        rho : float
            The density of the fluid. Default is 1.
        mu : float
            The dynamic viscosity of the fluid. Default is 0.01.
        """
        super(PINN, self).__init__()
        layers = [nn.Linear(2, num_neurons), nn.Tanh()]
        for _ in range(num_hidden_layers):
            layers.append(nn.Linear(num_neurons, num_neurons))
            layers.append(nn.Tanh())
        layers.append(nn.Linear(num_neurons, 3))

        self.network = nn.Sequential(*layers)
        ## PINN state information ##
        self.losses = {"total": [],"pde": [],"bc": [], "walls": [], "inlets": [], "outlet": []}
        self.epochs = []
        ## Physical Properties ##
        self.rho = rho  # Density
        self.mu = mu    # Dynamic viscosity
    
    def forward(self, xy):
        output = torch.hsplit(self.network(xy),3)
        return output
    
    def residuals_eq(self, domain: Domain):
        """ Compute the normalized residuals of the PDE equations for the PINN."""
        xy = domain.domain_data["xy_pde"]
        # Predict u, v, p from the model 
        u, v, p = self(xy)
        # Compute first-order derivatives using autograd
        du = torch.autograd.grad(u, xy, grad_outputs=torch.ones_like(u), create_graph=True, retain_graph=True)[0]
        dv = torch.autograd.grad(v, xy, grad_outputs=torch.ones_like(v), create_graph=True, retain_graph=True)[0]
        u_x, u_y, v_x, v_y = du[:,0:1], du[:,1:], dv[:,0:1], dv[:,1:]
        # Compute second-order derivatives for the viscosity terms
        ddu_x = torch.autograd.grad(u_x, xy, grad_outputs=torch.ones_like(u_x), create_graph=True, retain_graph=True)[0]
        u_xx = ddu_x[:,0:1]
        ddu_y = torch.autograd.grad(u_y, xy, grad_outputs=torch.ones_like(u_y), create_graph=True, retain_graph=True)[0]
        u_yy = ddu_y[:,1:]
        ddv_x = torch.autograd.grad(v_x, xy, grad_outputs=torch.ones_like(v_x), create_graph=True, retain_graph=True)[0]
        v_xx = ddv_x[:,0:1]
        ddv_y = torch.autograd.grad(v_y, xy, grad_outputs=torch.ones_like(v_y), create_graph=True, retain_graph=True)[0]
        v_yy = ddv_y[:,1:]
        # Compute pressure gradients
        dp = torch.autograd.grad(p, xy, grad_outputs=torch.ones_like(p), create_graph=True, retain_graph=True)[0]
        p_x = dp[:,0:1]
        p_y = dp[:,1:]
        # Continuity equation (incompressibility condition)
        continuity = u_x + v_y  # Shape: [num_points, 1]
        # Navier-Stokes momentum equations
        momentum_u = self.rho * (u * u_x + v * u_y) + p_x - self.mu * (u_xx + u_yy)  # Shape: [num_points, 1]
        momentum_v = self.rho * (u * v_x + v * v_y) + p_y - self.mu * (v_xx + v_yy)  # Shape: [num_points, 1]
        # Concatenate all residuals along dimension 0
        eq_residuals = torch.cat([continuity, momentum_u, momentum_v], dim=0)
        return (1/np.sqrt(domain.N_pde)) * eq_residuals
    
    def residuals_bc(self, domain: Domain):
        """ Compute the normalized residuals of the Boundary Conditions for the PINN."""
        u_bc, v_bc, _ = self(domain.domain_data["xy_bc"])
        _, _, p_out = self(domain.domain_data["xy_outlet"])
        # Compute residuals for boundary conditions
        uv_bc = domain.domain_data["uv_bc"]
        N_bc, N_out = uv_bc.shape[0], domain.domain_data["xy_outlet"].shape[0] 
        bc_residuals = torch.cat([u_bc - uv_bc[:,0:1], v_bc - uv_bc[:,1:], p_out - 0], dim=0)
        return (1/np.sqrt(N_bc + N_out)) * bc_residuals
    
    def residuals_total(self, domain: Domain):
        """ Compute the total residuals for the PINN, which includes both the PDE residuals and boundary condition residuals."""
        total_res = torch.cat([self.residuals_eq(domain), self.residuals_bc(domain)], dim = 0)
        return total_res
    
    def loss(self, domain: Domain):
        """
        Compute the individual and total losses for the PINN, which includes the PDE residuals and boundary condition residuals.
        """
        # Compute residuals for PDE and boundary conditions
        res_eq = self.residuals_eq(domain)
        loss_eq = torch.sum(res_eq**2)
        res_bc = self.residuals_bc(domain)
        loss_bc = torch.sum(res_bc**2)
        ## Split the BC residuals for boundary conditions into parts: Walls - Inlets - Outlet
        num_inl_pts = round(len(domain.inlets)*domain.N_inl)
        res_inlets = res_bc[:num_inl_pts] # Inlets residuals
        res_walls = res_bc[num_inl_pts:-domain.N_inl] # Walls residuals
        res_bc = res_bc[-domain.N_inl:]  # Outlet residuals
        loss_inlets = torch.sum(res_inlets**2)  # Inlet loss
        loss_walls = torch.sum(res_walls**2)  # Wall loss
        loss_outlet = torch.sum(res_bc**2)  # Outlet loss
        total_loss = torch.sum(res_eq**2) + torch.sum(res_bc**2) # Total loss
        # Store losses for monitoring
        self.losses["total"].append(total_loss.detach().cpu().item())
        self.losses["pde"].append(loss_eq.detach().cpu().item())
        self.losses["bc"].append(loss_bc.detach().cpu().item())
        self.losses["walls"].append(loss_walls.detach().cpu().item())
        self.losses["inlets"].append(loss_inlets.detach().cpu().item())
        self.losses["outlet"].append(loss_outlet.detach().cpu().item())
        
