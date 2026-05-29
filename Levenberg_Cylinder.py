import time
import torch
import torch.nn as nn
from torch.func import jacrev, functional_call  # Updated to use torch.func
import numpy as np
from torch.autograd import Variable
from pyDOE import lhs
from matplotlib import pyplot as plt
# Set Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Set bit precision !! (IMPORTANT for convergence)
torch.set_default_device(device)
default_dtype = torch.float64
torch.set_default_dtype(default_dtype)
savepath = "Levenberg_Benchmarks/Results/Cylinder/"
# Ensure Reproducability
torch.manual_seed(6969)
np.random.seed(6969)

# Define the neural network
class PINN(nn.Module):
    def __init__(self, num_hidden_layers=4, num_neurons=20, rho=1, mu=0.01):
        super(PINN, self).__init__()
        layers = [nn.Linear(2, num_neurons), nn.Tanh()]
        for _ in range(num_hidden_layers):
            layers.append(nn.Linear(num_neurons, num_neurons))
            layers.append(nn.Tanh())
        layers.append(nn.Linear(num_neurons, 3))
        self.network = nn.Sequential(*layers)
        self.losses = {"bc": [], "outlet": [], "pde": [],}
        self.epochs = []
        
        # Physical Quantities
        self.rho = rho 
        self.mu = mu

    def forward(self, xy):
        u, v, p = torch.hsplit(self.network(xy),3)
        return u, v, p
    
    def residuals_eq(self, xy):
        # Predict u, v, p from the model using the flattened inputs
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
        return eq_residuals
    
    def residuals_bc(self,xy_out, xy_bc, uv):
        u_bc, v_bc, _ = self(xy_bc)
        _, _, p_out = self(xy_out)
        # Compute residuals for boundary conditions
        bc_residuals = torch.cat([u_bc - uv[:,0:1], v_bc - uv[:,1:], p_out - 0], dim=0)
        return bc_residuals

    def residuals_total(self, xy_col, xy_out, xy_bc, uv_bc):
        total_res = torch.cat([self.residuals_eq(xy_col), self.residuals_bc(xy_out, xy_bc, uv_bc)], dim = 0)
        return total_res

### Data Preparation ###

x_min = -1.0
x_max = 2.0
y_min = -1.0
y_max = 1.0
r = 0.125
xc = 0.0
yc = 0.0

ub = np.array([x_max, y_max])
lb = np.array([x_min, y_min])


N_b = 100  # inlet & outlet
N_w = 100  # wall
N_s = 100  # surface
N_c = 5000  # collocation
#N_r = 2000  # refining around cyl

def getData():
    # inlet, v=0 & inlet velocity
    inlet_x = np.ones((N_b, 1))*x_min
    inlet_y = np.linspace(y_min, y_max, N_b).reshape(-1,1)
    H = y_max - y_min
    inlet_u =  1 * (-inlet_y**2 + (H/2)**2) / ((H/2)**2)
    inlet_v = np.zeros((N_b, 1))
    inlet_xy = np.concatenate([inlet_x, inlet_y], axis=1)
    inlet_uv = np.concatenate([inlet_u, inlet_v], axis=1)

    # outlet, p=0
    y_outlet = np.linspace(y_min, y_max, N_b).reshape(-1,1)
    x_outlet = np.ones((N_b, 1))*x_max
    xy_outlet = np.concatenate([x_outlet, y_outlet], axis=1)

    # wall, u=v=0
    upwall_x = np.linspace(x_min, x_max, N_w).reshape(-1,1)
    upwall_y = np.ones((N_w, 1))*y_max
    upwall_xy = np.concatenate([upwall_x, upwall_y], axis=1)
    dnwall_y = np.ones((N_w, 1))*y_min
    dnwall_xy = np.concatenate([upwall_x, dnwall_y], axis=1)
    upwall_uv = np.zeros((N_w, 2))
    dnwall_uv = np.zeros((N_w, 2))

    # cylinder surface, u=v=0
    theta = np.linspace(0.0, 2 * np.pi, N_s)
    cyl_x = (r * np.cos(theta) + xc).reshape(-1, 1)
    cyl_y = (r * np.sin(theta) + yc).reshape(-1, 1)
    cyl_xy = np.concatenate([cyl_x, cyl_y], axis=1)
    cyl_uv = np.zeros((N_s, 2))

    # all boundary except outlet
    xy_bnd = np.concatenate([inlet_xy, upwall_xy, dnwall_xy, cyl_xy], axis=0)
    uv_bnd = np.concatenate([inlet_uv, upwall_uv, dnwall_uv, cyl_uv], axis=0)

    # Collocation
    xy_col = lb + (ub - lb) * lhs(2, N_c)

    # # refine points around cylider
    # refine_ub = np.array([xc + 2 * r, yc + 2 * r])
    # refine_lb = np.array([xc - 2 * r, yc - 2 * r])

    # xy_col_refine = refine_lb + (refine_ub - refine_lb) * lhs(2, N_r)
    # xy_col = np.concatenate([xy_col, xy_col_refine], axis=0)

    # remove collocation points inside the cylinder

    dst_from_cyl = np.sqrt((xy_col[:, 0] - xc) ** 2 + (xy_col[:, 1] - yc) ** 2)
    xy_col = xy_col[dst_from_cyl > r].reshape(-1, 2)

    # concatenate all xy for collocation
    #xy_col = np.concatenate((xy_col, xy_bnd, xy_outlet), axis=0)

    # convert to tensor
    xy_bnd = torch.tensor(xy_bnd, dtype=default_dtype).to(device)
    uv_bnd = torch.tensor(uv_bnd, dtype=default_dtype).to(device)
    xy_outlet = torch.tensor(xy_outlet, dtype=default_dtype).to(device)
    xy_col = torch.tensor(xy_col, dtype=default_dtype).to(device).requires_grad_(True)
    return xy_col, xy_bnd, uv_bnd, xy_outlet


xy_col, xy_bnd, uv_bnd, xy_outlet = getData()


def Jacobian_Calc(model, xy_int, xy_out, xy_bc, uv_bc):
    params_dict = {k: v.detach() for k, v in model.named_parameters()}
    def physics_residuals(params, xy_int):
        # Remove Dims for vmaping
        if xy_int.dim() == 1:
            xy_int = xy_int[None]
        # Use the Variable() module to ENFORCE the creation of computational graph (has to be done for compatibility with javrev)
        xy = Variable(xy_int, requires_grad=True)
        # Predict u, v, p from the model
        u, v, p = functional_call(model,params,xy)

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
        momentum_u = model.rho * (u * u_x + v * u_y) + p_x - model.mu * (u_xx + u_yy)  # Shape: [num_points, 1]
        momentum_v = model.rho * (u * v_x + v * v_y) + p_y - model.mu * (v_xx + v_yy)  # Shape: [num_points, 1]


        # Concatenate all residuals along dimension 0
        eq_residuals = torch.cat([continuity, momentum_u, momentum_v], dim=0)
        return eq_residuals
    
    def boundary_residuals(params, xy_bc, uv_bc):
        # Remove Dims for vmaping
        if xy_bc.dim() == 1:
            xy_bc = xy_bc[None]
        if uv_bc.dim() == 1:
            uv_bc = uv_bc[None]

        # torch.nn.utils.vector_to_parameters(params,model.parameters())
        u_bc, v_bc, _ = functional_call(model,params,(xy_bc))

        # Compute residuals for boundary conditions
        bc_residuals = torch.cat([u_bc - uv_bc[:,0:1], v_bc - uv_bc[:,1:]], dim=0)
        return bc_residuals

    def outlet_residuals(params, xy_out):
        # Remove Dims for vmaping
        if xy_out.dim() == 1:
            xy_out = xy_out[None]
        _, _, p_out = functional_call(model,params,(xy_out))
        # Compute residuals for boundary conditions
        out_residuals = p_out
        return out_residuals 
    # Calculate Jacobians using jacrev
    jac_eq_fn = jacrev(physics_residuals)
    jac_eq_fn_v = torch.vmap(jac_eq_fn, (None, 0), chunk_size=15000)
    jac_eq_dict = jac_eq_fn_v(params_dict, xy_int)
    jac_bc_fn = jacrev(boundary_residuals)
    jac_bc_fn_v = torch.vmap(jac_bc_fn, (None,0,0), chunk_size=15000)
    jac_bc_dict = jac_bc_fn_v(params_dict, xy_bc, uv_bc)
    jac_out_fn = jacrev(outlet_residuals)
    jac_out_fn_v = torch.vmap(jac_out_fn, (None,0), chunk_size=15000)
    jac_out_dict = jac_out_fn_v(params_dict, xy_out)

    # Tensorize the dictionaries
    jac_eq = torch.hstack([jac_eq_dict[key].transpose(1,0).flatten(end_dim=1).flatten(start_dim=1) for key in params_dict.keys()])
    jac_bc = torch.hstack([jac_bc_dict[key].transpose(1,0).flatten(end_dim=1).flatten(start_dim=1) for key in params_dict.keys()])
    jac_out = torch.hstack([jac_out_dict[key].transpose(1,0).flatten(end_dim=1).flatten(start_dim=1) for key in params_dict.keys()])

    # Concat the Jacobian from Eqs and BCs
    Jacobian = torch.cat([jac_eq,jac_bc, jac_out],dim = 0)

    return Jacobian

# Levenberg-Marquardt optimizer
def levenberg_marquardt_step(model, xy_col, xy_out, xy_bc, uv_bc, damping):

    # Calculate Jacobian
    J = Jacobian_Calc(model, xy_col, xy_out, xy_bc, uv_bc)

    # Compute the initial residual vector R
    R_physics = model.residuals_eq(xy_col)
    R_boundary = model.residuals_bc(xy_out, xy_bc, uv_bc)
    R = torch.cat([R_physics.reshape(-1), R_boundary.reshape(-1)], dim=0)
    
    # Flatten all parameters into a single vector
    params_vector = torch.nn.utils.parameters_to_vector(model.parameters())

    # Compute Hessian approximation using einsum
    JtJ = torch.einsum('ij,ik->jk', J, J)
    #diag = torch.diagonal(JtJ)
    #D = torch.zeros_like(JtJ)
    #D.diagonal().copy_(diag+0.01)
    D = torch.eye(JtJ.size(0), device=R.device)
    H = JtJ + damping * D
    # Compute Grad(Loss_fn)
    JtR = -torch.einsum('ij,i->j', J, R)
    # Solve the Levenberg equation to calculate the parameters update
    update = torch.linalg.lstsq(H,JtR)[0]

    # Apply the update to the model parameters
    new_params_vector = params_vector + update
    torch.nn.utils.vector_to_parameters(new_params_vector, model.parameters())

    # Compute the new residual vector R_new
    R_new = model.residuals_total(xy_col, xy_out, xy_bc, uv_bc)

    # Calculate the criterion rho
    numerator = torch.norm(R, p=2)**2 - torch.norm(R_new,p=2)**2
    denominator = torch.abs(torch.dot(update, (damping * torch.einsum('ij,i->i',D, update) + torch.einsum('ij,i->j', J, R))))
    crit = numerator / denominator

    # Update the damping parameter based on criterion
    if crit > 1:
        damping = max(damping / 3, 10e-7)
    else:
        damping = min(damping * 2, 10e7)
        # Revert to the original parameters if the update was not successful
        torch.nn.utils.vector_to_parameters(params_vector, model.parameters())

    return damping  # Return the updated damping value

# Training function
def train_lm(model, xy_col, xy_out, xy_bc, uv_bc, num_epochs=1501, damping=1):
    t1 = time.time()
    for epoch in range(num_epochs):
        damping = levenberg_marquardt_step(model, xy_col, xy_out, xy_bc, uv_bc, damping)
        if epoch % 10 == 0:
            # Compute loss
            physics_res = model.residuals_eq(xy_col)
            boundary_res = model.residuals_bc(xy_out, xy_bc, uv_bc)
            loss_pde = torch.sum(physics_res**2)
            loss_bc = torch.sum(boundary_res[:N_b+2*N_w+N_s]**2)
            loss_outlet = torch.sum(boundary_res[N_b+2*N_w+N_s:]**2)
            total_loss = loss_bc + loss_pde + loss_outlet
            model.losses["bc"].append(loss_bc.detach().cpu().item())
            model.losses["pde"].append(loss_pde.detach().cpu().item())
            model.losses["outlet"].append(loss_pde.detach().cpu().item())
            model.epochs.append(epoch)
            t2 = time.time()
            print(
                f"\rEp: {epoch}, t: {t2-t1:.3e}, Damp = {damping:.2e} || Loss: {total_loss.item():.5e} || BC: {loss_bc.item():.3e} || Outlet: {loss_outlet.item():.3e}|| PDE: {loss_pde.item():.3e}",
                end="",
            )
            
            t1 = time.time()
        if epoch % 100 == 0:
            print("")
            torch.save({"model_state_dict": model.state_dict(),
                        "losses": model.losses,
                        "epochs": model.epochs},
                        savepath + "checkpoint.pth")
        if damping >= 9*10e6:
            break

            

if __name__ == "__main__":
    # Create the model
    model = PINN(4,20)
    # Train the model using Levenberg optimizer
    train_lm(model, xy_col, xy_outlet, xy_bnd, uv_bnd)
   
