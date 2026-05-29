from Pinn import *
from torch.func import jacrev, functional_call  # Updated to use torch.func
from torch.autograd import Variable
import time

class Training:
    """
    Class for training the PINN model by performing Levenberg optimization steps.
    """
    
    def __init__(self, domain: Domain, pinn: PINN, num_epochs: int = 5000, damping: float = 1):
        self.model = pinn
        self.domain = domain
        self.damping = damping
        self.num_epochs = num_epochs

    def JacobianCalculation(self):
        """Computes the Jacobian of ALL residuals with respect to the model parameters."""
        
        def physics_residuals(params: Dict, xy_eq: torch.Tensor):
            """Computes the PDE residuals for vmap-ed inputs, to be used for the Jacobian calculation."""
            if xy_eq.dim() == 1:
                xy_eq = xy_eq[None]
            # Use the Variable() module to ENFORCE the creation of computational graph (has to be done for compatibility with javrev)
            xy = Variable(xy_eq, requires_grad=True)
            # Predict u, v, p from the model
            u, v, p = functional_call(self.model,params,xy)

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
            momentum_u = self.model.rho * (u * u_x + v * u_y) + p_x - self.model.mu * (u_xx + u_yy)  # Shape: [num_points, 1]
            momentum_v = self.model.rho * (u * v_x + v * v_y) + p_y - self.model.mu * (v_xx + v_yy)  # Shape: [num_points, 1]

            # Concatenate all residuals along dimension 0
            eq_residuals = torch.cat([continuity, momentum_u, momentum_v], dim=0)
            return (1/np.sqrt(self.domain.N_pde)) * eq_residuals
        
        def boundary_residuals(params: Dict, xy_bc: torch.Tensor, uv_bc: torch.Tensor):
            """Computes the uv-BC residuals for vmap-ed inputs, to be used for the Jacobian calculation."""
            if xy_bc.dim() == 1:
                xy_bc = xy_bc[None]
            if uv_bc.dim() == 1:
                uv_bc = uv_bc[None]

            u_bc, v_bc, _ = functional_call(self.model,params,(xy_bc))

            # Compute residuals for boundary conditions
            bc_residuals = torch.cat([u_bc - uv_bc[:,0:1], v_bc - uv_bc[:,1:]], dim=0)
            return (1/np.sqrt(self.domain.domain_data['xy_bc'].shape[0])) * bc_residuals

        def outlet_residuals(params, xy_out):
            """Computes the outlet residuals (p-BC) for vmap-ed inputs, to be used for the Jacobian calculation."""
            if xy_out.dim() == 1:
                xy_out = xy_out[None]
            _, _, p_out = functional_call(self.model,params,(xy_out))
            out_residuals = torch.cat([p_out], dim=0)
            return (1/np.sqrt(self.domain.domain_data['xy_outlet'].shape[0])) * out_residuals
            
        ## Get the model parameters as a dictionary ##
        params_dict = {k: v.detach() for k, v in self.model.named_parameters()}
        
        ## Calculate the Jacobians using jacrev in combination with vmap (for vectorization of the process) ## 
        # PDE Jacobian
        jac_eq_fn = jacrev(physics_residuals)
        jac_eq_fn_v = torch.vmap(jac_eq_fn, (None, 0), chunk_size=15000)
        jac_eq_dict = jac_eq_fn_v(params_dict, self.domain.domain_data["xy_pde"])
        # BCs Jacobian
        jac_bc_fn = jacrev(boundary_residuals)
        jac_bc_fn_v = torch.vmap(jac_bc_fn, (None,0,0), chunk_size=15000)
        jac_bc_dict = jac_bc_fn_v(params_dict, self.domain.domain_data["xy_bc"], self.domain.domain_data["uv_bc"])
        # Outlet Jacobian
        jac_out_fn = jacrev(outlet_residuals)
        jac_out_fn_v = torch.vmap(jac_out_fn, (None,0), chunk_size=15000)
        jac_out_dict = jac_out_fn_v(params_dict, self.domain.domain_data["xy_outlet"])
        # Tensorize the dictionaries
        jac_eq = torch.hstack([jac_eq_dict[key].transpose(1,0).flatten(end_dim=1).flatten(start_dim=1) for key in params_dict.keys()])
        jac_bc = torch.hstack([jac_bc_dict[key].transpose(1,0).flatten(end_dim=1).flatten(start_dim=1) for key in params_dict.keys()])
        jac_out = torch.hstack([jac_out_dict[key].transpose(1,0).flatten(end_dim=1).flatten(start_dim=1) for key in params_dict.keys()])
        # Concat the Jacobian from Eqs and BCs
        Jacobian = torch.cat([jac_eq,jac_bc,jac_out],dim = 0)

        return Jacobian
    
    def LevebergStep(self):
        """
        Performs a single Levenberg optimization step to update the model parameters.
        """
        # Calculate Jacobian
        J = self.JacobianCalculation()

        # Compute the initial residual vector R
        R_physics = self.model.residuals_eq(self.domain)
        R_boundary = self.model.residuals_bc(self.domain)
        R = torch.cat([R_physics.reshape(-1), R_boundary.reshape(-1)], dim=0)

        # Flatten all parameters into a single vector
        params_vector = torch.nn.utils.parameters_to_vector(self.model.parameters())

        # Compute Hessian approximation using einsum
        JtJ = torch.einsum('ij,ik->jk', J, J)
        D = torch.eye(JtJ.size(0), device=R.device)
        H = JtJ + self.damping * D
        JtR = -torch.einsum('ij,i->j', J, R)
        #HtH = torch.einsum('ij,ik->jk', H, H)
        #H_inv = torch.inverse(HtH) @ H.permute(1,0)
        # Solve for the update step
        #update = torch.einsum('ij,j->i', H_inv, R_grad)
        update = torch.linalg.lstsq(H,JtR)[0]

        # Apply the update to the model parameters
        new_params_vector = params_vector + update
        torch.nn.utils.vector_to_parameters(new_params_vector, self.model.parameters())

        # Compute the new residual vector R_new
        R_new = self.model.residuals_total(self.domain)

        # Calculate the criterion
        numerator = torch.norm(R, p=2)**2 - torch.norm(R_new,p=2)**2
        #denominator = torch.abs(torch.dot(update, (damping * update + torch.einsum('ij,i->j', J, R))))
        denominator = torch.abs(torch.dot(update, (self.damping * torch.einsum('ij,i->i',D, update) + torch.einsum('ij,i->j', J, R))))
        crit = numerator / denominator

        # Update the damping parameter based on criterion
        if crit > 1:
            self.damping = max(self.damping / 3, 10e-7)
        else:
            self.damping = min(self.damping * 2, 10e7)
            # Revert to the original parameters if the update was not successful
            torch.nn.utils.vector_to_parameters(params_vector, self.model.parameters())

        return self.damping  # Return the updated damping value
    
    def TrainThePINN(self):
        t1 = time.time()
        # self.domain.domain_data = self.domain.GenerateGeom()
        for epoch in range(self.num_epochs):
            # Resample Points
            if epoch % 50 == 0:
                self.domain.domain_data = self.domain.GenerateGeom()
            self.damping = self.LevebergStep()
            if epoch % 10 == 0:
                
                # Compute loss
                self.model.loss(self.domain)

                self.model.epochs.append(epoch)
                t2 = time.time()
                print(
                    f"\rEp: {epoch}, t: {t2-t1:.3e}, Damp = {self.damping:.2e} || Loss: {self.model.losses['total'][-1]:.5e} || BC: {self.model.losses['bc'][-1]:.3e} || PDE: {self.model.losses['pde'][-1]:.3e}",
                    end="",
                )
                
                t1 = time.time()
            if epoch % 100 == 0:
                print("")
                torch.save({"model_state_dict": self.model.state_dict(),
                            "losses": self.model.losses,
                            "epochs": self.model.epochs},
                            self.domain.savepath + "checkpoint.pth")
                self.PlotResults()

            if self.damping >= 9*10e6:
                break

    def PlotResults(self):
        checkpoint = torch.load(self.domain.savepath + "checkpoint.pth")
        self.model.load_state_dict(checkpoint["model_state_dict"])
        x_min, x_max, y_min, y_max = self.domain.domain_lb[0], self.domain.domain_ub[0], self.domain.domain_lb[1], self.domain.domain_ub[1]
        N = 300

        x = np.linspace(x_min,x_max,N)
        y = np.linspace(y_min, y_max, N)
        X, Y = np.meshgrid(x, y)
        x = X.reshape(-1, 1)
        y = Y.reshape(-1, 1)
        xy = np.concatenate([x, y], axis=1)
        xy = torch.tensor(xy, dtype=default_dtype).to(device)



        with torch.no_grad():
            ## Output Field Calculation ##
            u, v, p = self.model(xy)
            u = u.cpu().numpy()
            v = v.cpu().numpy()
            p = p.cpu().numpy()
            ## Inlet Mass Flow ##
            mdot_in = 0
            for inlet in self.domain.inlets:
                wall, pos, width = inlet[0], inlet[1], inlet[2]
                y1, y2 = pos*y_max-width/2, pos*y_max+width/2
                x1, x2 = pos*x_max-width/2, pos*x_max+width/2

                if wall == 'L':
                    xy_in = torch.tensor(np.linspace([x_min,y1],[x_min,y2],N), dtype=default_dtype).to(device)
                    u_in, v_in, _= self.model(xy_in)
                    u_in, v_in = u_in.cpu().numpy(), v_in.cpu().numpy()
                    mdot_in_i = +1 * self.model.rho * np.sum(u_in) * width/N
                elif wall == 'R':
                    xy_in = torch.tensor(np.linspace([x_max,y1],[x_max,y2],N), dtype=default_dtype).to(device)
                    u_in, v_in, _= self.model(xy_in)
                    u_in, v_in = u_in.cpu().numpy(), v_in.cpu().numpy()
                    mdot_in_i = -1 * self.model.rho * np.sum(u_in) * width/N
                elif wall == 'T':
                    xy_in = torch.tensor(np.linspace([x1,y_max],[x2,y_max],N), dtype=default_dtype).to(device)
                    u_in, v_in, _= self.model(xy_in)
                    u_in, v_in = u_in.cpu().numpy(), v_in.cpu().numpy()
                    mdot_in_i = -1 * self.model.rho * np.sum(v_in) * width/N
                elif wall == 'B':
                    xy_in = torch.tensor(np.linspace([x1,y_min],[x2,y_min],N), dtype=default_dtype).to(device)
                    u_in, v_in, _= self.model(xy_in)
                    u_in, v_in = u_in.cpu().numpy(), v_in.cpu().numpy()
                    mdot_in_i = +1 * self.model.rho * np.sum(v_in) * width/N
                
                mdot_in += mdot_in_i 
            ## Outlet Mass Flow ##    
            wall, pos, width = self.domain.outlet[0], self.domain.outlet[1], self.domain.outlet[2]
            y1, y2 = pos*y_max-width/2, pos*y_max+width/2
            x1, x2 = pos*x_max-width/2, pos*x_max+width/2

            if wall == 'L':
                xy_out = torch.tensor(np.linspace([x_min,y1],[x_min,y2],N), dtype=default_dtype).to(device)
                u_out, v_out, _= self.model(xy_out)
                u_out, v_out = u_out.cpu().numpy(), v_out.cpu().numpy()
                mdot_out = self.model.rho * np.sum(u_out) * width/N
            elif wall == 'R':
                xy_out = torch.tensor(np.linspace([x_max,y1],[x_max,y2],N), dtype=default_dtype).to(device)
                u_out, v_out, _= self.model(xy_out)
                u_out, v_out = u_out.cpu().numpy(), v_out.cpu().numpy()
                mdot_out = -1 * self.model.rho * np.sum(u_out) * width/N
            elif wall == 'T':
                xy_out = torch.tensor(np.linspace([x1,y_max],[x2,y_max],N), dtype=default_dtype).to(device)
                u_out, v_out, _= self.model(xy_out)
                u_out, v_out = u_out.cpu().numpy(), v_out.cpu().numpy()
                mdot_out = -1 * self.model.rho * np.sum(v_out) * width/N
            elif wall == 'B':
                xy_out = torch.tensor(np.linspace([x1,y_min],[x2,y_min],N), dtype=default_dtype).to(device)
                u_out, v_out, _= self.model(xy_out)
                u_out, v_out = u_out.cpu().numpy(), v_out.cpu().numpy()
                mdot_out = -1 * self.model.rho * np.sum(v_out) * width/N
                
                

        def draw_wall(color1='r-', color2='k-', color3='g-'):
            plt.plot([x_min, x_min], [y_min, y_max], color1, lw=5)  # left wall
            plt.plot([x_min, x_max], [y_max, y_max], color1, lw=5) # top wall
            plt.plot([x_max,x_max], [y_max, y_min], color1, lw=5) # right wall
            plt.plot([x_max, x_min], [y_min, y_min], color1, lw=5) # bottom wall
            for inlet in self.domain.inlets:
                wall, pos, width = inlet[0], inlet[1], inlet[2]
                y1, y2 = pos*y_max-width/2, pos*y_max+width/2
                x1, x2 = pos*x_max-width/2, pos*x_max+width/2
                if wall == 'L':
                    plt.plot([x_min,x_min],[y1,y2],color2, lw=8)
                elif wall == 'R':
                    plt.plot([x_max,x_max],[y1,y2],color2, lw=8)
                elif wall == 'T':
                    plt.plot([x1,x2],[y_max,y_max],color2, lw=8)
                elif wall == 'B':
                    plt.plot([x1,x2],[y_min,y_min],color2, lw=8)
            wall, pos, width = self.domain.outlet[0], self.domain.outlet[1], self.domain.outlet[2]
            y1, y2 = pos*y_max-width/2, pos*y_max+width/2
            x1, x2 = pos*x_max-width/2, pos*x_max+width/2
            if wall == 'L':
                plt.plot([x_min,x_min],[y1,y2],color3, lw=8)
            elif wall == 'R':
                plt.plot([x_max,x_max],[y1,y2],color3, lw=8)
            elif wall == 'T':
                plt.plot([x1,x2],[y_max,y_max],color3, lw=8)
            elif wall == 'B':
                plt.plot([x1,x2],[y_min,y_min],color3, lw=8)
            
            plt.gca().set_aspect('equal')
             

        # plt.figure()
        # plt.contourf(x.reshape(N,N), y.reshape(N,N), p.reshape(N,N), cmap=cm.jet)
        fig1, ax1, = plt.subplots()
        contour1 = ax1.contourf(x.reshape(N,N), y.reshape(N,N), p.reshape(N,N), cmap=cm.jet)
        ax1.set_aspect('equal')
        draw_wall()
        fig1.colorbar(contour1, ax=ax1)
        ax1.set_title(f"Pressure Field")
        fig1.savefig(self.domain.savepath + 'Pressure_Field.png')
        # plt.figure()
        # plt.contourf(x.reshape(N,N), y.reshape(N,N), u.reshape(N,N), cmap=cm.jet)
        fig2, ax2, = plt.subplots()
        contour2 = ax2.contourf(x.reshape(N,N), y.reshape(N,N), u.reshape(N,N), cmap=cm.jet)
        ax2.set_aspect('equal')
        draw_wall()
        fig2.colorbar(contour2, ax=ax2)
        ax2.set_title(f"u Field")
        fig2.savefig(self.domain.savepath + 'u_Field.png')
        # plt.figure()
        # plt.contourf(x.reshape(N,N), y.reshape(N,N), v.reshape(N,N), cmap=cm.jet)
        fig3, ax3, = plt.subplots()
        contour3 = ax3.contourf(x.reshape(N,N), y.reshape(N,N), v.reshape(N,N), cmap=cm.jet)
        ax3.set_aspect('equal')
        draw_wall()
        fig3.colorbar(contour3, ax=ax3)
        ax3.set_title(f"v Field")
        fig3.savefig(self.domain.savepath + 'v_Field.png')


        plt.figure()
        norm = (u**2 + v**2)**0.5
        plt.streamplot(x.reshape(N,N), y.reshape(N,N), u.reshape(N,N), v.reshape(N,N), color=norm.reshape(N,N))
        plt.title(f"Streamlines -> Mass Flow: in {mdot_in:.3e}, out {mdot_out:.3e}")
        draw_wall()
        plt.savefig(self.domain.savepath + "Streamplot.png")

        def plotLoss(epochs_cnt, losses_dict, path, info=["total", "pde", "bc"]):
                fig, axes = plt.subplots(1, 3, sharex=True, sharey=True, figsize=(10, 6))
                axes[0].set_yscale("log")
                for i, j in zip(range(3), info):
                    axes[i].plot(epochs_cnt, losses_dict[j.lower()])
                    axes[i].set_title(j)
                plt.show()
                fig.savefig(path)
        plotLoss(checkpoint["epochs"], checkpoint["losses"], self.domain.savepath + "loss_curve.png", ["total", "pde", "bc"])
        plt.close()

    
