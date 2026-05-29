import torch
import numpy as np
from pyDOE import lhs
from typing import List, Tuple, Dict
import matplotlib.pyplot as plt
from matplotlib import cm

# Set default device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device = "cpu"
torch.set_default_device(device)
# Set default bit precision !!IMPORTANT!!
default_dtype = torch.float32
torch.set_default_dtype(default_dtype)
# Ensure Reproducability
torch.manual_seed(6969)                                                                  
np.random.seed(6969)

class Domain:
    """
    Class to generate the multiple inlet geometry data
    """
    def __init__(self, domain_lb: Tuple[float,float], domain_ub:Tuple[float,float], 
                 inlets: List[Tuple[str,float,float,float]], outlet: Tuple[str,float,float],
                 N_pde: int, N_w: int, N_inl: int, savepath: str = "Multiple_Inlets/Results/"):
        """
        Description
        -------------------------------------------------
        Define the domain's data.

        Parameters
        -------------------------------------------
        domain_lb: lower bound of the domain (x,y).

        domain_ub: upper bound of the domain (x,y).

        inlets: List of length = # inlets, with a Tuple for each inlet (wall_id,position,width,max_vel)

        --> wall_id = 'L','R','T','B', depending on which wall we want it at.

        --> position: ranges from 0 to 1.

        outlet: Tuple similar to each inlet definition, except for the max_vel.

        U_max: Maximum velocity for the parabolic velocity profile that will be applied to all inlets .

        N_eq, N_w, N_inl: Number of points for PDE, each wall, and each inlet/outlet respectively.

        savepath: Path to save the generated data and the PINN checkpoints.
        """
        self.domain_lb = domain_lb
        self.domain_ub = domain_ub
        self.inlets = inlets
        self.outlet = outlet
        self.N_pde, self.N_w, self.N_inl = N_pde, N_w, N_inl
        self.savepath = savepath
        self.domain_data = self.GenerateGeom()

    def GenerateGeom(self):
        """
        Description
        -------------------------------------------------
        Method to generate all data to be used be the PINN.

        Returns
        -----------------------------------------------
        A dictionary with all geometric and physics data.
        """
        # Translation of input data
        x_min, x_max, y_min, y_max = self.domain_lb[0],self.domain_ub[0], self.domain_lb[1], self.domain_ub[1]
        # Initial creation of the outline shape
        LW_xy = np.random.uniform([x_min,y_min], [x_min,y_max], (self.N_w,2))
        RW_xy = np.random.uniform([x_max,y_min], [x_max,y_max], (self.N_w,2))
        TW_xy = np.random.uniform([x_min,y_max], [x_max,y_max], (self.N_w,2))
        BW_xy = np.random.uniform([x_min,y_min], [x_max,y_min], (self.N_w,2))

        ## Collocation Points Generation
        xy_pde = np.array(self.domain_lb) + (np.array(self.domain_ub) - np.array(self.domain_lb)) * lhs(2, self.N_pde)

        ## Inlets Points Generation ##
        inlets_xy = []
        inlets_uv = []
        for i in range(len(self.inlets)):
            wall = self.inlets[i][0]
            pos = self.inlets[i][1]
            width = self.inlets[i][2]
            max_vel = self.inlets[i][3] 
            y1, y2 = pos*y_max-width/2, pos*y_max+width/2
            x1, x2 = pos*x_max-width/2, pos*x_max+width/2
            # Generation of inlet xy, removing inlet points from the original walls, and calculating velocity profiles. 
            # Be careful with the sign of the velocity that needs to change for opposite walls. Also, u for L/R, v for T/B.
            if wall == 'L':
                inlet_i_xy = np.random.uniform([x_min,y1],[x_min,y2], (self.N_inl,2))
                mask = (LW_xy[:,1] < y1) | (LW_xy[:,1] > y2)
                LW_xy = LW_xy[mask]
                inlet_i_u =  max_vel * (4 * (inlet_i_xy[:,1] - (pos*y_max-width/2)) * (width - (inlet_i_xy[:,1] - (pos*y_max-width/2))) / (width ** 2)).reshape(-1,1)
                inlet_i_v = np.zeros_like(inlet_i_u)
                inlet_i_uv = np.concatenate([inlet_i_u, inlet_i_v], axis=1)
            elif wall == 'R':
                inlet_i_xy = np.random.uniform([x_max,y1],[x_max,y2], (self.N_inl,2))
                mask = (RW_xy[:,1] < y1) | (RW_xy[:,1] > y2)
                RW_xy = RW_xy[mask]
                inlet_i_u = -1 * max_vel * (4 * (inlet_i_xy[:,1] - (pos*y_max-width/2)) * (width - (inlet_i_xy[:,1] - (pos*y_max-width/2))) / (width ** 2)).reshape(-1,1)
                inlet_i_v = np.zeros_like(inlet_i_u)
                inlet_i_uv = np.concatenate([inlet_i_u, inlet_i_v], axis=1)
            elif wall == 'T':
                inlet_i_xy = np.random.uniform([x1,y_max], [x2,y_max], (self.N_inl,2))
                mask = (TW_xy[:,0] < x1) | (TW_xy[:,0] > x2)
                TW_xy = TW_xy[mask]
                inlet_i_v =  -1 * max_vel * (4 * (inlet_i_xy[:,0] - (pos*x_max-width/2)) * (width - (inlet_i_xy[:,0] - (pos*x_max-width/2))) / (width ** 2)).reshape(-1,1)
                inlet_i_u = np.zeros_like(inlet_i_u)
                inlet_i_uv = np.concatenate([inlet_i_u, inlet_i_v], axis=1)
            elif wall == 'B':
                inlet_i_xy = np.random.uniform([x1,y_min], [x2,y_min], (self.N_inl,2))
                mask = (BW_xy[:,0] < x1) | (BW_xy[:,0] > x2)
                BW_xy = BW_xy[mask]
                inlet_i_v =  max_vel * (4 * (inlet_i_xy[:,0] - (pos*x_max-width/2)) * (width - (inlet_i_xy[:,0] - (pos*x_max-width/2))) / (width ** 2)).reshape(-1,1)
                inlet_i_u = np.zeros_like(inlet_i_u)
                inlet_i_uv = np.concatenate([inlet_i_u, inlet_i_v], axis=1)
            # Collect each calculated inlet
            inlets_xy.append(inlet_i_xy)
            inlets_uv.append(inlet_i_uv)
        
        ## Outlet Points Generation ##
        wall, pos, width= self.outlet[0], self.outlet[1], self.outlet[2]
        y1, y2, x1, x2 = pos*y_max-width/2, pos*y_max+width/2, pos*x_max-width/2, pos*x_max+width/2
        if wall == 'L':
            outlet_xy = np.random.uniform([x_min,y1],[x_min,y2], (self.N_inl,2))
            mask = (LW_xy[:,1] < y1) | (LW_xy[:,1] > y2)
            LW_xy = LW_xy[mask]
        elif wall == 'R':
            outlet_xy = np.random.uniform([x_max,y1],[x_max,y2], (self.N_inl,2))
            mask = (RW_xy[:,1] < y1) | (RW_xy[:,1] > y2)
            RW_xy = RW_xy[mask]
        elif wall == 'T':
            outlet_xy = np.random.uniform([x1,y_max], [x2,y_max], (self.N_inl,2))
            mask = (TW_xy[:,0] < x1) | (TW_xy[:,0] > x2)
            TW_xy = TW_xy[mask]
        elif wall == 'B':
            outlet_xy = np.random.uniform([x1,y_min], [x2,y_min], (self.N_inl,2))
            mask = (BW_xy[:,0] < x1) | (BW_xy[:,0] > x2)
            BW_xy = BW_xy[mask]

        ## Final concats and conversion to tensors ##
        # Outlet: Tensorize 
        xy_outlet = torch.tensor(outlet_xy, dtype=default_dtype).to(device)
        
        # Inlets: Concat all uv BCs (inlets and walls) and Tensorize
        inlets_xy = np.concatenate(inlets_xy,axis=0)
        inlets_uv = np.concatenate(inlets_uv,axis=0)
        walls_xy = np.concatenate([LW_xy,RW_xy,TW_xy,BW_xy],axis=0)
        walls_uv = np.zeros_like(walls_xy)

        xy_bc = np.concatenate([inlets_xy, walls_xy], axis=0)
        uv_bc = np.concatenate([inlets_uv, walls_uv], axis=0)

        xy_bc = torch.tensor(xy_bc, dtype=default_dtype).to(device)
        uv_bc = torch.tensor(uv_bc, dtype=default_dtype).to(device)

        # Collocation Points: Tensorize
        xy_pde = torch.tensor(xy_pde, dtype=default_dtype).to(device).requires_grad_(True)

        ## Collect all usefull data in a dictionary ##
        domain_dict = {"xy_pde": xy_pde,
                       "xy_outlet": xy_outlet,
                       "xy_bc": xy_bc,
                       "uv_bc": uv_bc,
                       "inlets": inlets_xy
                        }
        return domain_dict
    
    def PlotDomain(self):
        domain_dict = self.domain_data
        fig, ax, = plt.subplots()
        xy_pde = domain_dict["xy_pde"]
        xy_bc = domain_dict["xy_bc"]
        xy_outlet = domain_dict["xy_outlet"]
        xy_inlets = domain_dict["inlets"]
        scatter1 = ax.scatter(xy_pde[:,0].detach().cpu(),xy_pde[:,1].detach().cpu(), s=10)
        scatter2 = ax.scatter(xy_bc[:,0].detach().cpu(),xy_bc[:,1].detach().cpu(), s=10)
        scatter3 = ax.scatter(xy_outlet[:,0].detach().cpu(),xy_outlet[:,1].detach().cpu(), s=10)
        scatter4 = ax.scatter(xy_inlets[:,0],xy_inlets[:,1], s=10)

        ax.set_aspect('equal')
        ax.set_title("Sampled Points")
        fig.savefig(self.savepath + 'Sampled_Points.png')
