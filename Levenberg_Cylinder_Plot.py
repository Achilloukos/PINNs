import torch
from Levenberg_Cylinder import PINN, x_min, x_max, y_min, y_max, xc, yc, r, xy_col, xy_bnd, uv_bnd, N_b, savepath
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

pinn = PINN()
checkpoint = torch.load(savepath + "checkpoint.pth")
pinn.load_state_dict(checkpoint["model_state_dict"])

N = 300

x = np.linspace(x_min,x_max,N)
y = np.linspace(y_min, y_max, N)
X, Y = np.meshgrid(x, y)
x = X.reshape(-1, 1)
y = Y.reshape(-1, 1)
xy = np.concatenate([x, y], axis=1)
xy = torch.tensor(xy).to(device)

y_in = np.linspace(y_min,y_max,N).reshape(-1,1)
x_in = x_min * np.ones_like(y_in).reshape(-1,1)
xy_in = np.concatenate([x_in, y_in], axis=1)
xy_in = torch.tensor(xy_in).to(device)

y_out = np.linspace(y_min,y_max,N).reshape(-1,1)
x_out = x_max * np.ones_like(y_out).reshape(-1,1)
xy_out = np.concatenate([x_out, y_out], axis=1)
xy_out = torch.tensor(xy_out).to(device)

dst_from_cyl = np.sqrt((x - xc) ** 2 + (y - yc) ** 2)
cyl_mask = dst_from_cyl > r

with torch.no_grad():
    u, v, p = pinn(xy)
    u = u.cpu().numpy()
    u = np.where(cyl_mask, u, np.nan).reshape(Y.shape)
    v = v.cpu().numpy()
    v = np.where(cyl_mask, v, np.nan).reshape(Y.shape)
    p = p.cpu().numpy()
    p = np.where(cyl_mask, p, np.nan).reshape(Y.shape)
    u_in, v_in, _= pinn(xy_in)
    u_in = u_in.cpu().numpy()
    v_in = v_in.cpu().numpy()
    u_out, _, p_out = pinn(xy_out)
    u_out = u_out.cpu().numpy()
    p_out = p_out.cpu().numpy()


fig1, ax1, = plt.subplots(figsize=(12,5))
contour1 = ax1.contourf(x.reshape(N,N), y.reshape(N,N), p.reshape(N,N), cmap=cm.jet)
ax1.set_aspect('equal')
fig1.colorbar(contour1, ax=ax1)
ax1.set_title(f"Pressure Field")
fig1.savefig(savepath + 'Pressure_Field.png')
# plt.figure()
# plt.contourf(x.reshape(N,N), y.reshape(N,N), u.reshape(N,N), cmap=cm.jet)
fig2, ax2, = plt.subplots(figsize=(12,5))
contour2 = ax2.contourf(x.reshape(N,N), y.reshape(N,N), u.reshape(N,N), cmap=cm.jet)
ax2.set_aspect('equal')
fig2.colorbar(contour2, ax=ax2)
ax2.set_title(f"u Field")
fig2.savefig(savepath + 'u_Field.png')
# plt.figure()
# plt.contourf(x.reshape(N,N), y.reshape(N,N), v.reshape(N,N), cmap=cm.jet)
fig3, ax3, = plt.subplots(figsize=(12,5))
contour3 = ax3.contourf(x.reshape(N,N), y.reshape(N,N), v.reshape(N,N), cmap=cm.jet)
ax3.set_aspect('equal')
fig3.colorbar(contour3, ax=ax3)
ax3.set_title(f"v Field")
fig3.savefig(savepath + 'v_Field.png')
plt.figure()
plt.subplot(1,2,1)
plt.plot(y_in, u_in)
plt.title("u-profile at inlet")
plt.subplot(1,2,2)
plt.plot(y_out, p_out)
plt.title("p-profile at outlet")
plt.savefig(savepath + 'Inlet_Outlet.png')
fig4, ax4, = plt.subplots(figsize=(8,5))
scatte4 = ax4.scatter(xy_col[:,0].detach().cpu(),xy_col[:,1].detach().cpu(), s=3)
ax4.set_aspect('equal')
ax4.set_title("Sampled Points")
fig4.savefig(savepath + 'Sampled_Points.png')


def plotLoss(epochs_cnt, losses_dict, path, info=["BC", "Outlet", "PDE"]):
        fig, axes = plt.subplots(1, 3, sharex=True, sharey=True, figsize=(10, 6))
        axes[0].set_yscale("log")
        for i, j in zip(range(3), info):
            axes[i].plot(epochs_cnt, losses_dict[j.lower()])
            axes[i].set_title(j)
        plt.show()
        fig.savefig(path)
plotLoss(checkpoint["epochs"], checkpoint["losses"], savepath + "loss_curve.png", ["BC", "Outlet", "PDE"])
