from Domain import *
from Pinn import *
from Training import *

size_factor = 25

x_min = 0.0 * size_factor
x_max = 0.1 * size_factor
y_min = 0.0 * size_factor
y_max = 0.1 * size_factor

ub = (x_max, y_max)
lb = (x_min, y_min)

wid = y_max/4

# in1 = ('L',0.5,wid,0.4)
# in2 = ('B',0.25,wid,0.4)
# in3 = ('B',0.75,wid,0.4)
# inlets = [in1,in2,in3]
# outlet = ('R',0.5,wid)

in1 = ('L',0.65,wid,0.4)
inlets = [in1]
outlet = ('B',0.65,wid)

pts_factor = 0.3
N_pde, N_bc, N_out = round(pts_factor*3000), round(pts_factor*300), round(pts_factor*150)

savepath = "Multiple_Inlets/Results/"

dom = Domain(lb,ub,inlets,outlet,N_pde,N_bc,N_out,savepath)

dom.PlotDomain()

pinn = PINN()

training = Training(dom, pinn)
training.TrainThePINN()
