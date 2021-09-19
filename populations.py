from fipy import *
import matplotlib
from numpy import *
import matplotlib.pyplot as plt
import matplotlib.animation as animate
from mpl_toolkits.axes_grid1 import  make_axes_locatable
writegif = animate.PillowWriter(fps=30)
import tqdm
matplotlib.use('TkAgg')
nx = ny = 40
dx = dy = 1.
L = dx*nx
# Diffusion coeff
D_u, D_v, D_T = 1., 2., 4.
Growth_u, Growth_v, Growth_T = 0.05, 0.08, 0.01
Comp_u, Comp_v = 0.02, 0.04
Death_u, Death_v = -0.01, -0.02

def temp_dependency(x):
    sig = 5
    mu = 37
    return exp(-power(x - mu, 2.) / (2 * power(sig, 2.)))

# Initialise a 2d mesh
mesh = Grid2D(dx=dx, dy=dy, nx=nx, ny=ny)
# Create cell variables
u = CellVariable(name="Single population, u", mesh=mesh, hasOld=True)
v = CellVariable(name="Single population, v", mesh=mesh, hasOld=True)
temperature = CellVariable(name="temperature", mesh=mesh, hasOld=True)
DiffCoefU = CellVariable(name="Spacially variable diffusion coeff, u", mesh=mesh, value=D_u, hasOld=True)
DiffCoefV = CellVariable(name="Spacially variable diffusion coeff, v", mesh=mesh, value=D_v, hasOld=True)
DiffCoefT = CellVariable(name="Spacially variable diffusion coeff, T", mesh=mesh, value=D_T, hasOld=True)
DeathCoefU = CellVariable(name="Spacially variable death rate, u", mesh=mesh, value=Death_u, hasOld=True)
DeathCoefV = CellVariable(name="Spacially variable death rate, v", mesh=mesh, value=Death_v, hasOld=True)
GrowthCoefU = CellVariable(name="Spacially variable growth rate, u", mesh=mesh, value=Growth_u, hasOld=True)
GrowthCoefV = CellVariable(name="Spacially variable growth rate, v", mesh=mesh, value=Growth_v, hasOld=True)
GrowthCoefT = CellVariable(name="Spacially variable heating rate, T", mesh=mesh, value=Growth_T, hasOld=True)
# Fixed-gradient boundary condition around the boundary (how to extend this to fixed flux? they are not the same)
X, Y = mesh.cellCenters()
# Initialise temperature with a hot disk
rad = 10
minT = 12.
maxT = 60.
temperature.setValue(maxT, where= ( (X - L/2)**2 + (Y - L/2)**2 < rad**2) )
#  Initialise populations with noise
maxVal = 4.
minVal = 0.
u_init = maxVal*random.rand(nx*ny)
v_init = maxVal*random.rand(nx*ny)
u.setValue(u_init)
v.setValue(v_init)

# Temperature boundary conditions
temperature.constrain(minT, where=mesh.exteriorFaces) # Dirichlet boundary
# temperature.faceGrad.constrain(2 * mesh.faceNormals, where=mesh.exteriorFaces) # Neumann on exterior faces - does not work?
# Zero flux boundary condition
DiffCoefU.constrain(0, where=mesh.exteriorFaces)
DiffCoefV.constrain(0, where=mesh.exteriorFaces)
DeathCoefU.constrain(0, where=mesh.exteriorFaces)
DeathCoefV.constrain(0, where=mesh.exteriorFaces)
#  Decoupled method: define two separate equations that explicitly include other variable
maxCapacity = 4.
grow_u    = GrowthCoefU * ((maxCapacity - u - Comp_v * v)/maxCapacity) * u
grow_v    = GrowthCoefV * ((maxCapacity - v - Comp_u * u)/maxCapacity) * v
grow_T    = GrowthCoefT * (((maxCapacity - u - Comp_v * v)/maxCapacity) * u + ((maxCapacity - v - Comp_u * u)/maxCapacity) * v)
die_u     = DeathCoefU * u * v
die_v     = DeathCoefV * v * u
eq_u = TransientTerm(var=u) == DiffusionTerm(DiffCoefU, var=u) + grow_u + die_u
eq_v = TransientTerm(var=v) == DiffusionTerm(DiffCoefV, var=v) + grow_v + die_v
eq_T = TransientTerm(var=temperature) == DiffusionTerm(DiffCoefT, var=temperature) + grow_T
# vi_u = Viewer(u)

# eqn_u = TransientTerm() == DiffusionTerm(D_u) + compete_u + die_u
# eqn_v = TransientTerm() == DiffusionTerm(D_v) + compete_v + die_v
dt = dx**2/(4*max([D_u, D_v, D_T]))
eqns = eq_u & eq_v & eq_T
vi_u = Viewer(vars=u, datamin=minVal, datamax=3)
vi_v = Viewer(vars=v, datamin=minVal, datamax=3)
vi_T = Viewer(vars=temperature, datamin=minT, datamax=maxT)
steps = 100
resultsAnimate = [[],[],[]]
for t in range(steps):
    GrowthCoefU.setValue(Growth_u * temp_dependency(temperature))
    GrowthCoefV.setValue(Growth_v * temp_dependency(temperature))
    DeathCoefU.setValue(Death_u * (1 - temp_dependency(temperature)))
    DeathCoefV.setValue(Death_v * (1 - temp_dependency(temperature)))
    u.updateOld()
    v.updateOld()
    temperature.updateOld()
    resultsAnimate[0].append(u.numericValue.copy())
    resultsAnimate[1].append(v.numericValue.copy())
    resultsAnimate[2].append(temperature.numericValue.copy())
    eqns.solve(dt=dt)
    vi_u.plot()
    vi_v.plot()
    vi_T.plot()

# Animate
cax_mins = [minVal, minVal, minT]
cax_maxs = [maxVal, maxVal, maxT]
fig, axs = plt.subplots(1, 3, figsize=(25, 8))
names = ['u', 'v', 'T']
# fig.set_tight_layout(True)  # - this is not compatible with colorbar!
# Update for animation
def update(k):
    for iResult in range(3):
        # Reshape the saved array into 2D
        resplot = resultsAnimate[iResult][k].reshape(nx, ny)
        # Plot the heatmap at a given slice
        im = axs[iResult].imshow(resplot.copy(), cmap=plt.get_cmap('jet'), vmin=cax_mins[iResult], vmax=cax_maxs[iResult])
        axs[iResult].set_title(names[iResult])
        divider = make_axes_locatable(axs[iResult])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        fig.colorbar(im, cax=cax)
    return axs

animate_uv = animate.FuncAnimation(fig, update, interval=1, frames=steps, repeat=False)
animate_uv.save("uv_competish_tempr_2D.gif", writer=writegif)