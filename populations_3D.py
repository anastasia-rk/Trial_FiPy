from fipy import *
from numpy import *
from tqdm import tqdm
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animate
from mpl_toolkits.axes_grid1 import  make_axes_locatable
writegif = animate.PillowWriter(fps=30)

matplotlib.use('TkAgg')
nx = ny = nz= 20
dx = dy = dz = 1.
L = dx*nx
# Diffusion coeff
D_u, D_v, D_T = .4, .7, 2.
Growth_u, Growth_v, Growth_T = .05, .08, .01
Comp_u, Comp_v = .02, .04
Death_u, Death_v = .1, .2

def temp_dependency(x):
    sig = 5
    mu = 42
    return exp(-power(x - mu, 2.) / (2 * power(sig, 2.)))

def arrhenius(x,T, E_a):
    gas_const = 8.314446
    T = T + 273.15
    return x*exp(-E_a / (gas_const * T))

def water_activity(moisture_content, T, time):
    #
    # moisture_content - moisture content in parts
    # T - temperature in Kelvin
    # time - time in days
    rate = .01727 + .02613*(.9359**time)
    pow  = .69910 + .41730*(.9434**time)
    T = T + 273.15
    return 1 - exp( - T*rate*(moisture_content**pow))

# Arrhenius plot to check inactivation rates
tempr_plot = arange(10., 100., 10.)
energy_plot = arange(1., 5., 1.)
fig, axs = plt.subplots(2,1)
for e_power in energy_plot:
    energy = 10.**e_power
    rates_u = arrhenius(Death_u, tempr_plot, energy)
    rates_v = arrhenius(Death_v, tempr_plot, energy)
    axs[0].plot(tempr_plot, log(rates_u), label='E_a= %.3f' %(energy))
    axs[1].plot(tempr_plot, log(rates_v), label='E_a= %.3f' %(energy))
axs[0].legend()
axs[0].set_xlabel('T, C')
axs[0].set_ylabel('log(r_d), for u')
axs[1].legend()
axs[1].set_xlabel('T, C')
axs[1].set_ylabel('log(r_d), for v')
plt.show()
plt.savefig('Arrhenius plot')

# Initialise a 2d mesh
mesh = Grid3D(dx=dx, dy=dy, dz=dz, nx=nx, ny=ny, nz=nz)
meshSlice = Grid2D(dx=dx, dy=dy, nx=nx, ny=ny)
# Create cell variables
u = CellVariable(name="Single population, u", mesh=mesh, hasOld=True)
v = CellVariable(name="Single population, v", mesh=mesh, hasOld=True)
temperature = CellVariable(name="temperature", mesh=mesh, hasOld=True)
varSlice = CellVariable(name="Slice", mesh=meshSlice)
# Coefficients spatially varying
DiffCoefU = CellVariable(name="Spacially variable diffusion coeff, u", mesh=mesh, value=D_u, hasOld=True)
DiffCoefV = CellVariable(name="Spacially variable diffusion coeff, v", mesh=mesh, value=D_v, hasOld=True)
DiffCoefT = CellVariable(name="Spacially variable diffusion coeff, T", mesh=mesh, value=D_T, hasOld=True)
DeathCoefU = CellVariable(name="Spacially variable death rate, u", mesh=mesh, value=Death_u, hasOld=True)
DeathCoefV = CellVariable(name="Spacially variable death rate, v", mesh=mesh, value=Death_v, hasOld=True)
GrowthCoefU = CellVariable(name="Spacially variable growth rate, u", mesh=mesh, value=Growth_u, hasOld=True)
GrowthCoefV = CellVariable(name="Spacially variable growth rate, v", mesh=mesh, value=Growth_v, hasOld=True)
GrowthCoefT = CellVariable(name="Spacially variable heating rate, T", mesh=mesh, value=Growth_T, hasOld=True)
# Fixed-gradient boundary condition around the boundary (how to extend this to fixed flux? they are not the same)
X, Y, Z = mesh.cellCenters()
Xslice, Yslice = meshSlice.cellCenters()
# Initialise temperature with a hot disk
rad = 10
minT = 25.
maxT = 70.
temperature.setValue(maxT, where=((X - L/2)**2 + (Y - L/2)**2 + (Z - L/2)**2 < rad**2))
#  Initialise populations with noise
maxVal = 4.
minVal = 0.
u_init = maxVal*random.rand(nx*ny*nz)
v_init = maxVal*random.rand(nx*ny*nz)
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
#  Define sink and source terms as variables
maxCapacity = 4.
grow_u    = GrowthCoefU * ((maxCapacity - u - Comp_v * v)/maxCapacity) * u
grow_v    = GrowthCoefV * ((maxCapacity - v - Comp_u * u)/maxCapacity) * v
grow_T    = GrowthCoefT * (((maxCapacity - u - Comp_v * v)/maxCapacity) * u + ((maxCapacity - v - Comp_u * u)/maxCapacity) * v)
die_u     = DeathCoefU * u
die_v     = DeathCoefV * v
#  PDEs
eq_u = TransientTerm(var=u) == DiffusionTerm(DiffCoefU, var=u) + grow_u - die_u
eq_v = TransientTerm(var=v) == DiffusionTerm(DiffCoefV, var=v) + grow_v - die_v
eq_T = TransientTerm(var=temperature) == DiffusionTerm(DiffCoefT, var=temperature) + grow_T
eqns = eq_u & eq_v & eq_T
zstep = 10
Zslices = arange(0, nz+zstep, zstep).tolist()
steps = 300
dt = dx**3/(6*max([D_u, D_v, D_T]))
resultsU = [[], [], []]
resultsV = [[], [], []]
resultsT = [[], [], []]
for t in tqdm(range(steps)):
    GrowthCoefU.setValue(Growth_u * temp_dependency(temperature))
    GrowthCoefV.setValue(Growth_v * temp_dependency(temperature))
    # simple Gaussian dependency of death rate on temperature
    # DeathCoefU.setValue(Death_u * (1 - temp_dependency(temperature)))
    # DeathCoefV.setValue(Death_v * (1 - temp_dependency(temperature)))
    # Arrhenius model of thermal inactivation rate
    DeathCoefU.setValue(arrhenius(Death_u, temperature, 10**1))
    DeathCoefV.setValue(arrhenius(Death_v, temperature, 10**3))
    u.updateOld()
    v.updateOld()
    temperature.updateOld()
    eqns.solve(dt=dt)
    for i in range(len(Zslices)):
        varSlice.setValue(u((Xslice, Yslice, Zslices[i] * ones(varSlice.mesh.numberOfCells)), order=0))
        resultsU[i].append(varSlice.numericValue.copy())
        varSlice.setValue(v((Xslice, Yslice, Zslices[i] * ones(varSlice.mesh.numberOfCells)), order=0))
        resultsV[i].append(varSlice.numericValue.copy())
        varSlice.setValue(temperature((Xslice, Yslice, Zslices[i] * ones(varSlice.mesh.numberOfCells)), order=0))
        resultsT[i].append(varSlice.numericValue.copy())

# Animate
nSlices = 3
nResults = 3
cax_mins = [minVal-1, minVal-1, minT]
cax_maxs = [maxVal-1, maxVal-1, maxT]
fig, axs = plt.subplots(nSlices, nResults, figsize=(25, 25))
names = ['u', 'v', 'T']
# fig.set_tight_layout(True)  # - this is not compatible with colorbar!
# Update for animation
def update(k):
    for iSlice in range(nSlices):
        # Reshape the saved array into 2D
        resplot = resultsU[iSlice][k].reshape(nx, ny)
        # Plot the heatmap at a given slice
        im = axs[0, iSlice].imshow(resplot.copy(), cmap=plt.get_cmap('jet'), vmin=cax_mins[0], vmax=cax_maxs[0])
        axs[0, iSlice].set_title(names[0]+', Z={:.1f}'.format(Zslices[iSlice]))
        divider = make_axes_locatable(axs[0, iSlice])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        fig.colorbar(im, cax=cax)
        resplot = resultsV[iSlice][k].reshape(nx, ny)
        # Plot the heatmap at a given slice
        im = axs[1, iSlice].imshow(resplot.copy(), cmap=plt.get_cmap('jet'), vmin=cax_mins[1],
                                    vmax=cax_maxs[1])
        axs[1, iSlice].set_title(names[1]+', Z={:.1f}'.format(Zslices[iSlice]))
        divider = make_axes_locatable(axs[1, iSlice])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        fig.colorbar(im, cax=cax)
        resplot = resultsT[iSlice][k].reshape(nx, ny)
        # Plot the heatmap at a given slice
        im = axs[2, iSlice].imshow(resplot.copy(), cmap=plt.get_cmap('jet'), vmin=cax_mins[2],
                                    vmax=cax_maxs[2])
        axs[2, iSlice].set_title(names[2]+', Z={:.1f}'.format(Zslices[iSlice]))
        divider = make_axes_locatable(axs[2, iSlice])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        fig.colorbar(im, cax=cax)
    return axs

animate_uv = animate.FuncAnimation(fig, update, interval=1, frames=steps, repeat=False)
animate_uv.save("uv_competish_tempr_3D.gif", writer=writegif)