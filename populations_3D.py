from setup import *
from fipy import *
from tqdm import tqdm
from scipy.constants import Stefan_Boltzmann
import matplotlib.animation as animate
from mpl_toolkits.axes_grid1 import  make_axes_locatable
writegif = animate.PillowWriter(fps=6)

# Heap geometry and grid
nx = ny = nz = 20
dx = dy = dz = 1.
L = dx*nx
# Dynamical model coefficients
D_u, D_v, D_w, D_p, D_r, D_T = .65, .75, .31, .42, .15, 4
Growth_u, Growth_v, Growth_w, Growth_p, Growth_r, Growth_T = 1.314, 1.606, 1.911, 0.998, 1.5, .6 # hour^-1 , first three from combase, FE from Bologna sausage?
Comp_uv, Comp_uw, Comp_up, Comp_ur = 1.2, 1.1, 1.4, 1.5
Comp_vu, Comp_vw, Comp_vp, Comp_vr = 1.6, 1.1, 1.6, 1.5
Comp_wu, Comp_wv, Comp_wp, Comp_wr = 2.1, 2.2, 2.1, 1.5
Comp_pu, Comp_pv, Comp_pw, Comp_pr = 1.6, 2.4, 1.2, 1.5
Comp_ru, Comp_rv, Comp_rw, Comp_rp = 1.1, 1.1, 1.1, 1.1
Death_u, Death_v, Death_w, Death_p, Death_r = .1, .1, .1, .1, .1

# Integration settings
zstep   = 10
Zslices = arange(0, nz+zstep, zstep).tolist()
steps   = 50
dt      = dx**3/(6*max([D_u, D_v, D_w, D_p, D_r, D_T]))

# Measured parameters of composting windrow
tempr_core      = array([21., 30., 58., 61., 63., 65., 64., 63., 61., 60.,\
                    61., 62., 61., 60., 59., 57.,  51., 42., 35., 32.,\
                    30., 29., 27., 25., 25., 25.])
days_tempr      = array([1, 3, 5, 7, 9, 10, 12, 13, 14, 15,\
                    16, 18, 20, 22, 27, 29, 33, 39, 46, 49,\
                    54, 61, 63, 70, 84, 110])
ph_compsoting   = array([7.8, 8.1, 8.05, 8.0, 7.9, 7.5,\
                       7.5, 7.5, 7.4, 7.4, 7.3, 7.2, 7.2, 7.1, 7.1])
days_ph     = array([1, 3, 6, 8, 14, 21, 28, 35, 42, 49, 56, 63, 70, 84, 110])
moist_core  = array([60., 57., 45., 44., 60., 54., 51., 41., 37., 36., 28., 26.])/100
days_moist  = array([1, 8, 14, 21, 35, 42, 49, 56, 63, 70, 84, 110])

# Animation settings
nSlices  = 3 # number of slices along z-axis
nResults = 6 # number of population plots + temperature plots
minVal   = 0. # min population level for colorbars
maxVal   = 30. # max population level for colorbars
maxValAll = 50.
minT     = 21. # min temperature (ambient)
maxT     = 65. # max temperature (in the core)
cax_mins = [minVal, minVal, minVal, minVal, minVal, minT] #colorbar mins
cax_maxs = [maxVal, maxVal, maxVal, maxVal, maxValAll, maxT] #colorbar maxes

# Initialise meshes
mesh = Grid3D(dx=dx, dy=dy, dz=dz, nx=nx, ny=ny, nz=nz)
meshSlice = Grid2D(dx=dx, dy=dy, nx=nx, ny=ny)
# Create cell variables
u = CellVariable(name="$E.coli$", mesh=mesh, hasOld=True)
v = CellVariable(name="$C.perfringens", mesh=mesh, hasOld=True)
w = CellVariable(name="$B.cereus$", mesh=mesh, hasOld=True)
p = CellVariable(name="$E.faecium$", mesh=mesh, hasOld=True)
r = CellVariable(name="The rest", mesh=mesh, hasOld=True)
temperature = CellVariable(name="Temperature", mesh=mesh, hasOld=True)
varSlice = CellVariable(name="Slice", mesh=meshSlice)
# Coefficients are spatially varying
DiffCoefU = CellVariable(name="Spacially variable diffusion coeff, u", mesh=mesh, value=D_u, hasOld=True)
DiffCoefV = CellVariable(name="Spacially variable diffusion coeff, v", mesh=mesh, value=D_v, hasOld=True)
DiffCoefW = CellVariable(name="Spacially variable diffusion coeff, w", mesh=mesh, value=D_w, hasOld=True)
DiffCoefP = CellVariable(name="Spacially variable diffusion coeff, p", mesh=mesh, value=D_p, hasOld=True)
DiffCoefR = CellVariable(name="Spacially variable diffusion coeff, r", mesh=mesh, value=D_r, hasOld=True)
DiffCoefT = CellVariable(name="Spacially variable diffusion coeff, T", mesh=mesh, value=D_T, hasOld=True)
DeathCoefU = CellVariable(name="Spacially variable death rate, u", mesh=mesh, value=Death_u, hasOld=True)
DeathCoefV = CellVariable(name="Spacially variable death rate, v", mesh=mesh, value=Death_v, hasOld=True)
DeathCoefW = CellVariable(name="Spacially variable death rate, w", mesh=mesh, value=Death_w, hasOld=True)
DeathCoefP = CellVariable(name="Spacially variable death rate, p", mesh=mesh, value=Death_p, hasOld=True)
DeathCoefR = CellVariable(name="Spacially variable death rate, r", mesh=mesh, value=Death_r, hasOld=True)
GrowthCoefU = CellVariable(name="Spacially variable growth rate, u", mesh=mesh, value=Growth_u, hasOld=True)
GrowthCoefV = CellVariable(name="Spacially variable growth rate, v", mesh=mesh, value=Growth_v, hasOld=True)
GrowthCoefW = CellVariable(name="Spacially variable growth rate, w", mesh=mesh, value=Growth_w, hasOld=True)
GrowthCoefP = CellVariable(name="Spacially variable growth rate, p", mesh=mesh, value=Growth_p, hasOld=True)
GrowthCoefR = CellVariable(name="Spacially variable growth rate, r", mesh=mesh, value=Growth_r, hasOld=True)
GrowthCoefT = CellVariable(name="Spacially variable heating rate, T", mesh=mesh, value=Growth_T, hasOld=True)
# Fixed-gradient boundary condition around the boundary (how to extend this to fixed flux? they are not the same)
X, Y, Z = mesh.cellCenters()
Xslice, Yslice = meshSlice.cellCenters()
# Initialise tempr in the heap in the heap
rad = 5
temperature.setValue(minT+5)
# temperature.setValue(tempr_core[0], where=((X - L/2)**2 + (Y - L/2)**2 + (Z - L/2)**2 < rad**2))
# Initialise populations with noise
maxVal = 10.
u_init = maxVal*random.rand(nx*ny*nz)
v_init = maxVal*random.rand(nx*ny*nz)
w_init = maxVal*random.rand(nx*ny*nz)
p_init = maxVal*random.rand(nx*ny*nz)
r_init = 2*maxVal*random.rand(nx*ny*nz)
u.setValue(u_init)
v.setValue(v_init)
w.setValue(w_init)
p.setValue(p_init)
r.setValue(r_init)
# Temperature boundary conditions
# temperature.constrain(minT, where=mesh.exteriorFaces) # Dirichlet boundary
conv_rate = 0.01
convectionCoeff = FaceVariable(mesh=mesh, value=[conv_rate])
convectionCoeff[..., mesh.facesDown.value] = 0.
# constraint_value = FaceVariable(mesh=mesh) # create a facevar for constraint value and update on each sweep (inside the loop)
# temperature.faceGrad.constrain( constraint_value, where=mesh.exteriorFaces) # the constraint value is then fed into Robin const for solving
# temperature.faceGrad.constrain(2 * mesh.faceNormals, where=mesh.exteriorFaces) # Neumann on exterior faces - does not work?

# Zero flux boundary condition
DiffCoefU.constrain(0, where=mesh.exteriorFaces)
DiffCoefV.constrain(0, where=mesh.exteriorFaces)
DiffCoefW.constrain(0, where=mesh.exteriorFaces)
DiffCoefP.constrain(0, where=mesh.exteriorFaces)
DiffCoefR.constrain(0, where=mesh.exteriorFaces)
DeathCoefU.constrain(0, where=mesh.exteriorFaces)
DeathCoefV.constrain(0, where=mesh.exteriorFaces)
DeathCoefW.constrain(0, where=mesh.exteriorFaces)
DeathCoefP.constrain(0, where=mesh.exteriorFaces)
DeathCoefR.constrain(0, where=mesh.exteriorFaces)
GrowthCoefU.constrain(0, where=mesh.exteriorFaces)
GrowthCoefV.constrain(0, where=mesh.exteriorFaces)
GrowthCoefW.constrain(0, where=mesh.exteriorFaces)
GrowthCoefP.constrain(0, where=mesh.exteriorFaces)
GrowthCoefR.constrain(0, where=mesh.exteriorFaces)
#  Define sink and source terms as variables
maxCapacity = 100.
grow_u    = ((maxCapacity - u - Comp_uv * v - Comp_uw * w - Comp_up * p - Comp_ur * r)/maxCapacity) * u
grow_v    = ((maxCapacity - v - Comp_vu * u - Comp_vw * w - Comp_vp * p - Comp_vr * r)/maxCapacity) * v
grow_w    = ((maxCapacity - w - Comp_wu * u - Comp_wv * v - Comp_wp * p - Comp_wr * r)/maxCapacity) * w
grow_p    = ((maxCapacity - p - Comp_pu * u - Comp_pv * v - Comp_pw * w - Comp_pr * r)/maxCapacity) * p
grow_r    = ((maxCapacity - r - Comp_ru * u - Comp_rv * v - Comp_rw * w - Comp_rp * p)/maxCapacity) * r
grow_T    = (GrowthCoefU/(1+DeathCoefU)) * grow_u + (GrowthCoefV/(1+DeathCoefV))* grow_v +\
            (GrowthCoefW/(1+DeathCoefW)) * grow_w + (GrowthCoefP/(1+DeathCoefP)) * grow_p + (GrowthCoefR/(1+DeathCoefR)) * grow_r
die_u     = DeathCoefU * u
die_v     = DeathCoefV * v
die_w     = DeathCoefW * w
die_p     = DeathCoefP * p
die_r     = DeathCoefR * r
#  PDEs
eq_u = TransientTerm(var=u) == DiffusionTerm(DiffCoefU, var=u) + GrowthCoefU * grow_u - die_u
eq_v = TransientTerm(var=v) == DiffusionTerm(DiffCoefV, var=v) + GrowthCoefV * grow_v - die_v
eq_w = TransientTerm(var=w) == DiffusionTerm(DiffCoefW, var=w) + GrowthCoefW * grow_w - die_w
eq_p = TransientTerm(var=p) == DiffusionTerm(DiffCoefP, var=p) + GrowthCoefP * grow_p - die_p
eq_r = TransientTerm(var=r) == DiffusionTerm(DiffCoefP, var=r) + GrowthCoefR * grow_r - die_r
eq_T = TransientTerm(var=temperature) == DiffusionTerm(DiffCoefT, var=temperature) + grow_T - (conv_rate  * mesh.facesLeft).divergence
eqns = eq_u & eq_v & eq_w & eq_p & eq_r & eq_T
resultsU = [[], [], []]
resultsV = [[], [], []]
resultsW = [[], [], []]
resultsP = [[], [], []]
resultsR = [[], [], []]
resultsT = [[], [], []]
for t in tqdm(range(steps)):
    t_current = t*dt
    # convert all rates to day^{-1}
    ph_level = interp(t_current, days_ph, ph_compsoting)*ones_like(temperature.value)
    t_core   = interp(t_current, days_tempr, tempr_core)
    a_w      = water_activity(interp(t_current, days_moist, moist_core), temperature.value, t)
    GrowthCoefU.setValue(Growth_u * cardinal_model(temperature.value, 40.3, 5.6,  47.3) * cardinal_model(ph_level, 7.0, 4.0, 10.0) * mcmeekin_model(a_w, .95))  # e coli
    # GrowthCoefV.setValue(Growth_v * cardinal_model(temperature.value, 35.3, 10.0, 48.8) * cardinal_model(ph_level, 7.0, 4.6, 8.0) * mcmeekin_model(a_w, .935)) # clostridium botilinum type A
    GrowthCoefV.setValue(Growth_v * cardinal_model(temperature.value, 45.0, 10.0, 48.8) * cardinal_model(ph_level, 7.0, 5.5, 9.0) * mcmeekin_model(a_w, .93)) # clostridium perfringens
    GrowthCoefW.setValue(Growth_w * cardinal_model(temperature.value, 31.9, 10.0, 50.0) * cardinal_model(ph_level, 7.0, 4.35, 9.3) * mcmeekin_model(a_w, .912)) # b. cereus
    GrowthCoefP.setValue(Growth_p * cardinal_model(temperature.value, 42.0, 0.1, 53.5) * cardinal_model(ph_level, 7.0, 5.0, 9.6) * mcmeekin_model(a_w, .97)) # e. faecium
    GrowthCoefR.setValue(Growth_r * cardinal_model(temperature.value, 42.0, 0.1, 60.5) * cardinal_model(ph_level, 7.0, 5.0, 10.0) * mcmeekin_model(a_w, .97)) # the rest - slower growing but larger range
    # Arrhenius model of thermal inactivation rate
    # Thermal inactivation of e coli as modelled in Cerf et.al. 1996, rate is in seconds:
    Death_ecol     = 3600*cerf_model(temperature.value, ph_level, a_w, [86.49, -.3028 * (10 ** 5), -.5470, .0494, 3.067])
    Death_bacillis = 60*mafart_model_aw(temperature.value, ph_level, a_w, 0.676, 100, [9.28, 4.08, 0.164])
    # these two arbitrary for now
    # Death_clostr   = mafart_model_aw(temperature.value, ph_level, a_w, 0.000000045, 100, [7.97, 6.19, 0.125]) # clost bot type A
    Death_clostr = 60*mafart_model_aw(temperature.value, ph_level, a_w, 0.95, 100, [10.05, 6.19, 0.125]) # clostr perfringens
    Death_fe     = 60*mafart_model_aw(temperature.value, ph_level, a_w, 0.796, 100,[12.86, 4.19, 0.185]) # enterococcus
    Death_rest   = 1200* mafart_model_aw(temperature.value, ph_level, a_w,  1.996, 100, [7.86, 5.39, 1.087])  # the rest
    DeathCoefU.setValue(Death_ecol)
    DeathCoefV.setValue(Death_clostr)
    DeathCoefW.setValue(Death_bacillis)
    DeathCoefP.setValue(Death_fe)
    DeathCoefR.setValue(Death_rest)
    # t_core = interp(t_current, days_tempr, tempr_core)
    # temperature.setValue(t_core, where=((X - L / 2) ** 2 + (Y - L / 2) ** 2 + (Z - L / 2) ** 2 < rad ** 2))
    u.updateOld()
    v.updateOld()
    w.updateOld()
    p.updateOld()
    r.updateOld()
    temperature.updateOld()
    eqns.solve(dt=dt)
    for i in range(len(Zslices)):
        varSlice.setValue(u((Xslice, Yslice, Zslices[i] * ones(varSlice.mesh.numberOfCells)), order=0))
        resultsU[i].append(varSlice.numericValue.copy())
        varSlice.setValue(v((Xslice, Yslice, Zslices[i] * ones(varSlice.mesh.numberOfCells)), order=0))
        resultsV[i].append(varSlice.numericValue.copy())
        varSlice.setValue(w((Xslice, Yslice, Zslices[i] * ones(varSlice.mesh.numberOfCells)), order=0))
        resultsW[i].append(varSlice.numericValue.copy())
        varSlice.setValue(p((Xslice, Yslice, Zslices[i] * ones(varSlice.mesh.numberOfCells)), order=0))
        resultsP[i].append(varSlice.numericValue.copy())
        varSlice.setValue(r((Xslice, Yslice, Zslices[i] * ones(varSlice.mesh.numberOfCells)), order=0))
        resultsR[i].append(varSlice.numericValue.copy())
        varSlice.setValue(temperature((Xslice, Yslice, Zslices[i] * ones(varSlice.mesh.numberOfCells)), order=0))
        resultsT[i].append(varSlice.numericValue.copy())

# Animate

fig, axs = plt.subplots(nSlices, nResults, figsize=(27, 18))
names = ['$E.coli$', '$C.perfringens$', '$B.cereus$', '$E.faecium$','The rest', 'T, $^{\circ}C$']
# fig.set_tight_layout(True)  # - this is not compatible with colorbar!
# Update for animation
def update(k):
    for iSlice in range(nSlices):
        # Reshape the saved array into 2D
        resplot = resultsU[iSlice][k].reshape(nx, ny)
        # Plot the heatmap at a given slice
        im = axs[iSlice,0].imshow(resplot.copy(), cmap=plt.get_cmap('jet'), vmin=cax_mins[0], vmax=cax_maxs[0])
        axs[iSlice,0].set_title(names[0]+', Z={:.1f}'.format(Zslices[iSlice]))
        divider = make_axes_locatable(axs[iSlice,0])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        fig.colorbar(im, cax=cax)
        # Plot the heatmap at a given slice
        resplot = resultsV[iSlice][k].reshape(nx, ny)
        im = axs[iSlice,1].imshow(resplot.copy(), cmap=plt.get_cmap('jet'), vmin=cax_mins[1],
                                    vmax=cax_maxs[1])
        axs[iSlice,1].set_title(names[1]+', Z={:.1f}'.format(Zslices[iSlice]))
        divider = make_axes_locatable(axs[iSlice,1])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        fig.colorbar(im, cax=cax)
        # Plot the heatmap at a given slice
        resplot = resultsW[iSlice][k].reshape(nx, ny)
        im = axs[iSlice,2].imshow(resplot.copy(), cmap=plt.get_cmap('jet'), vmin=cax_mins[2], vmax=cax_maxs[2])
        axs[iSlice,2].set_title(names[2] + ', Z={:.1f}'.format(Zslices[iSlice]))
        divider = make_axes_locatable(axs[iSlice,2])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        fig.colorbar(im, cax=cax)
        # Plot the heatmap at a given slice
        resplot = resultsP[iSlice][k].reshape(nx, ny)
        im = axs[iSlice,3].imshow(resplot.copy(), cmap=plt.get_cmap('jet'), vmin=cax_mins[3],
                                   vmax=cax_maxs[3])
        axs[iSlice,3].set_title(names[3] + ', Z={:.1f}'.format(Zslices[iSlice]))
        divider = make_axes_locatable(axs[iSlice,3])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        fig.colorbar(im, cax=cax)
        # Plot the heatmap at a given slice
        resplot = resultsR[iSlice][k].reshape(nx, ny)
        im = axs[iSlice,4].imshow(resplot.copy(), cmap=plt.get_cmap('jet'), vmin=cax_mins[4],
                                    vmax=cax_maxs[4])
        axs[iSlice,4].set_title(names[4]+', Z={:.1f}'.format(Zslices[iSlice]))
        divider = make_axes_locatable(axs[iSlice,4])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        fig.colorbar(im, cax=cax)
        resplot = resultsT[iSlice][k].reshape(nx, ny)
        im = axs[iSlice,5].imshow(resplot.copy(), cmap=plt.get_cmap('jet'), vmin=cax_mins[4],
                                    vmax=cax_maxs[5])
        axs[iSlice,5].set_title(names[5]+', Z={:.1f}'.format(Zslices[iSlice]))
        divider = make_axes_locatable(axs[iSlice,5])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        fig.colorbar(im, cax=cax)
    return axs

animate_uv = animate.FuncAnimation(fig, update, interval=1, frames=steps, repeat=False)
animate_uv.save("5pops_robin.gif", writer=writegif)
# writervideo = animate.FFMpegWriter(fps=12)
# animate_uv.save("5pops_robin.mp4", writer=writervideo)