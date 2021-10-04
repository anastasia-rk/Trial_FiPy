from setup import *
from fipy import *
from tqdm import tqdm
import matplotlib.animation as animate
from mpl_toolkits.axes_grid1 import  make_axes_locatable
writegif = animate.PillowWriter(fps=6)
nx = ny = nz = 20
dx = dy = dz = 1.
L = dx*nx
# Diffusion coeff
D_u, D_v, D_w, D_p, D_T = .15, .25, .31, .2, 4
Growth_u, Growth_v, Growth_w, Growth_p, Growth_T = 1.0, 1.0, 1.0, 1.0, 1.0 # hour^-1
Comp_uv, Comp_uw, Comp_up = .2, .4, .4
Comp_vu, Comp_vw, Comp_vp = .3, .3, .3
Comp_wu, Comp_wv, Comp_wp = .2, .2, .3
Comp_pu, Comp_pv, Comp_pw = .3, .2, .3
Death_u, Death_v, Death_w, Death_p = .1, .05, .03, .01

# Measured parameters of composting windrow
tempr_core = array([21., 30., 58., 61., 63., 65., 64., 63., 61., 60.,\
                    61., 62., 61., 60., 59., 57.,  51., 42., 35., 32.,\
                    30., 29., 27., 25., 25., 25.])
days_tempr = array([1, 3, 5, 7, 9, 10, 12, 13, 14, 15,\
                    16, 18, 20, 22, 27, 29, 33, 39, 46, 49,\
                    54, 61, 63, 70, 84, 110])
ph_compsoting = array([7.8, 8.1, 8.05, 8.0, 7.9, 7.5,\
                       7.5, 7.5, 7.4, 7.4, 7.3, 7.2, 7.2, 7.1, 7.1])
days_ph = array([1, 3, 6, 8, 14, 21, 28, 35, 42, 49, 56, 63, 70, 84, 110])
moist_core = array([60., 57., 45., 44., 60., 54., 51., 41., 37., 36., 28., 26.])/100
days_moist = array([1, 8, 14, 21, 35, 42, 49, 56, 63, 70, 84, 110])

# Initialise meshes
mesh = Grid3D(dx=dx, dy=dy, dz=dz, nx=nx, ny=ny, nz=nz)
meshSlice = Grid2D(dx=dx, dy=dy, nx=nx, ny=ny)
# Create cell variables
u = CellVariable(name="$E.coli$", mesh=mesh, hasOld=True)
v = CellVariable(name="$C.botulinum$", mesh=mesh, hasOld=True)
w = CellVariable(name="$B.cereus$", mesh=mesh, hasOld=True)
p = CellVariable(name="$E.faecium$", mesh=mesh, hasOld=True)
temperature = CellVariable(name="Temperature", mesh=mesh, hasOld=True)
varSlice = CellVariable(name="Slice", mesh=meshSlice)
# Coefficients are spatially varying
DiffCoefU = CellVariable(name="Spacially variable diffusion coeff, u", mesh=mesh, value=D_u, hasOld=True)
DiffCoefV = CellVariable(name="Spacially variable diffusion coeff, v", mesh=mesh, value=D_v, hasOld=True)
DiffCoefW = CellVariable(name="Spacially variable diffusion coeff, w", mesh=mesh, value=D_w, hasOld=True)
DiffCoefP = CellVariable(name="Spacially variable diffusion coeff, p", mesh=mesh, value=D_p, hasOld=True)
DiffCoefT = CellVariable(name="Spacially variable diffusion coeff, T", mesh=mesh, value=D_T, hasOld=True)
DeathCoefU = CellVariable(name="Spacially variable death rate, u", mesh=mesh, value=Death_u, hasOld=True)
DeathCoefV = CellVariable(name="Spacially variable death rate, v", mesh=mesh, value=Death_v, hasOld=True)
DeathCoefW = CellVariable(name="Spacially variable death rate, w", mesh=mesh, value=Death_w, hasOld=True)
DeathCoefP = CellVariable(name="Spacially variable death rate, p", mesh=mesh, value=Death_p, hasOld=True)
GrowthCoefU = CellVariable(name="Spacially variable growth rate, u", mesh=mesh, value=Growth_u, hasOld=True)
GrowthCoefV = CellVariable(name="Spacially variable growth rate, v", mesh=mesh, value=Growth_v, hasOld=True)
GrowthCoefW = CellVariable(name="Spacially variable growth rate, w", mesh=mesh, value=Growth_w, hasOld=True)
GrowthCoefP = CellVariable(name="Spacially variable growth rate, p", mesh=mesh, value=Growth_p, hasOld=True)
GrowthCoefT = CellVariable(name="Spacially variable heating rate, T", mesh=mesh, value=Growth_T, hasOld=True)
# Fixed-gradient boundary condition around the boundary (how to extend this to fixed flux? they are not the same)
X, Y, Z = mesh.cellCenters()
Xslice, Yslice = meshSlice.cellCenters()
# Initialise temperature with a hot disk
rad = 5
minT = 21.
maxT = 65.
temperature.setValue(minT)
temperature.setValue(tempr_core[0], where=((X - L/2)**2 + (Y - L/2)**2 + (Z - L/2)**2 < rad**2))
#  Initialise populations with noise
maxVal = 10.
minVal = 0.
u_init = maxVal*random.rand(nx*ny*nz)
v_init = maxVal*random.rand(nx*ny*nz)
w_init = maxVal*random.rand(nx*ny*nz)
p_init = maxVal*random.rand(nx*ny*nz)
u.setValue(u_init)
v.setValue(v_init)
w.setValue(w_init)
p.setValue(p_init)
# Temperature boundary conditions
temperature.constrain(minT, where=mesh.exteriorFaces) # Dirichlet boundary
# temperature.faceGrad.constrain(2 * mesh.faceNormals, where=mesh.exteriorFaces) # Neumann on exterior faces - does not work?
# Zero flux boundary condition
DiffCoefU.constrain(0, where=mesh.exteriorFaces)
DiffCoefV.constrain(0, where=mesh.exteriorFaces)
DiffCoefW.constrain(0, where=mesh.exteriorFaces)
DiffCoefP.constrain(0, where=mesh.exteriorFaces)
DeathCoefU.constrain(0, where=mesh.exteriorFaces)
DeathCoefV.constrain(0, where=mesh.exteriorFaces)
DeathCoefW.constrain(0, where=mesh.exteriorFaces)
DeathCoefP.constrain(0, where=mesh.exteriorFaces)
GrowthCoefU.constrain(0, where=mesh.exteriorFaces)
GrowthCoefV.constrain(0, where=mesh.exteriorFaces)
GrowthCoefW.constrain(0, where=mesh.exteriorFaces)
GrowthCoefP.constrain(0, where=mesh.exteriorFaces)
#  Define sink and source terms as variables
maxCapacity = 12.
grow_u    = ((maxCapacity - u - Comp_uv * v - Comp_uw * w - Comp_up * p)/maxCapacity) * u
grow_v    = ((maxCapacity - v - Comp_vu * u - Comp_vw * w - Comp_vp * p)/maxCapacity) * v
grow_w    = ((maxCapacity - w - Comp_wu * u - Comp_wv * v - Comp_wp * p)/maxCapacity) * w
grow_p    = ((maxCapacity - p - Comp_pu * u - Comp_pv * v - Comp_pw * w)/maxCapacity) * p
grow_T    = grow_u + grow_v + grow_w + grow_p
die_u     = DeathCoefU * u
die_v     = DeathCoefV * v
die_w     = DeathCoefU * w
die_p     = DeathCoefV * p
#  PDEs
eq_u = TransientTerm(var=u) == DiffusionTerm(DiffCoefU, var=u) + GrowthCoefU * grow_u - die_u
eq_v = TransientTerm(var=v) == DiffusionTerm(DiffCoefV, var=v) + GrowthCoefV * grow_v - die_v
eq_w = TransientTerm(var=w) == DiffusionTerm(DiffCoefW, var=w) + GrowthCoefW * grow_w - die_w
eq_p = TransientTerm(var=p) == DiffusionTerm(DiffCoefP, var=p) + GrowthCoefP * grow_p - die_p
eq_T = TransientTerm(var=temperature) == DiffusionTerm(DiffCoefT, var=temperature) + GrowthCoefT * grow_T
eqns = eq_u & eq_v & eq_w & eq_p & eq_T
zstep = 10
Zslices = arange(0, nz+zstep, zstep).tolist()
steps = 126
dt = dx**3/(6*max([D_u, D_v, D_w, D_p, D_T]))
resultsU = [[], [], []]
resultsV = [[], [], []]
resultsW = [[], [], []]
resultsP = [[], [], []]
resultsT = [[], [], []]
for t in tqdm(range(steps)):
    t_current = t*dt
    # convert all rates to day^{-1}
    ph_level = interp(t_current, days_ph, ph_compsoting)*ones_like(temperature.value)
    t_core   = interp(t_current, days_tempr, tempr_core)
    a_w      = water_activity(interp(t_current, days_moist, moist_core), temperature.value, t)
    GrowthCoefU.setValue(Growth_u * cardinal_model(temperature.value, 40.3, 5.6,  47.3) * cardinal_model(ph_level, 7, 4, 10) * mcmeekin_model(a_w, .95))  # e coli
    GrowthCoefV.setValue(Growth_v * cardinal_model(temperature.value, 39.3, 11.0, 45.8) * cardinal_model(ph_level, 7, 4.6, 9) * mcmeekin_model(a_w, .935)) # clostridium type A
    GrowthCoefW.setValue(Growth_w * cardinal_model(temperature.value, 40.1, 4.1, 50.0) * cardinal_model(ph_level, 7, 4.9, 9.3) * mcmeekin_model(a_w, .92)) # b. cereus
    GrowthCoefP.setValue(Growth_p * cardinal_model(temperature.value, 42.0, 0.1, 53.5) * cardinal_model(ph_level, 7, 5, 9.6) * mcmeekin_model(a_w, .97)) # e. faecium
    # Arrhenius model of thermal inactivation rate
    # Thermal inactivation of e coli as modelled in Cerf et.al. 1996, rate is in seconds:
    Death_ecol = 3600*cerf_model(temperature.value, ph_level, a_w, [86.49, -.3028 * (10 ** 5), -.5470, .0494, 3.067])
    Death_bacillis = 60*mafart_model_aw(temperature.value, ph_level, a_w, 0.676, 100, [9.28, 4.08, 0.164])
    Death_clostr   = mafart_model_aw(temperature.value, ph_level, a_w, 0.000000045, 100, [7.97, 6.19, 0.125])
    Death_fe       = mafart_model_aw(temperature.value, ph_level, a_w, 0.796, 100, [12.86, 4.19, 0.185])
    DeathCoefU.setValue(Death_ecol)
    DeathCoefV.setValue(Death_clostr)
    DeathCoefW.setValue(Death_bacillis)
    DeathCoefP.setValue(Death_fe)
    u.updateOld()
    v.updateOld()
    w.updateOld()
    p.updateOld()
    temperature.updateOld()
    eqns.solve(dt=dt)
    t_core = interp(t_current, days_tempr, tempr_core)
    temperature.setValue(t_core, where=((X - L / 2) ** 2 + (Y - L / 2) ** 2 + (Z - L / 2) ** 2 < rad ** 2))
    for i in range(len(Zslices)):
        varSlice.setValue(u((Xslice, Yslice, Zslices[i] * ones(varSlice.mesh.numberOfCells)), order=0))
        resultsU[i].append(varSlice.numericValue.copy())
        varSlice.setValue(v((Xslice, Yslice, Zslices[i] * ones(varSlice.mesh.numberOfCells)), order=0))
        resultsV[i].append(varSlice.numericValue.copy())
        varSlice.setValue(w((Xslice, Yslice, Zslices[i] * ones(varSlice.mesh.numberOfCells)), order=0))
        resultsW[i].append(varSlice.numericValue.copy())
        varSlice.setValue(p((Xslice, Yslice, Zslices[i] * ones(varSlice.mesh.numberOfCells)), order=0))
        resultsP[i].append(varSlice.numericValue.copy())
        varSlice.setValue(temperature((Xslice, Yslice, Zslices[i] * ones(varSlice.mesh.numberOfCells)), order=0))
        resultsT[i].append(varSlice.numericValue.copy())

# Animate
nSlices  = 3
nResults = 5
cax_mins = [minVal, minVal, minVal, minVal, minT]
cax_maxs = [maxVal-1, maxVal-1, maxVal-1, maxVal-1, maxT]
fig, axs = plt.subplots(nSlices, nResults, figsize=(27, 18))
names = ['$E.coli$', '$C.botulinum$', '$B.cereus$', '$E.faecium$', 'T, $^{\circ}C$']
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
        resplot = resultsT[iSlice][k].reshape(nx, ny)
        im = axs[iSlice,4].imshow(resplot.copy(), cmap=plt.get_cmap('jet'), vmin=cax_mins[4],
                                    vmax=cax_maxs[4])
        axs[iSlice,4].set_title(names[4]+', Z={:.1f}'.format(Zslices[iSlice]))
        divider = make_axes_locatable(axs[iSlice,4])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        fig.colorbar(im, cax=cax)
    return axs

animate_uv = animate.FuncAnimation(fig, update, interval=1, frames=steps, repeat=False)
animate_uv.save("real_growths_3D.gif", writer=writegif)
writervideo = animate.FFMpegWriter(fps=12)
animate_uv.save("real_growths_3D.mp4", writer=writervideo)