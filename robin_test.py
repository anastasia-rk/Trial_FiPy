from fipy import Variable, FaceVariable, CellVariable, Grid1D, TransientTerm, DiffusionTerm, Viewer, ImplicitSourceTerm
from fipy.tools import numerix



#%%
##### Model parameters

L= 8.4853e-4 # m boundary layer thickness
dx= 1e-8 # mesh size
nx = int(L/dx)+1 # number of meshes
D = 1e-9 # m^2/s diffusion coefficient
k = 1e-4 # m/s reaction coefficient R = k [c_A],
c_inf =  0. # ROBIN general condition, once can think R = k ([c_A]-[c_inf])
c_init = 1. # Initial concentration of compound A, mol/m^3


#%%
###### Meshing and variable definition

mesh = Grid1D(nx=nx, dx=dx)
c_A = CellVariable(name="c_A", hasOld = True,
                    mesh=mesh,
                    value=c_init)
c_B = CellVariable(name="c_B", hasOld = True,
                    mesh=mesh,
                    value=0.)

#%%
##### Right boundary condition

valueRight = c_init
c_A.constrain(valueRight, mesh.facesRight)
c_B.constrain(0., mesh.facesRight)

#%%
### ROBIN BC requirements, defining cellDistanceVectors
## This code is for fixing celldistance via this link:
## https://stackoverflow.com/questions/60073399/fipy-problem-with-grid2d-celltofacedistancevectors-gives-error-uniformgrid2d
MA = numerix.MA
tmp = MA.repeat(mesh._faceCenters[..., numerix.NewAxis,:], 2, 1)
cellToFaceDistanceVectors = tmp - numerix.take(mesh._cellCenters, mesh.faceCellIDs, axis=1)
tmp = numerix.take(mesh._cellCenters, mesh.faceCellIDs, axis=1)
tmp = tmp[..., 1,:] - tmp[..., 0,:]
cellDistanceVectors = MA.filled(MA.where(MA.getmaskarray(tmp), cellToFaceDistanceVectors[:, 0], tmp))

#%%
##### Defining mask and Robin BC at left boundary
mask = mesh.facesLeft
Gamma0 = D
Gamma = FaceVariable(mesh=mesh, value=Gamma0)
Gamma.setValue(0., where=mask)
dPf = FaceVariable(mesh=mesh,
                   value=mesh._faceToCellDistanceRatio * cellDistanceVectors)
n = mesh.faceNormals
a = FaceVariable(mesh=mesh, value=k, rank=1)
b = FaceVariable(mesh=mesh, value=D, rank=0)
g = FaceVariable(mesh=mesh, value= k * c_inf, rank=0)
RobinCoeff = (mask * Gamma0 * n / (-dPf.dot(a)+b))

#%%
#### Making a plot
viewer = Viewer(vars=(c_A, c_B),
                     datamin=-0.2, datamax=c_init * 1.4)
viewer.plot()

#%% Time step and simulation time definition
time = Variable()
t_simulation = 4 # seconds
timeStepDuration = .05
steps = int(t_simulation/timeStepDuration)

#%% PDE Equations
eqcA = (TransientTerm(var=c_A) == DiffusionTerm(var=c_A, coeff=Gamma) +
            (RobinCoeff * g).divergence
            - ImplicitSourceTerm(var=c_A, coeff=(RobinCoeff * a.dot(-n)).divergence))

eqcB = (TransientTerm(var=c_B) == DiffusionTerm(var=c_B, coeff=Gamma) -
                (RobinCoeff * g).divergence
            + ImplicitSourceTerm(var=c_B, coeff=(RobinCoeff * a.dot(-n)).divergence))


#%% A loop for solving PDE equations
while time() <= (t_simulation):
    time.setValue(time() + timeStepDuration)
    c_B.updateOld()
    c_A.updateOld()
    res1=res2 = 1e10
    viewer.plot()
    while (res1 > 1e-6) & (res2 > 1e-6):
        res1 = eqcA.sweep(var=c_A, dt=timeStepDuration)
        res2 = eqcB.sweep(var=c_B, dt=timeStepDuration)