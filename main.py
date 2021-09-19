# This is a sample Python script to test FiPy package

#%% Example 1 - Diffusion on a 20x20 mesh
from fipy import CellVariable, Grid2D, Viewer, TransientTerm, ImplicitDiffusionTerm
from fipy.tools import numerix
import matplotlib
matplotlib.use('TkAgg')
# %% Definitions
nx = ny = 80
dx = dy = 1.
L = dx*nx
# Diffusion coeff
D = 1.
# Initialise a 2d mesh
mesh = Grid2D(dx=dx, dy=dy, nx=nx, ny=ny)
# Create a cell variable and initialise at 0
phi = CellVariable(name="Diffusion from the corner", mesh=mesh, value=2.)
# Create diffusion equation
eq = TransientTerm() == ImplicitDiffusionTerm(coeff=D)
# Dirichlet boundary conditions in two corners
valueTopLeft = 0.
# valueBottomLeft = 0
# valueTopRight = 0
valueBottomRight = 5.
# this automatically sets Neumann boundary condirions for the other two corners
X, Y = mesh.faceCenters
facesTopLeft = ((mesh.facesLeft & (Y > L/2)) | (mesh.facesTop & (X < L/2)))
facesBottomRight = ((mesh.facesRight & (Y < L/2)) | (mesh.facesBottom & (X > L/2)))
# facesBottomLeft = ((mesh.facesLeft & (Y < L/2)) | (mesh.facesBottom & (X < L/2)))
# facesTopRight = ((mesh.facesRight & (Y > L/2)) | (mesh.facesTop & (X > L/2)))
phi.constrain(valueTopLeft, facesTopLeft)
phi.constrain(valueBottomRight, facesBottomRight)
# phi.constrain(valueBottomLeft, facesBottomLeft)
# phi.constrain(valueTopRight, facesTopRight)
# Create a viewer
if __name__ == '__main__':
    viewer = Viewer(vars=phi, datamin=0, datamax=5)
    viewer.plot()

# Loop in time
timeStepDuration = (dy**2)*(dx**2)/(2*D*(dx**2 + dy**2))
steps = 5000
from builtins import range
for step in range(steps):
    eq.solve(var=phi, dt=timeStepDuration)
    if __name__ == '__main__':
        viewer.plot()


# See PyCharm help at https://www.jetbrains.com/help/pycharm/
