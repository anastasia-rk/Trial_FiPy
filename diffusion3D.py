from fipy import *
from numpy import *
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.animation as animate
from tqdm import tqdm
writegif = animate.PillowWriter(fps=30)
# %% Definitions
nx = ny = nz = 30
dx = dy = dz = 1.
L = dx*nx
# Diffusion coeff
D = 2.
# Initialise a 3d mesh and a 2d slice mesh
meshAll = Grid3D(dx=dx, dy=dy, dz=dz, nx=nx, ny=ny, nz=nz)
meshSlice = Grid2D(dx=dx, dy=dy, nx=nx, ny=ny)
# Create a cell variable and the placeholder for the slice variable
phiAll = CellVariable(name="All", mesh=meshAll, value=1.)
phiSlice = CellVariable(name="Slice", mesh=meshSlice)
# Create diffusion equation
eq = TransientTerm() == ImplicitDiffusionTerm(coeff=D)
# Dirichlet boundary conditions in two opposite corners of the cube
minVal = 0.
maxVal = 2.
# this automatically sets Neumann boundary condirions for the other corners (apparently)
X_face, Y_face, Z_face = meshAll.faceCenters()
facesTopLeftFront = ((meshAll.facesLeft & (Y_face > L/2) & (Z_face < L/2)) | (meshAll.facesTop & (X_face < L/2) & (Z_face <L/2)) | \
                     (meshAll.facesFront & (X_face < L/2) & (Y_face > L/2)) )
facesBottomRightBack = ((meshAll.facesRight & (Y_face < L/2) & (Z_face > L/2)) | (meshAll.facesBottom & (X_face > L/2) & (Z_face > L/2)) | \
                     (meshAll.facesBack & (X_face > L/2) & (Y_face < L/2)) )
phiAll.constrain(minVal, where=facesTopLeftFront)
phiAll.constrain(maxVal, where=facesBottomRightBack)
# Create a viewer
# if __name__ == '__main__':
#     viewer = Viewer(vars=phiSlice, datamin=minVal, datamax=maxVal):
#     viewer.plot()
# Get cell centre locations to copy slices into the 2D plceholder:
X, Y, Z = meshAll.cellCenters()
Xslice, Yslice = meshSlice.cellCenters()


# Loop in time
timeStepDuration = dx**3/(6*D)
steps = 120
zstep = 10
Zslices = arange(0, nz+zstep, zstep).tolist()
# Set a dummy variable to save simulation results and plot them later (this is for animation):
resultsAnimate = []
for i in range(len(Zslices)):
    resultsAnimate.append([])
    phiSlice.setValue(phiAll((Xslice, Yslice, Zslices[i] * ones(phiSlice.mesh.numberOfCells)), order=0))
    resultsAnimate[i].append(phiSlice.numericValue.copy())

for step in tqdm(range(steps)):
    eq.solve(var=phiAll, dt=timeStepDuration)
    for i in range(len(Zslices)):
        phiSlice.setValue(phiAll((Xslice, Yslice, Zslices[i] * ones(phiSlice.mesh.numberOfCells)), order=0))
        resultsAnimate[i].append(phiSlice.numericValue.copy())
    # if __name__ == '__main__':
    #     viewer.plot()

#  Animate slices:
numRaws = 1
numColumns = (len(Zslices) // numRaws + (len(Zslices) % numRaws > 0))  # this is a round-up number for setting the right number of axes
# Initialise the fiugure
fig, axs = plt.subplots(numRaws, numColumns, figsize=(15, 10))
# fig.set_tight_layout(True)  # - this is not compatible with colorbar!
# Update for animation
def update(k):
    for iResult in range(len(Zslices)):
        # Reshape the saved array into 2D
        resplot = resultsAnimate[iResult][k].reshape(nx, ny)
        # Plot the heatmap at a given slice
        im = axs[iResult].imshow(resplot.copy(), cmap=plt.get_cmap('jet'), vmin=minVal, vmax=maxVal)
        axs[iResult].set_title('Z={:.1f}'.format(Zslices[iResult]))
    fig.subplots_adjust(right=0.85)
    colorbar_all = fig.add_axes([0.9, 0.1, 0.03, 0.8])
    fig.colorbar(im, cax=colorbar_all)
    return axs

animSlices = animate.FuncAnimation(fig, update, interval=1, frames=steps, repeat=False)
animSlices.save("try_animation.gif", writer=writegif)







