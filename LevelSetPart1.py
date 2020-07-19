import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

phi = np.zeros((120,120))

for i in range(0,120):
    for j in range(0,120):
        x = 0.05*(i-60)
        y=0.05*(j-60)
        phi[i,j] = 3*(1-x)**2.*np.exp(-(x**2) - (y+1)**2)- 10*(x/5 - x**3 - y**5)*np.exp(-x**2-y**2)- 1/3*np.exp(-(x+1)**2 - y**2)

# phi = np.random.randn(20, 20) # initial value for phi
F = 2 # some function
dt = 0.01
it = 100
x_values = np.linspace(-10,10,120)
y_values = np.linspace(-10,10,120)
X,Y = np.meshgrid(x_values,y_values)
plt.figure(1)
ax = plt.gca(projection='3d')
ax.plot_surface(X,Y,phi,cmap='jet')
ax.contour3D(X,Y,phi,1,cmap='jet')

plt.figure(2)
plt.contour(phi, 0)
# plt.show()
plt.figure(3)
for i in range(it):
    dphi = np.array(np.gradient(phi))
    dphi_norm = np.sqrt(np.sum(dphi**2, axis=0))

    phi = phi + dt * F * dphi_norm

    # plot the zero level curve of phi
plt.contour(phi, 0)

plt.show()