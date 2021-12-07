# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import Functions as f

##############################################################################
# Custom conditions

## Initial conditions
dt = 1e-4 # Temporal Step
D = 1 # Diffusion coefficient

## Boundaries on the sphere
# Reflectant boundary
theta_min = 50
theta_max = 122
phi_min = 0
phi_max = 360
reg_refl = np.array([[theta_min,theta_max],[phi_min,phi_max]]) * np.pi/180  

# Absorbing boundary
theta_pole = 86
phi_pole = 90
delta = 19/2 # Max arc length of the absorbing patch
reg_abs = (np.array([theta_pole*np.pi/180,phi_pole*np.pi/180]),delta*np.pi/180) 

##############################################################################

## Simulation
pos0_3d = f.Initial_Position_Sphere(reg_refl, reg_abs)  # Initial position on the sphere
t,pos = f.Diff_Sphere_Pos(reg_refl,reg_abs,dt, D,pos0_3d)

## Plot
ax = plt.axes(projection='3d')

# Reflectant rigion
theta,phi = np.mgrid[reg_refl[0][0]: reg_refl[0][1]:30j,
                      reg_refl[1][0]: reg_refl[1][1]:30j]
x,y,z = f.Sph2Cart(theta, phi)
ax.plot_wireframe(x, y, z, color = 'b',alpha = 0.3)

# Absorbent region
f.plot_abs_patch(reg_abs,ax)
#Particle
ax.plot3D(pos[0],pos[1],pos[2],'r-.')
plt.axis(False)
plt.grid(False)
plt.show()


