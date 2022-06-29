# -*- coding: utf-8 -*-
from time import process_time
from numba import jit
import pandas as pd
import numpy as np
import multiprocessing as mp
import seaborn as sns
colors = sns.color_palette("colorblind")

##############################################################################
##############################################################################
## Auxiliar functions

@jit(nopython=True)
def Sph2Cart(theta,phi):
    """
    Sph2Cart: Computes the cartesian coordinates (x,y,z) from a point of the
    sphere (1,theta,phi).
    """
    "--------------------------------------------------------"
    # Computes x,y,z given the angles theta and phi
    x=np.cos(phi)*np.sin(theta)
    y=np.sin(phi)*np.sin(theta)
    z=np.cos(theta)
    return x,y,z
@jit(nopython=True)

def Cart2Sph(x,y,z):
    
    """
    Cart2Sph: Computes the spherical coordinates (theta, phi) from a point of the sphere
    (x,y,z).
    """
    "--------------------------------------------------------"
    
    if z > 0:
        theta = np.arctan(np.sqrt(x**2+y**2)/z)
    elif z == 0:
        theta = np.pi/2
    else:
        theta = np.pi + np.arctan(np.sqrt(x**2+y**2)/z)
    
    if x>0 and y > 0:
        phi = np.arctan(y/x)
    elif x>0 and y <0:
        phi = 2*np.pi + np.arctan(y/x)
    elif x == 0:
        phi = np.sign(y)*np.pi/2
    else:
        phi = np.pi + np.arctan(y/x)      
    return theta, phi

def cart2polar(x,y):   
    
    """
    cart2polar: Computes the polar coordinates (r,theta) from a point (x,y).
    """
    "--------------------------------------------------------"
    
    r = np.sqrt(x**2+y**2)
    if x>0 and y>0:
        theta = np.arctan(y/x)
    elif x < 0:
        theta = np.arctan(y/x) + np.pi
    elif x>0 and y <0:
        theta = np.arctan(y/x) + 2*np.pi
    elif x==0 and y>0:
         theta = np.pi/2
    else:
         theta = 3*np.pi/2
         
    return r,theta

def polar2cart(r,theta):
    """
    polar2cart: Computes the cartesian coordinates (x,y) from a point (r,theta).
    """
    "--------------------------------------------------------"
    x = r*np.cos(theta)
    y = r*np.sin(theta)
    
    return x,y

@jit(nopython=True)
def great_circle_dist(p1,p2):
    """
    great_circle_dist: Computes the arc length distance between two points. We 
    consider the great circle distance, i.e, shortest distance between two points
    on the surface of the sphere.
    Inputs:
        p1: [theta_1, phi_1] (point1).
        p2: [theta_2, phi_2] (point2).
    """
    th1 = np.pi/2 - p1[0]
    th2 = np.pi/2 - p2[0]
    d_phi = p2[1]-p1[1]
    dist = np.arccos(np.sin(th1)*np.sin(th2)+np.cos(th1)*np.cos(th2)*
                    np.cos(d_phi))

    return dist

# Rotation Matrix
def apply_rotation(pos,theta,direction):
    
    """
    apply_rotation:  Apply rotation in 3D space.
    Inputs:
        pos: position vector [x,y,z].
        theta: angle of rotation.
        direction: axis of rotation.
    """
    "--------------------------------------------------------"
    if direction ==0: # x direction
        rx = np.array([ [1,0,0], 
                         [0,np.cos(theta),-np.sin(theta)],
                         [0,np.sin(theta),np.cos(theta)]])
        return np.dot(pos,rx)
    
    elif direction == 1: # y direction
        ry = np.array([ [np.cos(theta),0,np.sin(theta)], 
                         [0,1,0],
                         [-np.sin(theta),0,np.cos(theta)]])
        return np.dot(pos,ry)
    elif direction == 2: # z direction
        rz = np.array([[np.cos(theta),-np.sin(theta),0], 
                        [np.sin(theta),np.cos(theta),0],
                        [0,0,1]])
        return np.dot(pos,rz)
    else:
        return None

# Plot circular patch    
def plot_abs_patch(reg_abs,ax):
    p1 = np.array(reg_abs[0])
    theta_max = reg_abs[1]

    # Reflectant rigion
    theta,phi = np.mgrid[0.01:theta_max:80j,
                        0: 2*np.pi:80j]
    points = np.vstack([theta.ravel(), phi.ravel()]).T
    x,y,z = Sph2Cart(points[:,0], points[:,1])
    new_points = np.vstack([x,y,z]).T
    new_pos = []
    for point in new_points:
        new_point = apply_rotation(point, -p1[0], 1)
        new_point = apply_rotation(new_point, -p1[1], 2)
        new_point = np.asarray(new_point)
        theta,phi = Cart2Sph(new_point[0], new_point[1],new_point[2])
        new_pos.append([theta,phi])
    new_pos = np.array(new_pos)
    #Particle
    pos= Sph2Cart(new_pos[:,0],new_pos[:,1])
    ax.plot3D(pos[0],pos[1],pos[2],'-.',color="#CB3234")
    return

##############################################################################
##############################################################################
# Functions related with diffusuion in a circunference.

def Initial_Position_Circumference(reg_abs):
    theta_0 = np.random.uniform(reg_abs[1], 2*np.pi-reg_abs[0])
    return theta_0

@jit(nopython=True)
def Diff_Circumference(reg_abs,dt,D,pos0):
    """
    Diff_Circumference: reproduces the diffusion of a particle in a circunference.
    Inputs:
        reg_abs: Absorbent region [theta_min, theta_max].
        dt: Temporal step.
        D: Diffusion coefficiente.
        pos0: Initial position theta_0.
    
    Outputs:
        t: end time.
        pos: vector with positions.
    """   
    "--------------------------------------------------------"
    
    pos = pos0
    t = 0
    stop = False
    ## Main loop
    while not(stop):
        # Check absorption
        if reg_abs[0] <= pos<= reg_abs[1]:
            break
        # New position
        t = t + dt
        pos =  pos + np.sqrt(2*D*dt)*np.random.normal()
        pos = pos%(2*np.pi) # Negative angles to positive angles
    return t

def MFPT_Circumference(reg_abs,dt,D,n_initial_pos,n_rep,list_mfpt,list_theta):
    
    """
    MFPT_Circumference: Computes the MFPT in the circumference for n_initial_pos with n_rep for each initial
    position.
    Inputs:
        Initial Conditions: reg_abs, dt, D (As described before).
        n_initial_pos: Number of initial positions to compute the MFPT for each core.
        n_rep: Number of rep to use for each initial position.
        list_mftp: Multiproccesing manger.list() 
        list_theta: Multiproccesing manger.list()
    """    
    "--------------------------------------------------------"
    start_time = process_time()

    for i_initial_pos in range(n_initial_pos):
        # Random initial position
        theta_0 = Initial_Position_Circumference(reg_abs)
        times  = [] # Vector with times for each repetition
        for j_repetitions in range(n_rep):
            t = Diff_Circumference(reg_abs,dt,D,theta_0)
            times.append(t)

        # Add the mfpts to the list 
        list_mfpt.append(np.mean(times))
        #Add initial position to the list 
        list_theta.append(theta_0)
        print("Time spent in initial position "+str(i_initial_pos)+": ",process_time()-start_time)
    return

##############################################################################
# Paralelilazation of simulations circumference

def MFPT_Circumference_MP(reg_abs,dt,D,n_cores,n_initial_pos,n_rep):
    """
    MFPT_Circum_MP: Executes the function MFPT_phere in parallel trough n_cores. 
                    The results of the simulations are saved in a pandas dataframe.
    Inputs:
        Initial Conditions: reg_abs, reg_refl, dt, D (As described before).
        n_cores: Number of cores to use with multiproccesing.
        n_initial_pos: Number of initial positions to compute the MFPT for each core.
        n_rep: Number of rep to use for each initial position.

    Outputs:
        df: Pandas dataframe with the initial positions and the mean firts passage time
            of each initial position.  

            df["mfpt"]: Mean first passage times.
            df["theta"]: Angle theta of each initial position.
    """
    "--------------------------------------------------------" 
    ## Multiproccesing
    manager = mp.Manager()
    #Generate a manager.list() for each variable
    list_mfpt = manager.list()
    list_theta = manager.list()
    #nCores = multiprocessing.cpu_count() -1
    jobs = []

    ## For each core, call MFPT_Sphere
    for _ in range(n_cores):
        p = mp.Process(target=MFPT_Circumference, args =(reg_abs,dt,D,n_initial_pos,n_rep,
                                    list_mfpt, list_theta))
        jobs.append(p)
        p.start() #Start each procces

    for proc in jobs:
        proc.join()
    print()
    ## Add variables to dic
    dic = {"mfpt":list(list_mfpt), 
           "theta":list(list_theta)}

    ## Convert dic to dataframe
    df = pd.DataFrame.from_dict(dic)
    return df


##############################################################################
##############################################################################
## Functions related with diffusion on the sphere.

@jit(nopython=True)
def Initial_Position_Sphere_Negative(reg_abs):
    """
    Initial_Position_Sphere_Negative: Computes the initial position of a particle, when we are interested 
    on computing time until the particle escapes from a circular patch of aperture angle 
    delta.
    Inputs:
        domain: [[theta_pole,phi_pole],delta] (This domain is a circular patch with pole [theta_pole,phi_pole]
                                                and aperture angle delta).
    Ouputs:
        [theta_0, phi_0]: Random initial position inside the domain.
    """
    theta_0 = np.random.uniform(reg_abs[0][0]-reg_abs[1], reg_abs[0][0]+reg_abs[1])
    phi_0 = np.random.uniform(reg_abs[0][1]-reg_abs[1], reg_abs[0][1]+ reg_abs[1])
    point = [theta_0, phi_0]
    dist = great_circle_dist(point,reg_abs[0])
    stop = True 
    if dist > reg_abs[1]:
        stop = False
    while not(stop):
        theta_0 = np.random.uniform(reg_abs[0][0]-reg_abs[1], reg_abs[0][0]+reg_abs[1])
        phi_0 = np.random.uniform(reg_abs[0][1]-reg_abs[1], reg_abs[0][1]+ reg_abs[1])
        point = [theta_0, phi_0]
        dist = great_circle_dist(point,reg_abs[0])
        if dist > reg_abs[1]:
            break

    return theta_0, phi_0


@jit(nopython=True)
def Initial_Position_Sphere(reg_refl, reg_abs):
    """
    Initial_Position: Computes the intial position of the particle. This position will not be
                     in the absorbent region. (Equivalent to a uniform distribution).
    Inputs:
        reg_refl: Boundary for the reflecting region [[theta_min, theta_max], [phi_min, phi_max]].
        reg_abs:  Pole and and aperture angle of the circular absorbent patch, [[theta_pole,phi_pole], theta_max].
    """
    "--------------------------------------------------------"
    # Random point in the whole region
    theta_0 = np.random.uniform(reg_refl[0][0], reg_refl[0][1])
    phi_0 = np.random.uniform(reg_refl[1][0],reg_refl[1][1])

    ## Uncomment for a initial position that can't be on the absorbing patch
    # point = [theta_0,phi_0] 
    # # Compute the great circle distance between the pole of the patch and the point
    # dist = great_circle_dist(point, reg_abs[0])

    # while dist < reg_abs[1]: # Repit until we find a point that is not in the circular patch
    #     theta_0 = np.random.uniform(reg_refl[0][0], reg_refl[0][1])
    #     phi_0 = np.random.uniform(reg_refl[1][0],reg_refl[1][1])   
    #     point = [theta_0,phi_0] 
    #     dist = great_circle_dist(point, reg_abs[0])

    return theta_0, phi_0



@jit(nopython=True)
def Reflection(reg_refl, theta, phi):
    """
    Reflection: Check and do the reflection when the particle surppass the boundary.
    Inputs:
        reg_refl: Boundary for the reflection.
        theta: Angle theta on the sphere.
        phi: Angle phi on the sphere.
    """
    "--------------------------------------------------------"
    # Theta boundary
    if theta <= reg_refl[0][0]:
        theta = reg_refl[0][0] + abs(reg_refl[0][0] - theta)
    elif theta >= reg_refl[0][1]:
        theta = reg_refl[0][1] - abs(theta - reg_refl[0][1] )
    if phi <= reg_refl[1][0]:
        phi = reg_refl[1][0] + abs(reg_refl[1][0] - phi)
    elif phi >= reg_refl[1][1]:
        phi = reg_refl[1][1] - abs(phi - reg_refl[1][1] )    
    
    return theta, phi




@jit(nopython=True)
def Diff_Tang_Plane(theta,phi,dt,D):

    # Base of the tangential plane to the north pole
    e1 = np.array([1,0,0])
    e2 = np.array([0,1,0])
    
    # Diffussion
    r1 = np.random.normal(0,1)
    r2 = np.random.normal(0,1)
    
    delta = np.sqrt(2*dt*D)*(e1*r1+e2*r2)
    pos = np.array([0,0,1])
    pos = (pos + delta)
    new_pos = pos/np.sqrt(pos[0]**2+ pos[1]**2 + pos[2]**2) 
    
    # Rotations
    x1 = new_pos[0]*np.cos(-theta) - new_pos[2]*np.sin(-theta)
    y1 = new_pos[1]
    z = new_pos[0]*np.sin(-theta) + new_pos[2]*np.cos(-theta)
    
    x = x1*np.cos(-phi)+y1*np.sin(-phi)
    y = -x1*np.sin(-phi)+y1*np.cos(-phi)

    return [x,y,z]


def Diff_Sphere(reg_refl,reg_abs,dt, D,pos0): 
    """
    Diff_Sphere: Simulates the diffusion of a Brownian particle on a spherical surface delimited by a 
                reflecting boundary. Each step of the simulation is done diffusing on the tanget 
                perperdicular plane to the north pole and rotating the resulting point with the rotation
                that maps the north pole to the previos point. The simulation ends when the particle
                reaches the absorbing domain.
    Inputs:
        reg_refl: Reflecting domain [[theta_min, theta_max], [phi_min,phi_max]] (numpy array)
        reg_abs: Absorbing circular patch ([theta_pole,phi_pole],delta). The circular patch is expressed as 
                spherical cap with pole [theta_pole,phi_pole] and an aperture angle delta.
        dt: Temporal step.
        D: Diffusion coefficient.
        pos0: Initial positin [theta_0,phi_0] (numpy array).
    Outputs:
        t: Time when the particle reaches the absorbing domain.
    """

    # Initial condition in cartesian coordinates
    x0,y0,z0 = Sph2Cart(pos0[0], pos0[1])
    
    last_pos = [x0,y0,z0] #last position
    t = 0 #Time

    ## Check if the initial position is the absorbing patch
    dist = great_circle_dist(np.array([pos0[0],pos0[1]]),reg_abs[0])
    if dist <= reg_abs[1]:
        return t
    else: # Main loop
        stop = False
        ## Loop    
        while not(stop):
            ## Diffusion step
            t = t+dt #Update time
            theta,phi = Cart2Sph(last_pos[0],last_pos[1],last_pos[2])
            x,y,z = Diff_Tang_Plane(theta, phi,dt,D)
            
            ## Absorption and reflection check
            theta, phi = Cart2Sph(x, y,z)

            # Absorprtion check
            dist = great_circle_dist(np.array([theta,phi]),reg_abs[0])
            if dist <= reg_abs[1]: # If the point [theta,phi] is in the absorbing patch, end.
                stop = True
                break
            
            # Reflection check
            theta, phi = Reflection(reg_refl, theta, phi)  

            ## Update new positions
            x,y,z = Sph2Cart(theta, phi)
            last_pos = [x,y,z]
        
        return t



def Diff_Sphere_Pos(reg_refl,reg_abs,dt, D,pos0): 
    """
    Diff_Sphere_Pos: Similar to Diff_Sphere but the position of the particle 
                    in each time is also saved.
    
    Inputs:
        reg_refl: Reflecting domain [[theta_min, theta_max], [phi_min,phi_max]] (numpy array).
        reg_abs: Absorbing circular patch ([theta_pole,phi_pole],delta). The circular patch is expressed as 
            spherical cap with pole [theta_pole,phi_pole] and an aperture angle delta.
        dt: Temporal step.
        D: Diffusion coefficient.
        pos0: Initial positin [theta_0,phi_0] (numpy array).
    Outputs:
        t: Vector with times.
        pos: Vector with the position of the particle for each time.
    """

    # Initial condition in cartesian
    x0,y0,z0 = Sph2Cart(pos0[0], pos0[1])
    
    pos = [[x0],[y0],[z0]] #Vector with positions
    t = [0] #Vector with times

    ## Check if the initial position is the absorbing patch
    dist = great_circle_dist(np.array([pos0[0],pos0[1]]),np.array(reg_abs[0]))
    if dist <= reg_abs[1]:
        return t, pos
    else: # Main loop
        stop = False
        ## Loop    
        while not(stop):
            ## Diffusion step
            t.append(t[-1] + dt)
            theta,phi = Cart2Sph(pos[0][-1],pos[1][-1],pos[2][-1])
            x,y,z = Diff_Tang_Plane(theta, phi,dt,D)

            ## Reflection and absorption check
            theta, phi = Cart2Sph(x, y, z)

            # Absorbing region 
            dist = great_circle_dist(np.array([theta,phi]),np.array(reg_abs[0]))
            if dist <= reg_abs[1]: # If the point [theta,phi] is in the absorbing patch, end.
                pos[0].append(x)
                pos[1].append(y)
                pos[2].append(z)
                stop = True
                break
            
            # Reflection
            theta, phi = Reflection(reg_refl, theta, phi)  

            ## Update new positions
            x,y,z = Sph2Cart(theta, phi)
            pos[0].append(x); pos[1].append(y); pos[2].append(z)   
        
        return t, pos


def Diff_Sphere_Negative(reg_abs,dt,D,pos0):

    """
    Diff_Sphere_Negative: Simulates a Brownian particle inside a circular patch on the sphere.
                         The simulation end when the particle leaves the patch.
    Inputs:
        reg_abs: Circular patch ([theta_pole,phi_pole],delta). The circular patch is expressed as 
                spherical cap with pole [theta_pole,phi_pole] and an aperture angle delta.
        dt: Temporal step.
        D: Diffusion coefficient.
        pos0: Initial positin [theta_0,phi_0] (numpy array).
    Outputs:
        t: Time when the particle leaves the circular patch.
    """
    # Initial condition in cartesian
    x0,y0,z0 = Sph2Cart(pos0[0], pos0[1])
    
    last_pos = [x0,y0,z0] #last position
    t = 0 #Time
    stop = False
    ## Loop    
    while not(stop):
        ## Diffusion step
        t = t+dt
        theta,phi = Cart2Sph(last_pos[0],last_pos[1],last_pos[2])
        x,y,z= Diff_Tang_Plane(theta, phi,dt,D)

        ## Reflection and absorption check
        theta, phi = Cart2Sph(x, y,z)

        # Escape form the circle
        dist = great_circle_dist(np.array([theta,phi]),reg_abs[0])
        if dist > reg_abs[1]: # If the point [theta,phi] is in the absorbing patch, end.
            stop = True
            break

        ## Update new positions
        x,y,z = Sph2Cart(theta, phi)
        last_pos = [x,y,z]
        
    return t    


def Diff_Sphere_Negative_Pos(reg_abs,dt,D,pos0):


    """
    Diff_Sphere_Negative_Pos: Similar to Diff_Sphere_Negative but the position of the particle 
                    in each time is also saved.
    Inputs:
        reg_abs: Circular patch ([theta_pole,phi_pole],delta). The circular patch is expressed as 
                spherical cap with pole [theta_pole,phi_pole] and an aperture angle delta.
        dt: Temporal step.
        D: Diffusion coefficient.
        pos0: Initial positin [theta_0,phi_0] (numpy array).
    Outputs:
        t: Time when the particle leaves the circular patch.
    """
    
    # Initial condition in cartesian
    x0,y0,z0 = Sph2Cart(pos0[0], pos0[1])
    
    pos = [[x0],[y0],[z0]] #Vector with positions
    t = [0] #Vector with times
    stop = False
    ## Loop    
    while not(stop):
        ## Diffusion step
        t.append(t[-1] + dt)
        theta,phi = Cart2Sph(pos[0][-1],pos[1][-1],pos[2][-1])
        x,y,z = Diff_Tang_Plane(theta, phi,dt,D)

        ## Reflection and absorption check
        theta, phi = Cart2Sph(x, y, z)

        # Absorbing region 
        dist = great_circle_dist(np.array([theta,phi]),np.array(reg_abs[0]))
        if dist > reg_abs[1]: # If the point [theta,phi] is in the absorbing patch, end.
            pos[0].append(x)
            pos[1].append(y)
            pos[2].append(z)
            stop = True
            break

        ## Update new positions
        x,y,z = Sph2Cart(theta, phi)
        pos[0].append(x); pos[1].append(y); pos[2].append(z)   
        
    return t,pos

##############################################################################
# Paralelilazation of the simulations on the sphere

def MFPT_Sphere(reg_abs,reg_refl,dt,D,n_initial_pos,n_rep,list_mfpt,
                list_theta, list_phi):
    """
    MFPT_Sphere: Computes the MFPT in the sphere for n_initial_pos with n_rep for each initial
    position.
    Inputs:
        Initial Conditions: reg_abs, reg_refl, dt, D (As described before).
        n_initial_pos: Number of initial positions to compute the MFPT for each core.
        n_rep: Number of rep to use for each initial position.  
        list_mftp: Multiproccesing manger.list() 
        list_theta: Multiproccesing manger.list()
        list_phi: Multiproccesing manger.list()       
    """
    "--------------------------------------------------------"
    
    for i_initial_pos in range(n_initial_pos):
        # Random initial position
        start_time = process_time()
        pos0_3d = Initial_Position_Sphere(reg_refl, reg_abs)  # Initial pos on the sphere
        times_3d = [] #Vector with times for diffusion on the sphere
        for j_repetitions in range(n_rep):
            t = Diff_Sphere(reg_refl,reg_abs,dt, D,pos0_3d)
            times_3d.append(t)           
        # Add the mfpts to the list
        list_mfpt.append(np.mean(times_3d))
        # Add initial positions to the list
        list_theta.append(pos0_3d[0])
        list_phi.append(pos0_3d[1])
        print("Time spent in initial position "+str(i_initial_pos)+": ",process_time()-start_time)
    return

def MFPT_Sphere_MP(reg_abs, reg_refl, dt, D, n_cores, n_initial_pos, n_rep):

    """
    MFPT_Sphere_MP: Executes the function MFPT_phere in parallel trough n_cores. 
                    The results of the simulations are saved in a pandas dataframe.

    Inputs:
        Initial Conditions: reg_abs, reg_refl, dt, D (As described before).
        n_cores: Number of cores to use with multiproccesing.
        n_initial_pos: Number of initial positions to compute the MFPT for each core.
        n_rep: Number of rep to use for each initial position.
    
    Outputs:
        df: Pandas dataframe with the initial positions and the mean firts passage time
            of each initial position.
        
            df["mfpt"]: Mean first passages times.
            df["theta]: Angle theta of each initial position.
            df["phi"]: Angle phi of each initial positon.

    """
    "--------------------------------------------------------"
    ## Multiproccesing
    manager = mp.Manager()
    #Generate a manager.list() for each variable
    list_mfpt = manager.list()
    list_theta = manager.list()
    list_phi = manager.list()
    #nCores = multiprocessing.cpu_count() -1
    jobs = []

    ## For each core, call MFPT_Sphere
    for core in range(n_cores):
        p = mp.Process(target=MFPT_Sphere, args =(reg_abs,reg_refl,dt,D,n_initial_pos,n_rep,
                                    list_mfpt, list_theta, list_phi))
        jobs.append(p)
        p.start() #Start each procces
    
    for proc in jobs:
        proc.join()
          
    ## Add variables to dic
    dic = {"mfpt":list(list_mfpt), "theta":list(list_theta),
            "phi": list(list_phi)}
    ## Convert dic to dataframe
    df = pd.DataFrame.from_dict(dic)
    return df

def MFPT_Sphere_Negative(reg_abs,dt,D,n_initial_pos,n_rep,list_mfpt,
                list_theta, list_phi):
    """
    MFPT_Sphere_Negative: Computes the MFPT of particle escaping a circular patch on the sphere 
                          for n_initial_pos with n_rep for each initial position.

    Inputs:
        Initial Conditions: reg_abs, dt, D (As described before).
        n_initial_pos: Number of initial positions to compute the MFPT for each core.
        n_rep: Number of rep to use for each initial position.
        list_mftp: Multiproccesing manger.list() 
        list_theta: Multiproccesing manger.list()
        list_phi: Multiproccesing manger.list()
    """
    "--------------------------------------------------------"

    for i_initial_pos in range(n_initial_pos):
        # Random initial position
        start_time = process_time()
        pos0_3d = Initial_Position_Sphere_Negative(reg_abs)  # Initial pos on the sphere
        times_3d = [] #Vector with times for diffusion on the sphere
        for j_repetitions in range(n_rep):
            t = Diff_Sphere_Negative(reg_abs,dt, D,pos0_3d)
            times_3d.append(t)           
        # Add the mfpts to the list
        list_mfpt.append(np.mean(times_3d))
        # Add initial positions to the list
        list_theta.append(pos0_3d[0])
        list_phi.append(pos0_3d[1])
        print("Time spent in initial position "+str(i_initial_pos)+": ",process_time()-start_time)
        
    return

def MFPT_Sphere_Negative_MP(reg_abs, dt, D, n_cores, n_initial_pos, n_rep):

    """
    MFPT_Sphere_Negative_MP: Executes the function MFPT_phere in parallel trough n_cores. 
                            The results of the simulations are saved in a pandas dataframe. 
    Inputs:
        Initial Conditions: reg_abs, reg_refl, dt, D (As described before).
        n_cores: Number of cores to use with multiproccesing.
        n_initial_pos: Number of initial positions to compute the MFPT for each core.
        n_rep: Number of rep to use for each initial position.
    Outputs:
        df: Pandas dataframe with the initial positions and the mean firts passage time
            of each initial position.       

            df["mfpt"]: Mean first passages times for each initial position.
            df["theta"]: Theta angle of each initial position.
            df["phi]: Phi angle of each initial position.

    """
    "--------------------------------------------------------"
    ## Multiproccesing
    manager = mp.Manager()
    #Generate a manager.list() for each variable
    list_mfpt = manager.list()
    list_theta = manager.list()
    list_phi = manager.list()
    #nCores = multiprocessing.cpu_count() -1
    jobs = []

    ## For each core, call MFPT_Sphere
    for core in range(n_cores):
        p = mp.Process(target=MFPT_Sphere_Negative, args =(reg_abs,dt,D,n_initial_pos,n_rep,
                                    list_mfpt, list_theta, list_phi))
        jobs.append(p)
        p.start() #Start each procces
    
    for proc in jobs:
        proc.join()
          
    ## Add variables to dic
    dic = {"mfpt":list(list_mfpt), "theta":list(list_theta),
            "phi": list(list_phi)}
    ## Convert dic to dataframe
    df = pd.DataFrame.from_dict(dic)
    return df