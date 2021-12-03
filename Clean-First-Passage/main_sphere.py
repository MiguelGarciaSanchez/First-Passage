import numpy as np
import Functions as f
import os




if __name__ == "__main__":
    
    ##############################################################################
    ## Customizable parameters

    # Initial Condtions
    dt = 0.0001
    D = 1       
    # Reflecting boundary
    theta_min = 77
    theta_max = 137
    reg_refl = np.array([[theta_min,theta_max],[0,360]]) * np.pi/180   #[[theta_min, theta_max], [phi_min,phi_max]]
    # Absorbing boundary
    theta_pole = 86
    delta = 19/2
    phi_pole = 90
    reg_abs = (np.array([theta_pole*np.pi/180,phi_pole*np.pi/180]),delta*np.pi/180) #[[theta_pole,phi_pole], Angle_aperture]
    # Number of simulations
    n_initial_pos = 1
    n_rep = 10
    n_cores = 10
    file_name = "Example_IgG2b"

    ##############################################################################
    file_path = "../Data/Sphere/"
    directory = os.path.dirname(file_path)
    try:
        os.stat(directory)
    except:
        os.makedirs(directory)

    # Simulation
    df = f.MFPT_Sphere_MP(reg_abs, reg_refl, dt, D, n_cores, n_initial_pos, n_rep)
    #df = f.MFPT_Sphere_Negative_MP(reg_abs,dt,D,n_cores,n_initial_pos,n_rep) #Uncomment for particle escaping the circular patch (reg_abs)
    df.to_pickle("../Data/Sphere/IgG2a.pkl")


