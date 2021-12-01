import numpy as np
import Functions as f
import os




if __name__ == "__main__":

    """
    Creates a pandas dataframe containg the position and the mean first passage
    time for each position.
    """

    ##############################################################################
    ## Customizable parameters
    
    # Initial conditions
    dt = 1e-4
    D = 1
    # Absorbing boundary
    theta_min = 0
    theta_max = 19
    reg_abs = np.array([theta_min,theta_max])*np.pi/180   
    # Number of simulations
    n_initial_pos = 10 # Number of initial positions 
    n_rep = 100 # Number of repetitions for each initial position
    n_cores = 10  # Number of cores for paralelization
    file_name = "Example"

    ##############################################################################
    file_path = "../Data/Circumference/"
    directory = os.path.dirname(file_path)
    try:
        os.stat(directory)
    except:
        os.makedirs(directory)



    ## Simulations
    df = f.MFPT_Circumference_MP(reg_abs, dt, D, n_cores, n_initial_pos, n_rep)
    ## Save dataframe file
    print(df)
    df.to_pickle("../Data/Circumference/"+file_name+".pkl")
    
