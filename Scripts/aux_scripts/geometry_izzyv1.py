
def grad_sphere(field, llon, llat):
    """
    Function to calculate the gradient of a 2D scalar field over a sphere, given the coordinates in degrees on 
    the same 2D grid. The derivatives are taken as second-order differences in the interior, and first-order 
    (forward or backward) on the edges.
    """
    import numpy as np
    R = 6371.0e3 # Earth radius in km.
    
    field = np.double(field)
    llon = np.double(llon)
    llat = np.double(llat)
    
    costheta = np.cos(llat*np.pi/180)
    
    df_dx = field-field
    df_dx[:,1:-1] = (field[:,2:]-field[:,:-2])/(R*costheta[:,1:-1]*(llon[:,2:]-llon[:,:-2])*np.pi/180)
    df_dx[:,0] = (field[:,1]-field[:,0])/(R*costheta[:,0]*(llon[:,1]-llon[:,0])*np.pi/180)
    df_dx[:,-1] = (field[:,-1]-field[:,-2])/(R*costheta[:,-1]*(llon[:,-1]-llon[:,-2])*np.pi/180)
    
    df_dy = field-field
    df_dy[1:-1,:] = (field[2:,:]-field[:-2,:])/(R*(llat[2:,:]-llat[:-2,:])*np.pi/180)
    df_dy[0,:] = (field[1,:]-field[0,:])/(R*(llat[1,:]-llat[0,:])*np.pi/180)
    df_dy[-1,:] = (field[-1,:]-field[-2,:])/(R*(llat[-1,:]-llat[-2,:])*np.pi/180)
    
    return df_dx, df_dy
