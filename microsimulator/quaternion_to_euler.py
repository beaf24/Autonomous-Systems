import numpy as np

def z_quaternion_to_euler( w, z):
    
    t3 = +2.0 * (w * z)
    t4 = +1.0 - 2.0 * (z * z)

    Z = np.degrees( np.arctan2( t3, t4 ))
    return Z
