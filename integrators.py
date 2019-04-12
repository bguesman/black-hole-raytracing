# returns tuple of updated position and updated
# velocity
def euler_step(pos, pos_, accel_function, delta_t):
    pos__ = accel_function(pos, pos_)
    return (pos + delta_t * pos, pos_ + delta_t * pos__)

def rk4_step(pos, pos_, accel_function, delta_t):
    k1 = delta_t * accel_function(pos, pos_)
    k2 = 
    k3 = 0
    k4 = 0
