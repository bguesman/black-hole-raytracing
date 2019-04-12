#We have the main differential equations

# t'' = -2GM/(r(r-2GM)) * r' *t'
#r'' = -(GM/r^3)*(r-2GM)*(t')^2 + (GM/(r(r-@GM)))
import numpy as np
import copy

def sin(theta):
	return np.sin(theta)
def cos(theta):
	return np.cos(theta)
G=1
M=1

def sphere_to_cartesian(vec):
	t, r, thet, phi = tuple(vec)
	x= r*sin(thet)*cos(phi)
	y= r*sin(thet)*sin(phi)
	z= r*cos(thet)
	return np.array([t, x, y, z])

def cartesian_to_sphere(vec):
	t, x, y, z = tuple(vec)
	r= (x**2 + y**2 + z**2)**.5
	thet= np.arccos(z/r)
	phi= np.arctan2(y,x)
	return np.array([t, r, thet, phi])

class photon(object):
	def __init__(pos, pos_):
		self.pos0 = copy.copy(pos)
		self.pos0_ = copy.copy(pos_)
		self.sph_pos = cartesian_to_sphere(copy.copy(pos))
		self.sph_pos_ = cartesan_to_sphere(copy.copy(pos_))
		self.sph_poss = np.array([cartesian_to_sphere(copy.copy(pos))])
		self.sph_pos_s = np.array([cartesan_to_sphere(copy.copy(pos_))])

		def step_rk4():
			#make state updated

def make_photon_at_grid_pt(pt, eye_r, film_r, film_height, film_height, film_width):
	cart_pos = np.array([0, pt[1] - film_width/2, film_height/2 - pt[0], camera_r])
	vx = cart_pos[1]
	vy = cart_pos[2]
	vz = film_r - eye_r
	v_norm = (vx**2 + vy**2 + vz**2)**.5
	cart_pos_ = np.array([1, vx/v_norm, vy/v_norm, vz/v_norm])
	return photon(cart_pos, cart_pos_)



def pos__(pos, pos_):
	t, r, thet, phi = tuple(pos)
	t_, r_, thet_, phi_ = tuple(pos_)
	t__ = ((-2*G*M)/(r*(r-2*G*M)))*r_*t_
	
	r__ = -(G*M/(r**3))*(r-2*G*M)*(t_**2) + (G*M/(r*(r-2*G*M)))*(r_**2) + (r-2*G*M)*((thet_**2) + (sin(thet)*phi_)**2)

	thet__ = (-2/r)*thet_*r_ + sin(thet)*cos(thet)*(phi_**2)

	phi__ = (-2/r)*phi_*r_ + 2*(cos(thet)/sin(thet))*thet_*phi_

	return np.array([t__, r__, thet__, phi__])

print(cartesian_to_sphere(sphere_to_cartesian(np.array([1,2,3,4]))))
print(sphere_to_cartesian(cartesian_to_sphere(np.array([1,2,3,4]))))