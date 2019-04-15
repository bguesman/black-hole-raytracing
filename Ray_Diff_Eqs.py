#We have the main differential equations

# t'' = -2GM/(r(r-2GM)) * r' *t'
#r'' = -(GM/r^3)*(r-2GM)*(t')^2 + (GM/(r(r-@GM)))
import numpy as np
import copy
from integrators import euler_step
from matplotlib import pyplot as plt
from tqdm import tqdm
from numba import jit
from mpl_toolkits.mplot3d import Axes3D

@jit
def euler_step_txyz(pos, pos_, accel_function, delta_l):
    pos__ = accel_function(pos, pos_)
    #print(pos__)
    cart_pos = pos + delta_l*pos_
    cart_pos_ = pos_+ delta_l*pos__
    return cart_pos, cart_pos_


def sin(theta):
	return np.sin(theta)
def cos(theta):
	return np.cos(theta)
G=1
M=.5

@jit
def sphere_to_cartesian(vec):
	t, r, thet, phi = tuple(vec)
	x= r*sin(phi)*cos(thet)
	y= r*sin(thet)*sin(phi)
	z= r*cos(phi)
	return np.array([t, x, y, z])

@jit
def cartesian_to_sphere(vec):
	t, x, y, z = tuple(vec)
	r= (x**2 + y**2 + z**2)**.5
	phi= np.arccos(z/r)
	thet=  np.arctan2(y,x)
	return np.array([t, r, thet, phi])

def dl_func(sph_pos):
	print(10* np.exp(-sph_pos[1]))
	return 10* np.exp(-sph_pos[1])

class photon(object):
	def __init__(self, pos, pos_):
		self.pos0 = copy.copy(pos)
		self.pos0_ = copy.copy(pos_)
		self.pos = copy.copy(pos)
		self.pos_ = copy.copy(pos_)
		self.poss = np.array([copy.copy(pos)])
		self.pos_s = np.array([copy.copy(pos_)])
		self.dl = 0.01
		self.finished = False

	def step(self):
		if not self.finished:
			#print(self.sph_pos_)
			self.pos, self.pos_ = euler_step_txyz(self.pos, self.pos_, txyz_pos__, self.dl)
			if (cartesian_to_sphere(self.pos)[1] < 2*G*M + 0.0001):
				print(cartesian_to_sphere(self.pos))
				self.finished = True
				print("finished")
			else:
				self.poss = np.append(self.poss, np.array([copy.copy(self.pos)]), axis=0)
				self.pos_s = np.append(self.pos_s, np.array([copy.copy(self.pos_)]), axis=0)

def make_photon_at_grid_pt(pt, eye_r, film_r, film_height, film_width):
	cart_pos = np.array([0, pt[1] - film_width/2, film_height/2 - pt[0], film_r])
	vx = cart_pos[1]
	vy = cart_pos[2]
	vz = -(film_r - eye_r)
	v_norm = (vx**2 + vy**2 + vz**2)**.5
	cart_pos_ = np.array([1, vx/v_norm, vy/v_norm, vz/v_norm])

	return photon(cart_pos, cart_pos_)

def make_basic_photon_at_grid_pt(pt, eye_r, film_r, film_height, film_width):
	cart_pos = np.array([0, film_width*(1/2 - pt[1]), film_height*(1/2 - pt[0]), -eye_r])
	vx = 0
	vy = 0
	vz = 1
	v_norm = (vx**2 + vy**2 + vz**2)**.5
	cart_pos_ = np.array([1, vx/v_norm, vy/v_norm, vz/v_norm])
	return photon(cart_pos, cart_pos_)

@jit
def txyz_pos__(txyz_pos, txyz_pos_):
	t, x, y, z = tuple(txyz_pos)
	t, r, thet, phi = tuple(cartesian_to_sphere(txyz_pos))

	t_, x_, y_, z_ = tuple(txyz_pos_)
	# print("t is %s, t_ is %s"%(t, t_))
	# print("x is %s, x_ is %s"%(x, x_))
	# print("y is %s, y_ is %s"%(y, y_))
	# print("z is %s, z_ is %s"%(z, z_))
	r_ = cos(thet)*sin(phi)*x_ + sin(thet)*sin(phi)*y_ + cos(phi)*z_
	thet_ = (-sin(thet)/(r*sin(phi)))*x_ + (cos(thet)/(r*sin(phi)))*y_
	phi_ = (cos(thet)*cos(phi)/r)*x_ + (sin(thet)*cos(phi)/r)*y_ + (-sin(phi)/r)*z_
	# r_ = (x*x_ +y*y_ + z*z_)/r
	# thet_ =	(x*y_ - y*x_)/(x**2 + y**2)
	# phi_ =  (-1/((1 - ((z/r)**2))**.5))*((r*z_ -z*r_)/r**2)

	# CARROL CONVENTION
	# t__ = ((-2*G*M)/(r*(r - 2*G*M)))*r_*t_

	# r__ = -(G*M/(r**3))*(r-2*G*M)*(t_**2) + (G*M/(r*(r-2*G*M)))*(r_**2) + (r-2*G*M)*((thet_**2) + (sin(thet)**2)*(phi_)**2)

	# thet__ = (-2/r)*thet_*r_ + sin(thet)*cos(thet)*(phi_**2)

	# phi__ = (-2/r)*phi_*r_ - 2*(cos(thet)/sin(thet))*thet_*phi_

	# WOLFRAM CONVENTION
	t__ = ((-2*G*M)/(r*(r - 2*G*M)))*r_*t_

	r__ = -(G*M/(r**3))*(r-2*G*M)*(t_**2) + (G*M/(r*(r-2*G*M)))*(r_**2) + (r-2*G*M)*((phi_**2) + (sin(phi)**2)*(thet_)**2)

	phi__ = (-2/r)*phi_*r_ + sin(phi)*cos(phi)*(thet_**2)

	thet__ = (-2/r)*thet_*r_ - 2*(cos(phi)/sin(phi))*thet_*phi_

	# print("+++++++++++++++++++++++++++++++++")
	# print("r - 2GM is %s"%(r - 2*G*M))
	# print("---------------------------------")
	# print("t is %s, t_ is %s, t__ is %s"%(t, t_, t__))
	# print("r is %s, r_ is %s, r__ is %s"%(r, r_, r__))
	# print("theta is %s, theta_ is %s, thet__ is %s"%(thet, thet_, thet__))
	# print("phi is %s, phi_ is %s, phi__ is %s"%(phi, phi_, phi__))
	# print("+++++++++++++++++++++++++++++++++")
	x__ = -2*sin(thet)*sin(phi)*thet_*r_ + 2*cos(thet)*cos(phi)*r_*phi_ - 2*r*sin(thet)*cos(phi)*phi_*thet_ + cos(thet)*sin(phi)*r__ - r*sin(thet)*sin(phi)*thet__ + r*cos(thet)*cos(phi)*phi__ - r*cos(thet)*sin(phi)*(thet_**2 + phi_**2)
	# print("\n\nX__ IS\n\n %s" % x__)
	y__ = 2*cos(thet)*sin(phi)*thet_*r_ + 2*sin(thet)*cos(phi)*r_*phi_ + 2*r*cos(thet)*cos(phi)*thet_*phi_ + sin(thet)*sin(phi)*r__  + r*cos(thet)*sin(phi)*thet__ +r*sin(thet)*cos(phi)*phi__ - r*sin(thet)*sin(phi)*(thet_**2 + phi_**2)

	z__ = -r*cos(phi)*(phi_**2) + cos(phi)*r__ - 2*sin(phi)*phi_*r_ - r*sin(phi_)*phi__

	txyz_pos__ = np.array([t__, x__, y__, z__])
	#print(txyz_pos__)
	return txyz_pos__

# photon_1 = make_photon_at_grid_pt([2,2], 6, 4, 4, 4)
# num_steps = 2000
# for i in tqdm(range(num_steps)):
# 	photon_1.step()

# poss_txyz = []
# for i in tqdm(range(np.shape(photon_1.sph_poss)[0])):
# 	poss_txyz.append(sphere_to_cartesian(photon_1.sph_poss[i,:]))

#M=.000001


# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.set_xlabel("x")
# ax.set_ylabel("y")
# ax.set_zlabel("z")

# num_steps = 1000
# for m in tqdm(range(0,7)):
# 	for n in range(0,7):
# 		photon_i = make_photon_at_grid_pt([m,n], 4, 2, .2, .2)
# 		for i in range(num_steps):
# 			photon_i.step()
# 		poss_txyz = np.array(photon_i.poss)
# 		ax.plot(poss_txyz[:,1], poss_txyz[:,2], poss_txyz[:,3])

# photon1 = photon(np.array([0, 3.8, 3.8, -5]), np.array([1, 0,0,1]))
# for i in tqdm(range(num_steps)):
# 	photon1.step()
# poss_txyz = photon1.poss
# print(photon1.pos_s)
# ax.plot(poss_txyz[:,1], poss_txyz[:,2], poss_txyz[:,3])

#draw sphere
# u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
# x = (2*G*M)*np.cos(u)*np.sin(v)
# y = (2*G*M)*np.sin(u)*np.sin(v)
# z = (2*G*M)*np.cos(v)
# ax.plot_wireframe(x, y, z, color="k")
# plt.show()

# print(cartesian_to_sphere(sphere_to_cartesian(np.array([1,2,3,4]))))
# print(sphere_to_cartesian(cartesian_to_sphere(np.array([1,2,3,4]))))
