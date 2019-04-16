from Ray_Diff_Eqs import make_basic_photon_at_grid_pt, make_photon_at_grid_pt, photon
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot as plt
from tqdm import tqdm
import numpy as np 
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")
G=1
M=.1
num_steps = 3000
for m in tqdm(range(0,9)):
	for n in range(0,9):
		photon_i = make_basic_photon_at_grid_pt([m,n], 10, 5, 8, 8)
		for i in range(num_steps):
			photon_i.step()
		poss_txyz = np.array(photon_i.poss)
		ax.plot(poss_txyz[:,1], poss_txyz[:,2], poss_txyz[:,3])

# photon1 = photon(np.array([0, 3.8, 3.8, -5]), np.array([1, 0,0,1]))
# for i in tqdm(range(num_steps)):
# 	photon1.step()
# poss_txyz = photon1.poss
# print(photon1.pos_s)
# ax.plot(poss_txyz[:,1], poss_txyz[:,2], poss_txyz[:,3])

#draw sphere
u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
x = (2*G*M)*np.cos(u)*np.sin(v)
y = (2*G*M)*np.sin(u)*np.sin(v)
z = (2*G*M)*np.cos(v)
ax.plot_wireframe(x, y, z, color="k")
plt.show()
