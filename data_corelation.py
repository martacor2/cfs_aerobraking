import numpy as np
from numpy.linalg import inv
from numpy.linalg import norm
import pandas as pd
import matplotlib.pyplot as plt
import os
from lamberts_problem import *
from data_analysis import *
from closed_form_solution import *
import csv

#constants
aoa=np.pi/2
g_ref = 3.71  # m/s^2
rho_ref = 8.748923102971180e-07 #kg/m^3
mu = 4.2828e13 # gravitational parameter, m^3/s^2
h_ref = 90000 #m
H =6300 #m
m=461 #kg
Rp=3396.2*1000 #m
CD_0=1.477
CL_0=0.1
Sref=11 #m^2

csv_file=pd.read_csv("data/data_all_sims.csv")

name = np.array(csv_file["Simulation"])
hp = np.array(csv_file["hp"])
a_coeff = np.array(csv_file["a"])
b_coeff = np.array(csv_file["b"])
c_coeff = np.array(csv_file["c"])
h0 = np.array(csv_file["h0"])
v0 = np.array(csv_file["v0"])
y0 = np.array(csv_file["y0"])
rho0 = np.array(csv_file["rho0"])
hddot0 = np.array(csv_file["hddot0"])
tf_lamb = np.array(csv_file["tf_lamb"])

a_coeff90 = a_coeff[0:-1:5]
a_coeff95 = a_coeff[1:-1:5]
a_coeff100 = a_coeff[2:-1:5]
a_coeff105 = a_coeff[3:-1:5]
a_coeff110 = a_coeff[4:-1:5]

a_coeff90 = a_coeff90.tolist()[9:12] + a_coeff90.tolist()[0:9]
a_coeff95 = a_coeff95.tolist()[9:12] + a_coeff95.tolist()[0:9]
a_coeff100 = a_coeff100.tolist()[9:12] + a_coeff100.tolist()[0:9]
a_coeff105 = a_coeff105.tolist()[9:12] + a_coeff105.tolist()[0:9]
a_coeff110 = a_coeff110.tolist()[9:12] + a_coeff110.tolist()[0:9]


rho090 = rho0[0:-1:5]
rho095 = rho0[1:-1:5]
rho0100 = rho0[2:-1:5]
rho0105 = rho0[3:-1:5]
rho0110 = rho0[4:-1:5]

rho090 = rho090.tolist()[9:12] + rho090.tolist()[0:9]
rho095 = rho095.tolist()[9:12] + rho095.tolist()[0:9]
rho0100 = rho0100.tolist()[9:12] + rho0100.tolist()[0:9]
rho0105 = rho0105.tolist()[9:12] + rho0105.tolist()[0:9]
rho0110 = rho0110.tolist()[9:12] + rho0110.tolist()[0:9]

v090 = v0[0:-1:5]
v095 = v0[1:-1:5]
v0100 = v0[2:-1:5]
v0105 = v0[3:-1:5]
v0110 = v0[4:-1:5]

v090 = v090.tolist()[9:12] + v090.tolist()[0:9]
v095 = v095.tolist()[9:12] + v095.tolist()[0:9]
v0100 = v0100.tolist()[9:12] + v0100.tolist()[0:9]
v0105 = v0105.tolist()[9:12] + v0105.tolist()[0:9]
v0110 = v0110.tolist()[9:12] + v0110.tolist()[0:9]


# fig_acoeff = plt.figure("a_coeff", dpi=500, figsize=[6,4])
# plt.figure("a_coeff")
# plt.scatter(y0[0:-1:5], a_coeff90, color='green', marker='o', label = '$h_p = 90$ km')   
# plt.scatter(y0[1:-1:5], a_coeff95, color='red', marker='o', label = '$h_p = 95$ km')   
# plt.scatter(y0[2:-1:5], a_coeff100, color='blue', marker='o',  label = '$h_p = 100$ km')   
# plt.scatter(y0[3:-1:5], a_coeff105, color='orange', marker='o', label = '$h_p = 105$ km')   
# plt.scatter(y0[4:-1:5], a_coeff110, color='purple', marker='o',  label = '$h_p = 110$ km')   
# # plt.scatter(y0, a_coeff, color='blue', marker='o')    
# plt.legend();plt.grid()
# plt.xlabel("$y_0$")
# plt.ylabel("$a$")
# fig_acoeff.savefig("all_sims_figures/a_coeff_y0.png")

# X = [rho0[0:-1:5],rho0[1:-1:5],rho0[2:-1:5],rho0[3:-1:5],rho0[4:-1:5]]
# Y = [a_coeff90,a_coeff90,a_coeff90,a_coeff90,a_coeff90]
# Z = [v0[0:-1:5],v0[1:-1:5],v0[2:-1:5],v0[3:-1:5],v0[4:-1:5]]

# fig_contour = plt.figure("contour", dpi=500, figsize=[6,4])
# plt.figure("contour")
# CS = plt.tricontourf(a_coeff, rho0, v0)
# CB = fig_contour.colorbar(CS)
# # plt.scatter(rho0[0:-1:5], a_coeff90, color='green', marker='o', label = '$h_p = 90$ km')   
# # plt.scatter(rho0[1:-1:5], a_coeff95, color='red', marker='o', label = '$h_p = 95$ km')   
# # plt.scatter(rho0[2:-1:5], a_coeff100, color='blue', marker='o',  label = '$h_p = 100$ km')   
# # plt.scatter(rho0[3:-1:5], a_coeff105, color='orange', marker='o', label = '$h_p = 105$ km')   
# # plt.scatter(rho0[4:-1:5], a_coeff110, color='purple', marker='o',  label = '$h_p = 110$ km')   
# plt.legend();plt.grid()
# plt.xlabel("Initial Density")
# plt.ylabel("$a$")
# fig_contour.savefig("all_sims_figures/contour.png")


# fig_3D = plt.figure("3d", dpi=500, figsize=[6,4])
fig_3D = plt.figure("3d")
ax = fig_3D.add_subplot(projection = '3d')
# CS = ax.plot_trisurf(rho0, a_coeff, v0, cmap=plt.cm.Spectral, linewidth=1, antialiased=True)
# CB = fig_3D.colorbar(CS)
ax.plot(rho090, a_coeff90, v090,'b.-', label = '$h_p = 90$ km')   
ax.plot(rho095, a_coeff95, v095,'g.-', label = '$h_p = 95$ km')   
ax.plot(rho0100, a_coeff100,v0100, 'r.-',  label = '$h_p = 100$ km')   
ax.plot(rho0105, a_coeff105, v0105,'k.-', label = '$h_p = 105$ km')   
ax.plot(rho0110, a_coeff110, v0110,'c.-',  label = '$h_p = 110$ km')  


for i in range(12):
    print(i)

    r = [rho090[i], rho095[i], rho0100[i], rho0105[i], rho0110[i]]
    a = [a_coeff90[i], a_coeff95[i], a_coeff100[i], a_coeff105[i], a_coeff110[i]]
    v = [v090[i], v095[i], v0100[i], v0105[i], v0110[i]]

    ax.plot(r, a, v,'k-')


# ax.set_ylabel("$a$",size='x-large')
# ax.set_xlabel("$D_{max}$ (N)",size='x-large')
plt.xlabel("Density")
plt.ylabel("a")
# plt.bar_label("$v_0$")
plt.legend(fontsize=16)
plt.grid()
plt.show()
fig_3D.savefig("all_sims_figures/3d.png")