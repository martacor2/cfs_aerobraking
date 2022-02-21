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

ra = np.array(csv_file["Simulation"])
hp = np.array(csv_file["hp"])
a_coeff = np.array(csv_file["a"])
b_coeff = np.array(csv_file["b"])
c_coeff = np.array(csv_file["c"])
h0 = np.array(csv_file["h0"])
v0 = np.array(csv_file["v0"])
y0 = np.array(csv_file["y0"])
rho0 = np.array(csv_file["rho0"])
hddot0 = np.array(csv_file["hddot0"])
tf_sim = np.array(csv_file["tf_sim"])
drag0 = np.array(csv_file["Initial drag"])
drag_int = np.array(csv_file["Integral drag"])
drag_int_no_v = np.array(csv_file["Integral drag no v"])
drag_int_hor = np.array(csv_file["Integral hor drag"])
drag_int_ver = np.array(csv_file["Integral ver drag"])
drag_int = np.array(csv_file["Integral drag"])
vp = np.array(csv_file["Periapsis velocity"])

a_axis = np.divide(np.subtract(np.multiply(ra,1000),np.add(np.multiply(hp,1000),Rp)),2)
# energy = a_axis
energy = np.multiply(1/(Sref*(CD_0*aoa)), np.divide(drag0,rho0))
# energy = np.divide(-mu/2, a_axis)

def array_conversion(x):
    x90 = x[0:-1:5]
    x95 = x[1:-1:5]
    x100 = x[2:-1:5]
    x105 = x[3:-1:5]
    x110 = x[4:-1:5]

    x90 = x90.tolist()[9:12] + x90.tolist()[0:9]
    x95 = x95.tolist()[9:12] + x95.tolist()[0:9]
    x100 = x100.tolist()[9:12] + x100.tolist()[0:9]
    x105 = x105.tolist()[9:12] + x105.tolist()[0:9]
    x110 = x110.tolist()[9:12] + x110.tolist()[0:9]

    return (x90, x95, x100, x105, x110)


t90, t95, t100, t105, t110 = array_conversion(tf_sim)
vp90, vp95, vp100, vp105, vp110 = array_conversion(vp)
a_coeff90, a_coeff95, a_coeff100, a_coeff105, a_coeff110 = array_conversion(a_coeff)
b_coeff90, b_coeff95, b_coeff100, b_coeff105, b_coeff110 = array_conversion(b_coeff)
c_coeff90, c_coeff95, c_coeff100, c_coeff105, c_coeff110 = array_conversion(c_coeff)
h090, h095, h0100, h0105, h0110 = array_conversion(h0)
v090, v095, v0100, v0105, v0110 = array_conversion(v0)
y090, y095, y0100, y0105, y0110 = array_conversion(y0)
rho090, rho095, rho0100, rho0105, rho0110 = array_conversion(rho0)
hddot090, hddot095, hddot0100, hddot0105, hddot0110 = array_conversion(hddot0)
drag090, drag095, drag0100, drag0105, drag0110 = array_conversion(drag0)
energy90, energy95, energy100, energy105, energy110 = array_conversion(energy)
drag_int90, drag_int95, drag_int100, drag_int105, drag_int110 = array_conversion(drag_int)
drag_int_no_v90, drag_int_no_v95, drag_int_no_v100, drag_int_no_v105, drag_int_no_v110 = array_conversion(drag_int_no_v)
drag_int_ver90, drag_int_ver95, drag_int_ver100, drag_int_ver105, drag_int_ver110 = array_conversion(drag_int_ver)
drag_int_hor90, drag_int_hor95, drag_int_hor100, drag_int_hor105, drag_int_hor110 = array_conversion(drag_int_hor)


# fig_acoeff = plt.figure("a_coeff", dpi=500, figsize=[6,4])
# plt.figure("a_coeff")
# plt.scatter(energy90, a_coeff90, color='green', marker='o', label = '$h_p = 90$ km')   
# plt.scatter(energy95, a_coeff95, color='red', marker='o', label = '$h_p = 95$ km')   
# plt.scatter(energy100, a_coeff100, color='blue', marker='o',  label = '$h_p = 100$ km')   
# plt.scatter(energy105, a_coeff105, color='orange', marker='o', label = '$h_p = 105$ km')   
# plt.scatter(energy110, a_coeff110, color='purple', marker='o',  label = '$h_p = 110$ km')   
# # plt.scatter(y0, a_coeff, color='blue', marker='o')    
# plt.legend();plt.grid()
# plt.xlabel("Specific Energy")
# plt.ylabel("$a$")
# fig_acoeff.savefig("all_sims_figures/a_coeff_specific_energy.png")

fig_acoeff = plt.figure("a_coeff", dpi=500, figsize=[6,4])
plt.figure("a_coeff")

# for i in range(0, 12):
#     # x = [drag_int90[i], drag_int95[i], drag_int100[i], drag_int105[i], drag_int110[i]]
#     x = [drag_int90[i], drag_int110[i]]

#     # y = [a_coeff90[i], a_coeff95[i], a_coeff100[i], a_coeff105[i], a_coeff110[i]]
#     y = [a_coeff90[i], a_coeff110[i]]

#     plt.plot(x, y,'k--', linewidth = "1")

# plt.scatter(y090, c_coeff90, color='green', marker='o', label = '$h_p = 90$ km')   
# plt.scatter(y095, c_coeff95, color='red', marker='o', label = '$h_p = 95$ km')   
# plt.scatter(y0100, c_coeff100, color='blue', marker='o',  label = '$h_p = 100$ km')   
# plt.scatter(y0105, c_coeff105, color='orange', marker='o', label = '$h_p = 105$ km')   
# plt.scatter(y0110, c_coeff110, color='purple', marker='o',  label = '$h_p = 110$ km')  

# # plt.scatter(y0, a_coeff, color='blue', marker='o')    
# plt.legend();plt.grid()
# plt.xlabel(r"$\gamma_0$ (rad)")
# plt.ylabel("$c$")
# fig_acoeff.savefig("all_views/c_coefficient/c_coeff_y0.png")


# plt.scatter(drag_int_no_v90, a_coeff90, color='green', marker='o', label = '$h_p = 90$ km')   
plt.scatter(drag_int_no_v95, a_coeff95, color='red', marker='o', label = '$h_p = 95$ km')   
plt.scatter(drag_int_no_v100, a_coeff100, color='blue', marker='o',  label = '$h_p = 100$ km')   
plt.scatter(drag_int_no_v105, a_coeff105, color='orange', marker='o', label = '$h_p = 105$ km')   
plt.scatter(drag_int_no_v110, a_coeff110, color='purple', marker='o',  label = '$h_p = 110$ km')  

# linear_fit = np.polyfit( drag_int_ver,a_coeff, 1)
# m = linear_fit[0]
# b = linear_fit[1]

# x_plot = np.linspace(np.min(drag_int_ver), np.max(drag_int_ver), 1000)
# y_plot = np.add(b, np.multiply(x_plot,m))

# plt.plot(x_plot,y_plot, label = f"y = {np.round(m,5)}x + {np.round(b,5)}" )

plt.legend();plt.grid()
plt.xlabel(r"Integral of Drag w/o velocity (N s$^2$/m$^2$)")
plt.ylabel("$a$")
fig_acoeff.savefig("all_views/a_coefficient/a_coeff_drag_integral_no_v.png")

# X = [v090,v095,v0100,v0105,v0110]
# X = [y090,y095,y0100,y0105,y0110]
# X = [rho090,rho095,rho0100,rho0105,rho0110]
X = [energy90,energy95,energy100,energy105,energy110]
Y = [drag090,drag095,drag0100,drag0105,drag0110]
Z = [a_coeff90,a_coeff95,a_coeff100,a_coeff105,a_coeff110]
# Z = [c_coeff90,c_coeff95,c_coeff100,c_coeff105,c_coeff110]


# fig_contour = plt.figure("contour", dpi=500, figsize=[6,4])
# plt.figure("contour")
# CS = plt.contourf(X, Y, Z, 15)
# CB = fig_contour.colorbar(CS)
# # plt.scatter(rho090, drag090, color='green', marker='o', label = '$h_p = 90$ km')   
# # plt.scatter(rho095, drag095, color='red', marker='o', label = '$h_p = 95$ km')   
# # plt.scatter(rho0100,drag0100, color='blue', marker='o',  label = '$h_p = 100$ km')   
# # plt.scatter(rho0105, drag0105, color='orange', marker='o', label = '$h_p = 105$ km')   
# # plt.scatter(rho0110, drag0110, color='purple', marker='o',  label = '$h_p = 110$ km')   
# plt.grid()
# # plt.xlabel(r'$\rho_0$')
# plt.xlabel("Specific Kinetic Energy")
# plt.ylabel("$D_0$")
# CB.ax.set_ylabel("$a$")
# # CB.ax.set_ylabel(r"$\rho_0$")
# fig_contour.savefig("all_sims_figures/contour_a_D0_asem.png")


# # fig_3D = plt.figure("3d", dpi=500, figsize=[6,4])
# fig_3D = plt.figure("3d")
# ax = fig_3D.add_subplot(projection = '3d')
# # CS = ax.plot_trisurf(rho0, a_coeff, v0, cmap=plt.cm.Spectral, linewidth=1, antialiased=True)
# # CB = fig_3D.colorbar(CS)
# ax.plot(energy90[1:], drag090[1:], a_coeff90[1:],'b.-', label = '$h_p = 90$ km')   
# ax.plot(energy95[1:], drag095[1:], a_coeff95[1:],'g.-', label = '$h_p = 95$ km')   
# ax.plot(energy100[1:], drag0100[1:],a_coeff100[1:], 'r.-',  label = '$h_p = 100$ km')   
# ax.plot(energy105[1:], drag0105[1:], a_coeff105[1:],'m.-', label = '$h_p = 105$ km')   
# ax.plot(energy110[1:], drag0110[1:], a_coeff110[1:],'c.-',  label = '$h_p = 110$ km')  


# for i in range(1, 12):
#     # print(i)

#     r = [energy90[i], energy95[i], energy100[i], energy105[i], energy110[i]]
#     v = [drag090[i], drag095[i], drag0100[i], drag0105[i], drag0110[i]]
#     # v = [y090[i], y095[i], y0100[i], y0105[i], y0110[i]]
#     a = [a_coeff90[i], a_coeff95[i], a_coeff100[i], a_coeff105[i], a_coeff110[i]]
#     # v = [v090[i], v095[i], v0100[i], v0105[i], v0110[i]]

#     ax.plot(r, v, a,'k-')


# # ax.set_ylabel("$a$",size='x-large')
# # ax.set_xlabel("$D_{max}$ (N)",size='x-large')
# plt.xlabel("Specific kinetic energy")
# plt.ylabel("Initial Drag")
# ax.set_zlabel("$a$")
# # plt.bar_label("$v_0$")
# plt.legend(fontsize=16)
# plt.grid()
# # plt.show()
# fig_3D.savefig("all_sims_figures/3d.png")