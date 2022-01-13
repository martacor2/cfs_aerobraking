# -*- coding: utf-8 -*-
"""
Created on Tue Apr 13 15:52:55 2021

@author: marta
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from lamberts_problem import Lambert
from data_analysis import *
from closed_form_solution import *

amplitude=[]
start=[]
c_lamb=[]
c_sim=[]
energy_d=[]

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


#change path to appropriate folder
folder='9000'
folder2='Results_ctrl=0_ra=9000_rp=100.0_hl=0.1_90deg'

directory = "DragPassageResults/"+folder+"/"+folder2

#check each folder in the results folder
for file in sorted(os.listdir(directory)):
    #velocity from simulation at aoa=pi/2
    if(file.endswith("csv")):
        csv_file=pd.read_csv(os.path.join(direct,folder,folder2,file))
    files=data_gathering(csv_file)
    #altitude
    h_sim, v_sim, y_sim, f_sim, a_sim, e_sim, r_sim, t_sim = files[0:8]

    h0,v0,y0,f0,a0,e0,r0,t0 = files[8]
    hf,vf,yf,ff,af,ef,rf,tf = files[9]
    
    index = np.argmin(h_sim)    
    #periapsis data using altitude
    hp=h_sim[index]
    vp=v_sim[index]
    yp=y_sim[index]
    fp=f_sim[index]
    tp_sim=t_sim[index]-t0
        
    files2=perturbation_data(csv_file)
    v1_sim, v2_sim, v3_sim, rho_sim, Drag1_sim, Drag2_sim, Drag3_sim, Omega_sim, w_sim, i_sim= files2[0:10]

    v1_0,v2_0,v3_0,rho0,drag1_0,drag2_0,drag3_0,Omega0,w0_1,i0 = files2[10]
    v1_f,v2_f,v3_f,rhof,drag1_f,drag2_f,drag3_f,Omegaf,wf_1,i_f = files2[11]
    E_sim = np.multiply(np.arctan(np.multiply(np.sqrt((np.divide(np.subtract(1,np.array(e_sim)), np.add(1, np.array(e_sim))))),np.tan(np.divide(f_sim, 2)))),2)

#functions to help calculate the simulation h double_dot
#y_dot
def fpa_rate(CL_0,Sref,v_sim,rho_sim,Rp,h_sim):
    y_rate=[]
    
    for i in range(len(rho_sim)):
        y_rate_sim=rho_sim[i]*CL_0*Sref/(2*m)*v_sim[i] - g_ref/v_sim[i] + v_sim[i]/(Rp+h_sim[i])
        y_rate.append(y_rate_sim)

    return y_rate
#h_dot
def h_rate(v_sim,y_sim):
    h_rate=[]
    
    for i in range(len(v_sim)):
        h_rate_sim=v_sim[i]*np.sin(y_sim[i])
        h_rate.append(h_rate_sim)

    return h_rate
#v_dot
def v_rate(CD_0,Sref,v_sim,rho_sim,y_sim):
    v_rate=[]
    
    for i in range(len(rho_sim)):
        v_rate_sim=-rho_sim[i]*CD_0*Sref/(2*m)*(v_sim[i]**2) - g_ref*(y_sim[i])
        v_rate.append(v_rate_sim)

    return v_rate

def hddot_full_form(CL_0,CD_0,Sref,v_sim,rho_sim,Rp,h_sim,y_sim):
    hd_rate=[]
    
    for i in range(len(rho_sim)):
        hd_rate_sim=-rho_sim[i]*CD_0*Sref/(2*m)*(v_sim[i]**2)*y_sim[i] - g_ref*(y_sim[i])**2 + \
        rho_sim[i]*CL_0*Sref/(2*m)*(v_sim[i]**2) - g_ref + v_sim[i]**2/(Rp+h_sim[i])
        
        hd_rate.append(hd_rate_sim)

    return hd_rate

def cfs_hddot(CL_0,CD_0,Sref,v,rho,Rp,h,y):
    hd_rate=[]
    
    for i in range(len(rho_sim)):
        hd_rate_sim=-rho_sim[i]*CD_0*Sref/(2*m)*(v_sim[i]**2)*y_sim[i] - g_ref*(y_sim[i])**2 + \
        rho_sim[i]*CL_0*Sref/(2*m)*(v_sim[i]**2) - g_ref + v_sim[i]**2/(Rp+h_sim[i])
        
        hd_rate.append(hd_rate_sim)

    return hd_rate

def approx_hddot(x):     
    y = (-4.60429*10**(-9))*x**2 + 2.1316643847*10**(-6)*x +0.00135
    return y


hddot_array=[]
time_array=[]
c_sim_array=[]
amplitude_ra=[]
start_ra=[]

#simualtion time
t_simulation=np.subtract(t_sim,t0)
f0=f0-2*np.pi
#Lambert's problem time
t_lamb=np.linspace(0,Lambert(f0,e0,a0,mu)[0],1000)
tp_lamb = Lambert(f0,e0,a0,mu)[1]
        
#without drag
alt_lamb=ALT(t_lamb,tp_lamb,v0,y0,h0)
vel_km1=np.multiply(-1,np.abs(V(t_lamb,tp_lamb,alt_lamb,v0,y0,h0,aoa)[0]))
vel_lamb=np.add(vel_km1,v0-vel_km1[0])
y_lamb=y(t_lamb,tp_lamb,v0,y0,h0,vel_lamb)


alt_rate_sim=h_rate(v_sim,y_sim)
v_rate_sim= v_rate(CD_0,Sref,v_sim,rho_sim,y_sim)
y_rate_sim = fpa_rate(CL_0,Sref,v_sim,rho_sim,Rp,h_sim)

h_ddot = np.add(np.multiply(v_rate_sim,np.sin(y_sim)),np.multiply(y_rate_sim,(np.multiply(v_sim,np.cos(y_sim)))))
#assuming small angle
h_ddot_small = sim_hddot(CL_0,CD_0,Sref,v_sim,rho_sim,Rp,h_sim,y_sim)

fig2, ax2=plt.subplots()
ax2.plot(t_simulation,h_ddot,color='blue', label="Simulation")
ax2.plot(t_simulation,h_ddot_small,color='r', label = "Simulation - small Angle")
ax2.plot(tp_sim,h_ddot[index],'o',color='orange',markersize=10)
ax2.plot(t_simulation,approx_hddot(t_simulation),color='orange', label='parabola approximation')
ax2.set_title("$\ddot{h}$ vs time | "+folder2[15:-13],size='xx-large')
ax2.set_xlim(0)
ax2.set_ylabel("$\ddot{h}$",size='x-large')
ax2.set_xlabel("Time (sec)",size='x-large')
ax2.legend(fontsize=16)
fig2.set_size_inches(8,12)
plt.tick_params(labelsize=14)
ax2.grid()


vel_km2=np.abs(VEL(t_simulation,tp_sim,h_sim,v0,y0,h0,aoa,tf,h_ddot)[0])
vel_new=np.add(vel_km2,v0-vel_km2[0])

fig3, ax3=plt.subplots()
ax3.plot(t_lamb,vel_lamb,label="CFS 1")
ax3.plot(t_simulation,vel_new,label="CFS 2")
ax3.plot(t_simulation,v_sim,label="Simulation")
ax3.set_title("Velocity",size='x-large')
ax3.set_xlim(0)
ax3.set_ylabel("Velocity (km/s)",size='x-large')
ax3.set_xlabel("Time (sec)",size='x-large')
fig3.set_size_inches(10,5)
plt.tick_params(labelsize=16)
plt.legend(fontsize=16)
ax3.grid()





