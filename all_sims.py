# -*- coding: utf-8 -*-
"""
Created on Sat Jul  3 11:59:41 2021

@author: marta
"""

import numpy as np
from numpy.linalg import inv
from numpy.linalg import norm
import pandas as pd
import matplotlib.pyplot as plt
import os
from lamberts_problem import *
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

diff_tot = []
target_array = []
time_array = []

ra = [5000, 7000, 9000, 11000, 13000, 15000, 17000, 19000, 21000, 23000, 25000, 27000]

for folder in sorted(os.listdir("DragPassageResults")):
    
    final_diff=[]
    
    for folder2 in sorted(os.listdir(os.path.join("DragPassageResults",folder))):
        #check each folder in the results folder
        for file in sorted(os.listdir(os.path.join("DragPassageResults",folder,folder2))):
            #velocity from simulation at aoa=pi/2
            if(file.endswith("csv")):
                csv_file=pd.read_csv(os.path.join("DragPassageResults",folder,folder2,file))
            files=data_gathering(csv_file)
            #altitude
            h_sim, v_sim, y_sim, f_sim, a_sim, e_sim, r_sim, t_sim, rho_sim = files[0:9]
        
            h0,v0,y0,f0,a0,e0,r0,t0,rho0 = files[9]
            hf,vf,yf,ff,af,ef,rf,tf,rhof = files[10]
            
            index = np.argmin(h_sim)    
            #periapsis data using altitude
            hp=h_sim[index]
            vp=v_sim[index]
            yp=y_sim[index]
            fp=f_sim[index]
            tp_sim=t_sim[index]-t0
                
            files2=perturbation_data(csv_file)
            Grav1_sim, Grav2_sim, Grav3_sim, Drag1_sim, Drag2_sim, Drag3_sim, Lift1_sim, Lift2_sim, Lift3_sim, Omega_sim, w_sim, SAM_sim= files2[0:12]
        
            E_sim = np.multiply(np.arctan(np.multiply(np.sqrt((np.divide(np.subtract(1,np.array(e_sim)), np.add(1, np.array(e_sim))))),np.tan(np.divide(f_sim, 2)))),2)
            
        
        #from equations of planar motion of a point mass for a non-rotating spherical planet
        
        #fpa dot
        def fpa_rate(CL_0,Sref,v,rho,Rp,h):
            y_rate=[]
            for i in range(len(rho)):
                g = mu/(Rp+h[i])**2
                y_rate_sim=rho[i]*CL_0*Sref/(2*m)*v[i] - g/v[i] + v[i]/(Rp+h[i])
                y_rate.append(y_rate_sim)
            return y_rate
        
        #h dot
        def h_rate(v,y):
            h_rate=[]
            for i in range(len(v)):
                h_rate_sim=v[i]*np.sin(y[i])
                h_rate.append(h_rate_sim)
            return h_rate
        
        #v dot
        def v_rate(CD_0,Sref,v,rho,y,h):
            v_rate=[]
            for i in range(len(rho_sim)):
                g = mu/(Rp+h[i])**2
                v_rate_sim = -rho[i]*CD_0*aoa*Sref/(2*m)*(v[i]**2) - g*(y[i])
                v_rate.append(v_rate_sim)
            return v_rate
        
        #h doubledot
        def hddot_simulation(CL_0,CD_0,Sref,v,rho,Rp,h,y):
            hd_rate=[]
            for i in range(len(rho)):
                g = mu/(Rp+h[i])**2
                v_rate_sim = -rho[i]*CD_0*aoa*Sref/(2*m)*(v[i]**2) - g*y[i]
                y_rate_sim = rho[i]*CL_0*Sref/(2*m)*v[i] - g/v[i] + v[i]/(Rp+h[i])
                
                hd_rate_sim = (v_rate_sim)*y[i] + v[i]*(y_rate_sim)
                hd_rate.append(hd_rate_sim)
        
            return hd_rate
        
        #h doubledot full form
        def hddot_full_form(D,L,y,v,h):
            hd_rate=[]
            for i in range(len(D)):
                g = mu/(Rp+h[i])**2
                
                v_rate_sim = -D[i]/m - g*np.sin(y[i])
                y_rate_sim = L[i]/(v[i]*m) - g/v[i]*np.cos(y[i]) + v[i]*np.cos(y[i])/(Rp+h[i])
                
                hd_rate_sim = (v_rate_sim)*np.sin(y[i]) + v[i]*(y_rate_sim)*np.cos(y[i])
                hd_rate.append(hd_rate_sim)
        
            return hd_rate
        
        #h doubledot with small fpa assumption
        def hddot_small_fpa(CL_0,CD_0,Sref,v,rho,Rp,h,y):
            hd_rate=[]
            for i in range(len(rho)):
                g = mu/(Rp+h[i])**2
                v_rate_sim = -rho[i]*CD_0*aoa*Sref/(2*m)*(v[i]**2) - g*(y[i])
                y_rate_sim = rho[i]*CL_0*Sref/(2*m)*v[i] - g/v[i] + v[i]/(Rp+h[i])
                
                hd_rate_sim = (v_rate_sim)*y[i] + v[i]*(y_rate_sim)
                hd_rate.append(hd_rate_sim)
            return hd_rate
        
        #h doubledot with all assumptions
        def hddot_approx(CL_0,CD_0,Sref,v,rho,Rp,h,y):
            hd_rate=[]
            for i in range(len(rho)):
                g = mu/(Rp+h[i])**2
                
                hd_rate_sim = -rho[i]*CD_0*aoa*Sref/(2*m)*(v[i]**2)*(y[i]) + rho[i]*CL_0*Sref/(2*m)*v[i]**2 - g \
                    + v[i]**2/(Rp+h[i])
                
                hd_rate.append(hd_rate_sim)
            return hd_rate
        
        #rddot calulcations
        def rddot_OM(h,sam):
            rate = []
            for i in range(len(h)):
                g = mu/(Rp+h[i])**2       
                rddot = -g + sam[i]**2/(Rp+h[i])**3
                rate.append(rddot)     
            return rate
        
        def rddot_RE(h,v,y):
            rate = []
            rho_0 = 0.020 #kg/m^3
            LD = CL_0/(CD_0*aoa)
            Beta = 2*m/(CD_0*aoa*rho_0*Sref*Rp)
            
            for i in range(len(h)):
                g = mu/(Rp+h[i])**2  
                rho=rho_ref*np.exp(np.divide(90000-h[i],H))
                
                rddot = -g + ((rho/rho_0)*v[i]**2)/(Beta*Rp)*(LD - y[i]) \
                + v[i]**2/(h[i]+Rp)
                
                rate.append(rddot)      
            return rate
        
        
        hddot_array=[]
        time_array=[]
        c_sim_array=[]
        amplitude_ra=[]
        start_ra=[]
        
        diff_array = []
        
        grav_sim=[]
        drag_sim=[]
        lift_sim=[]
        
        for i in range(len(Drag1_sim)):
            
            grav = norm([Grav1_sim[i], Grav2_sim[i], Grav3_sim[i]])
            drag = norm([Drag1_sim[i], Drag2_sim[i], Drag3_sim[i]])
            lift = norm([Lift1_sim[i], Lift2_sim[i], Lift3_sim[i]])
            
            grav_sim.append(grav)
            drag_sim.append(drag)
            lift_sim.append(lift)
        
        
        #simualtion time
        t_simulation=np.subtract(t_sim,t0)
        f0=f0-2*np.pi
        
        nu_list = np.linspace(f0,-1*f0,1000)
        
        #Lambert's problem time (method 1)
        t_lamb=Lambert2(nu_list,e0,a0,mu,Rp)[1]
        tp_lamb = t_lamb[-1]/2
        tf_lamb = Lambert(f0,e0,a0,mu)[0]
        h_lambert = Lambert2(nu_list,e0,a0,mu,Rp)[0]
        v_lambert = Lambert2(nu_list,e0,a0,mu,Rp)[2]
        y_lambert = Lambert2(nu_list,e0,a0,mu,Rp)[3]
        
        #Lambert's problem time (method 2)
        # t_lamb= Lambert3(t_simulation,e0,a0,mu,Rp,f0)[3]
        # tp_lamb = t_lamb[-1]/2
        # tf_lamb = Lambert(f0,e0,a0,mu)[0]
        # h_lambert = Lambert3(t_simulation,e0,a0,mu,Rp,f0)[0]
        # v_lambert = Lambert3(t_simulation,e0,a0,mu,Rp,f0)[1]
        # y_lambert = Lambert3(t_simulation,e0,a0,mu,Rp,f0)[2]
        # nu_array = Lambert3(t_simulation,e0,a0,mu,Rp,f0)[4]
        
        SAM_lambert = np.multiply((Rp + h_lambert[0])*v_lambert[0],np.ones(np.size(t_lamb)))
        
        #SAM_lambert = np.multiply((Rp + h_lambert[0])*v_lambert[0]*np.cos(y_lambert[0]),np.ones(np.size(t_lamb)))            
    
        #closed form solution
        # c_lamb = np.multiply(-v0*y0/tp_lamb,np.ones(np.size(t_lamb)))
        # alt_cfs=ALT(t_lamb,tp_lamb,v0,y0,h0)
        # vel_km1=np.multiply(-1,np.abs(VEL4(t_lamb,tp_lamb,alt_cfs,v0,y0,h0,aoa,tf_lamb,c_lamb)[0]))
        # vel_cfs=np.add(vel_km1,v0-vel_km1[0])
        # y_cfs=FPA(t_lamb,tp_lamb,v0,y0,h0,vel_cfs)
        
        # fig3, ax3=plt.subplots()
        # ax3.plot(t_lamb,vel_cfs,label="4th order")
        # ax3.plot(t_simulation,v_sim,label="Simulation")
        # ax3.set_title("Velocity",size='x-large')
        # ax3.set_xlim(0)
        # ax3.set_ylabel("Velocity (km/s)",size='x-large')
        # ax3.set_xlabel("Time (sec)",size='x-large')
        # fig3.set_size_inches(10,5)
        # plt.tick_params(labelsize=16)
        # plt.legend(fontsize=16)
        # ax3.grid()
        
        hddot = hddot_small_fpa(CL_0,CD_0,Sref,v_sim,rho_sim,Rp,h_sim,y_sim)
        hddot_sim = hddot_simulation(CL_0,CD_0,Sref,v_sim,rho_sim,Rp,h_sim,y_sim)
        hddot_app = hddot_approx(CL_0,CD_0,Sref,v_sim,rho_sim,Rp,h_sim,y_sim)
        
        hddot_ff = hddot_full_form(drag_sim,lift_sim,y_sim,v_sim,h_sim)
        
        rddotOM = rddot_OM(h_sim,SAM_sim)
        rddotRE = rddot_RE(h_lambert,v_lambert,y_lambert)
        rddotLAMB = rddot_OM(h_lambert,SAM_lambert)
        
        rddotLAMB_w_LD = rddot_RE(h_lambert,v_lambert,y_lambert)
        
        
        c_lamb = np.multiply(-v0*y0/tp_lamb,np.ones(np.size(t_lamb)))
        alt_cfs=ALT(t_simulation,tp_sim,v0,y0,h0)
        vel_km5 = VEL2(t_simulation,tp_sim,alt_cfs,v0,y0,h0,aoa,t_simulation[-1], hddot_app)[0]
        # vel_cfs=np.add(vel_km5,v0-vel_km5[0])
        y_cfs=FPA(t_simulation,tp_sim,v0,y0,h0,vel_km5)

        
        # fig2, ax2=plt.subplots()
        # # # ax2.plot(t_simulation,hddot,color='black', label="$\ddot{h}_{small-angle}$")
        # # ax2.plot(t_simulation,hddot_sim,color='blue', label="$\ddot{h}$")
        # ax2.plot(t_simulation,hddot_ff,color='red', label="$\ddot{h}_{full-form}$")
        # # ax2.plot(t_simulation,hddot_app,color='green', label="$\ddot{h}_{approx}$")
        # # ax2.plot(t_simulation, rddotOM, color='green', label="$\ddot{r}_{OM}$")
        # # ax2.plot(t_simulation, rddotRE, color='orange', label="$\ddot{r}_{RE}$")
        # ax2.plot(t_lamb, rddotLAMB, color='grey', label="$\ddot{r}_{Lambert}$")
        # # ax2.plot(t_lamb, rddotLAMB_w_LD, color='pink', label="$\ddot{r}_{Lambert}$ with LD")
        # ax2.plot(tp_sim,hddot_ff[index],'o',color='orange',markersize=10)
        # ax2.set_title("$\ddot{h}$ vs time | "+folder2[15:-13],size='xx-large')
        # ax2.set_xlim(0)
        # ax2.set_ylabel("$\ddot{h}$ (m/s)",size='x-large')
        # ax2.set_xlabel("Time (sec)",size='x-large')
        # ax2.legend(fontsize=16)
        # fig2.set_size_inches(10,5)
        # plt.tick_params(labelsize=14)
        # ax2.grid()
        
        
        #should probably pass into y^2 one
        vel_km6 = VEL2(t_simulation,tp_sim,alt_cfs,v0,y0,h0,aoa,t_simulation[-1], hddot_sim)[0]
        
        diff = (rddotRE[-1]-hddot_app[-1])/rddotRE[-1]*100
        
        diff = (hddot_app[0]-hddot_app[-1])
        
        print(diff)
    
    
        
        
        # diff_array = np.subtract(np.subtract(rddotRE,rddotRE[0]),np.subtract(hddot_app,hddot_app[0]))
        
        # diff_array = np.subtract(rddotRE,hddot_app)
        
        # diff_array = np.subtract(np.subtract(rddotLAMB,rddotLAMB[0]),np.subtract(hddot_ff,hddot_ff[0]))
        

        # fig2, ax2=plt.subplots()
        # # # ax2.plot(t_simulation,hddot,color='black', label="$\ddot{h}_{small-angle}$")
        # # ax2.plot(t_simulation,hddot_sim,color='blue', label="$\ddot{h}$")
        # ax2.plot(t_diff,diff_array,color='red')
        # ax2.plot(time_target,target,'o',color='orange',markersize=10)
        # # ax2.plot(t_simulation, rddotOM, color='green', label="$\ddot{r}_{OM}$")
        # # ax2.plot(t_simulation, rddotRE, color='orange', label="$\ddot{r}_{RE}$")
        # # ax2.plot(t_lamb, rddotLAMB_w_LD, color='pink', label="$\ddot{r}_{Lambert}$ with LD")
        # ax2.set_title("$\ddot{h}$ vs time | "+folder2[15:-13],size='xx-large')
        # ax2.set_xlim(0)
        # ax2.set_ylabel("$\ddot{h}$ (m/s)",size='x-large')
        # ax2.set_xlabel("Time (sec)",size='x-large')
        # fig2.set_size_inches(10,5)
        # plt.tick_params(labelsize=14)
        # ax2.grid()
        
        #print(time_target)
        
        final_diff.append(diff)

    diff_tot.append([final_diff[3],final_diff[4],final_diff[0],final_diff[1],final_diff[2]])

    

r90 = []
r95 = []
r100 = []
r105 = []
r110 = []
    
for i in range(len(ra)):
    r90.append(diff_tot[i][0])
    r95.append(diff_tot[i][1])
    r100.append(diff_tot[i][2])
    r105.append(diff_tot[i][3])
    r110.append(diff_tot[i][4])
    

r90 = r90[9:12] + r90[0:9]
r95 = r95[9:12] + r95[0:9]
r100 = r100[9:12] + r100[0:9]
r105 = r105[9:12] + r105[0:9]
r110 = r110[9:12] + r110[0:9]


fig20, ax20=plt.subplots()
ax20.plot(ra,r90,color='black', linestyle='-', marker='.', markersize=12,  label="$r_{p} = 90$")
ax20.plot(ra,r95,color='purple', linestyle='-', marker='.', markersize=12,  label="$r_{p} = 95$")
ax20.plot(ra,r100,color='blue', linestyle='-', marker='.', markersize=12,  label="$r_{p}$ = 100")
ax20.plot(ra,r105,color='red', linestyle='-', marker='.', markersize=12,  label="$r_{p}$ = 105")
ax20.plot(ra,r110,color='green', linestyle='-', marker='.', markersize=12,  label="$r_{p}$ = 110")
ax20.set_xlabel("$r_{a}$ (km)",size='x-large')
fig20.set_size_inches(10,12)
plt.tick_params(labelsize=14)
plt.legend(fontsize=16)
ax20.grid()


    
    
        