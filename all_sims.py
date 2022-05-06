# -*- coding: utf-8 -*-
"""
Created on Tue Aug 31 13:07:16 2021

@author: Marta
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
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.tri as mtri
from scipy import integrate
#initialize arrays to store data
amplitude=[]
start=[]
c_lamb=[]
c_sim=[]
energy_array=[]

diff_array = []
diff_array_2 = []

a_tot=[]
b_tot=[]
c_tot=[]
tp_lamb_tot=[]
tf_sim_tot=[]
max_D_tot = []
hddot0_tot=[]
ICs_tot = []
drag_integral_tot = []
vp_tot = []
drag_integral_no_v_tot = []
drag_integral_hor_tot = []
drag_integral_ver_tot = []


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


diff_tot = []
target_array = []
time_array = []


ra  = np.array(range(5000,41000,1000))
rp = np.array(range(90,124,1))

ra_list = []


directory = "ClosedFormSol_MarsGram"
track = 0

for folder in sorted(os.listdir(directory)):
    
    final_diff=[]
    a_array=[]
    b_array=[]
    c_array=[]
    hddot0_array =[]
    ICs_array=[]
    tp_lamb_array=[]
    tf_sim_array=[]
    max_D_array=[]
    drag_integral = []
    vp_array = []

    drag_integral_no_v = []
    drag_integral_hor = []
    drag_integral_ver = []

    energy_ra = []

    rp_list = []

    if track ==6:
        break

    track = track+1


    for folder2 in sorted(os.listdir(os.path.join(directory,folder))):
        #check each folder in the results folder
        for file in sorted(os.listdir(os.path.join(directory,folder,folder2))):
            #velocity from simulation at aoa=pi/2
            if(file.endswith("csv")):
                csv_file=pd.read_csv(os.path.join(directory,folder,folder2,file))
            files=data_gathering(csv_file)
            #altitude
            h_sim, v_sim, y_sim, f_sim, a_sim, e_sim, r_sim, t_sim, rho_sim, energy_sim = files[0:10]
        
            h0,v0,y0,f0,a0,e0,r0,t0,rho0,energy0 = files[10]
            hf,vf,yf,ff,af,ef,rf,tf,rhof, energyf = files[11]
            
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
        
        ra_file = file.split("_")[2][3:]
        rp_file = file.split("_")[3][3:]
        
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
        
        def rho_exp(h):
            rho_list = []
            for i in range(len(h)):  
                rho=rho_ref*np.exp(np.divide(90000-h[i],H))
                rho_list.append(rho)
            return rho_list
        
        def parabola_app(pt1,pt2,pt3,time):
    
            A = np.array([ [pt1[0]**2,pt1[0],1], [pt2[0]**2,pt2[0],1] ,[pt3[0]**2,pt3[0],1] ])
            B =  np.array( [ pt1[1],pt2[1],pt3[1] ] )
            
            a,b,c = np.linalg.solve(A,B)
            
            par_app=[]
            
            for i in range(len(time)):
                par_app.append(a*(time[i]**2) + b*time[i] +c)
                
            return par_app
        
        
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
        #Lambert's problem time
        t_lamb= Lambert3(t_simulation,e0,a0,mu,Rp,f0)[3]
        tf_lamb = t_lamb[-1]
        
        delta_t_lamb = Lambert(f0,e0,a0,mu)[0]
        tp_lamb = delta_t_lamb/2
        
        h_lambert = Lambert3(t_simulation,e0,a0,mu,Rp,f0)[0]
        v_lambert = Lambert3(t_simulation,e0,a0,mu,Rp,f0)[1]
        y_lambert = Lambert3(t_simulation,e0,a0,mu,Rp,f0)[2]
        nu_array = Lambert3(t_simulation,e0,a0,mu,Rp,f0)[4]
        
        SAM_lambert = np.multiply((Rp + h_lambert[0])*v_lambert[0],np.ones(np.size(t_lamb)))
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
        
        
        diff_array = np.subtract(np.subtract(rddotLAMB,rddotLAMB[0]),np.subtract(hddot_ff,hddot_ff[0]))
        
        diff_array2 = np.subtract(np.subtract(rddotLAMB_w_LD,rddotLAMB_w_LD[0]),np.subtract(hddot_app,hddot_app[0]))
        
        diff_array2 = np.subtract(rddotLAMB_w_LD,hddot_app)
        
        t_diff = np.divide(t_lamb,delta_t_lamb)
        
        
        index_to_app = np.argmin(np.abs(diff_array2[50:]))
        
        target = np.min(np.abs(diff_array[100:]))
        time_target = t_diff[100:][np.argmin(np.abs(diff_array[100:]))]
        target2 = np.min(np.abs(diff_array2[50:]))
        time_target2 = t_diff[50:][index_to_app]
    
        
        # diff_array = np.subtract(np.subtract(rddotRE,rddotRE[0]),np.subtract(hddot_app,hddot_app[0]))
        
        # diff_array = np.subtract(rddotRE,hddot_app)
        
        # diff_array = np.subtract(np.subtract(rddotLAMB,rddotLAMB[0]),np.subtract(hddot_ff,hddot_ff[0]))

        pt1 = [t0,hddot_app[0]]
        pt2 = [delta_t_lamb*time_target2, hddot_app[50:][index_to_app] ]
        pt3 = [2*(delta_t_lamb*time_target2),hddot_app[0]]
        # pt3 = [delta_t_lamb,hddot_app[-1]]
        
        par_app = parabola_app(pt1,pt2,pt3,t_lamb)
        
        
        # fig2, ax2=plt.subplots()
        # # # ax2.plot(t_simulation,hddot,color='black', label="$\ddot{h}_{small-angle}$")
        # # ax2.plot(t_simulation,hddot_sim,color='blue', label="$\ddot{h}$")
        # # ax2.plot(t_simulation,hddot_ff,color='red', label="$\ddot{h}_{full-form}$")
        # ax2.plot(t_simulation,hddot_app,color='green', label="$\ddot{h}_{approx}$")
        
        # ax2.plot(t_lamb,par_app,color='red', label="Parabola")
        
        # # ax2.plot(t_simulation, rddotOM, color='cyan', label="$\ddot{r}_{OM}$")
        # # ax2.plot(t_simulation, rddotRE, color='orange', label="$\ddot{r}_{RE}$")
        # # ax2.plot(t_lamb, rddotLAMB, color='grey', label="$\ddot{r}_{Lambert}$")
        # ax2.plot(t_lamb, rddotLAMB_w_LD, color='blue', label="$\ddot{r}_{Lambert}$ with LD")
        # # ax2.plot(pt1[0],pt1[1], 'o',color='orange',markersize=10)
        # # ax2.plot(pt2[0],pt2[1], 'o',color='orange',markersize=10)
        # # ax2.plot(pt3[0],pt3[1], 'o',color='orange',markersize=10)
        # # ax2.plot(tp_sim,hddot_ff[index],'o',color='orange',markersize=10)
        # ax2.set_title("$\ddot{h}$ vs time | "+folder2[15:-13],size='xx-large')
        # ax2.set_xlim(0)
        # ax2.set_ylabel("$\ddot{h}$ (m/s$^2$)",size='x-large')
        # ax2.set_xlabel("Time (sec)",size='x-large')
        # ax2.legend(fontsize=16)
        # fig2.set_size_inches(10,5)
        # plt.tick_params(labelsize=14)
        # ax2.grid()
        
        # fig2.savefig("all_sims_figures/" + folder2[15:-13]+ "hddot.png");

        
        #print(time_target)

        tot_error = []
        
        for i in range(len(hddot_app)):
            
            tot_error.append((hddot_app[i] - rddotLAMB_w_LD[i]) / rddotLAMB_w_LD[i])
            
        x= np.subtract(t_simulation,delta_t_lamb/2)
        y = tot_error
        
        
        from scipy import optimize
        
        def test_func(x, a, b, c):
            return a * np.tanh(c* x) + b
        
        params, params_covariance = optimize.curve_fit(test_func, x, y, p0=[0.002, 0.02,0.001])
        
        if params[0]<0:
            params[0]=-params[0]
            
        if params[2]>0:
            params[2]=-params[2]
            

        error_app =  test_func(x, params[0], params[1], params[2])
        
        hope_app = np.add(np.multiply(error_app, rddotLAMB_w_LD), rddotLAMB_w_LD)
        
        a_element = params[0]
        b_element = params[1]
        c_element = params[2]
        
        # print(folder2[15:-13])
        # print(params)
        
        # fig1, ax1=plt.subplots()
        
        # ax1.plot(t_simulation,tot_error,color='green', label="$error$")
        # ax1.set_title("Total error vs time | "+folder2[15:-13],size='xx-large')
        # ax1.set_xlim(0)
        # ax1.set_ylabel("$\ddot{h}$ (m/s$^2$)",size='x-large')
        # ax1.set_xlabel("Time (sec)",size='x-large')
        # ax1.legend(fontsize=16)
        # fig1.set_size_inches(10,5)
        # plt.tick_params(labelsize=14)
        # ax1.grid()

        # fig1.savefig("all_sims_figures/" + folder2[15:-13]+ "error.png");

        #should probably pass into y^2 one
        vel_km6 = VEL2(t_simulation,tp_sim,alt_cfs,v0,y0,h0,aoa,t_simulation[-1], hope_app)[0]
        
                
        # calculate error
        # err = (vel_km6[-1] - v_sim[-1])/(v_sim[-1])*100
        err = (vel_km6[-1] - v_sim[-1])
        # err = (hope_app[-1] - hddot_app[-1])/(hddot_app[-1])*100

        # fig3, ax3=plt.subplots()
        # ax3.plot(t_simulation,v_sim,label="Simulation")
        # # ax3.plot(t_simulation,vel_km5,label="Closed-Form Solution")
        # ax3.plot(t_simulation,vel_km6,label="Closed-Form Solution 6", color='red')

        # ax3.plot(t_lamb, v_lambert, color='grey', label="$v_{Lambert}$")

        # ax3.set_title("Velocity "+folder2[15:-13],size='x-large')
        # ax3.set_xlim(0)
        # ax3.set_ylabel("Velocity (m/s)",size='x-large')
        # ax3.set_xlabel("Time (sec)",size='x-large')
        # fig3.set_size_inches(10,5)
        # plt.tick_params(labelsize=16)
        # plt.legend(fontsize=16)
        # ax3.grid()
        
        # fig3.savefig("all_sims_figures/" + folder2[15:-13]+ "velocity.png");

        
        # fig2, ax2=plt.subplots()
        # # # ax2.plot(t_simulation,hddot,color='black', label="$\ddot{h}_{small-angle}$")
        # # ax2.plot(t_simulation,hddot_sim,color='blue', label="$\ddot{h}$")
        # # ax2.plot(t_simulation,hddot_ff,color='red', label="$\ddot{h}_{full-form}$")
        # ax2.plot(t_simulation,hddot_app,color='green', label="$\ddot{h}_{approx}$")
        
        # ax2.plot(t_simulation,hope_app,color='red', label="Approximation??")
        # # ax2.plot(pt3[0],pt3[1], 'o',color='orange',markersize=10)
        
        # # ax2.plot(t_simulation, rddotOM, color='cyan', label="$\ddot{r}_{OM}$")
        # # ax2.plot(t_simulation, rddotRE, color='orange', label="$\ddot{r}_{RE}$")
        # # ax2.plot(t_lamb, rddotLAMB, color='grey', label="$\ddot{r}_{Lambert}$")
        # ax2.plot(t_lamb, rddotLAMB_w_LD, color='blue', label="$\ddot{r}_{Lambert}$ with LD")
        # # ax2.plot(pt1[0],pt1[1], 'o',color='orange',markersize=10)
        # # ax2.plot(pt2[0],pt2[1], 'o',color='orange',markersize=10)
        # # ax2.plot(pt3[0],pt3[1], 'o',color='orange',markersize=10)
        # # ax2.plot(tp_sim,hddot_ff[index],'o',color='orange',markersize=10)
        # ax2.set_title("$\ddot{h}$ vs time | "+folder2[15:-13],size='xx-large')
        # ax2.set_xlim(0)
        # ax2.set_ylabel("$\ddot{h}$ (m/s$^2$)",size='x-large')
        # ax2.set_xlabel("Time (sec)",size='x-large')
        # ax2.legend(fontsize=16)
        # fig2.set_size_inches(10,5)
        # plt.tick_params(labelsize=14)
        # ax2.grid()

        final_diff.append(err)
        
        a_array.append(a_element)
        b_array.append(b_element)
        c_array.append(c_element)
        hddot0_array.append(hddot_app[0])
        ICs_array.append([h_sim[0],v_sim[0],y_sim[0],rho_sim[0]])
        tp_lamb_array.append(tp_lamb)
        tf_sim_array.append(t_simulation[-1])

        # max_D_array.append((np.max(v_lambert)**2)*np.max(rho_exp(h_lambert))/2*Sref*CD_0*aoa)
        # max_D_array.append((np.max(v_lambert)**2)*(rho_exp(h_lambert)[0])/2*Sref*CD_0*aoa)

        max_D_array.append(drag_sim[0])

        drag_int = integrate.simps(drag_sim, x = t_simulation)
        drag_integral.append(drag_int)

        vp_array.append(vp)

        drag_int_no_v = integrate.simps(np.divide(drag_sim,v_sim**2), x = t_simulation)
        drag_integral_no_v.append(drag_int_no_v)

        drag_int_ver = integrate.simps(np.multiply(drag_sim,np.sin(y_sim)), x = t_simulation)
        drag_integral_ver.append(drag_int_ver)

        drag_int_hor = integrate.simps(np.multiply(drag_sim,np.cos(y_sim)), x = t_simulation)
        drag_integral_hor.append(drag_int_hor)
            
        energy_ra.append(energy_sim[0] - energy_sim[-1])

        rp_list.append(float(rp_file))

    def sorting_fun(x):
        return x[24:] + x[0:24]

    final_diff = sorting_fun(final_diff)
    a_array = sorting_fun(a_array)
    b_array = sorting_fun(b_array)
    c_array = sorting_fun(c_array)
    hddot0_array = sorting_fun(hddot0_array)
    ICs_array = sorting_fun(ICs_array)
    tp_lamb_array = sorting_fun(tp_lamb_array)
    tf_sim_array = sorting_fun(tf_sim_array)
    max_D_array = sorting_fun(max_D_array)
    drag_integral_no_v = sorting_fun(drag_integral_no_v)
    drag_integral_ver = sorting_fun(drag_integral_ver)
    drag_integral_hor = sorting_fun(drag_integral_hor)
    drag_integral = sorting_fun(drag_integral)
    vp_array = sorting_fun(vp_array)
    energy_ra = sorting_fun(energy_ra)

    diff_tot.append([final_diff])
    a_tot.append([a_array])
    b_tot.append([b_array])
    c_tot.append([c_array])
    hddot0_tot.append([hddot0_array])
    ICs_tot.append([ICs_array])
    tp_lamb_tot.append([tp_lamb_array])
    tf_sim_tot.append([tf_sim_array]) 
    max_D_tot.append([max_D_array]) 
    drag_integral_no_v_tot.append([drag_integral_no_v])
    drag_integral_ver_tot.append([drag_integral_ver])
    drag_integral_hor_tot.append([drag_integral_hor])
    drag_integral_tot.append([drag_integral])
    vp_tot.append([vp_array])
    energy_array.append([energy_ra])

    ra_list.append(float(ra_file))

    print(ra_list)

# r90 = []
# r95 = []
# r100 = []
# r105 = []
# r110 = []

# there are 34 rps

def rp_sorting(x):
    rp_values = []
    for i in range(len(ra_list)):
        for jj in range(len(rp)):
            rp_values.append(x[i][jj])

    for i in range(len(ra_list)):
        rp_values[i] = rp_values[i][31:] + rp_values[i][0:31]

    return rp_values


# for i in range(len(ra_list)):
#     r90.append(diff_tot[i][0])
#     r95.append(diff_tot[i][5])
#     r100.append(diff_tot[i][10])
#     r105.append(diff_tot[i][15])
#     r110.append(diff_tot[i][20])
    

# r90 = r90[6:] + r90[0:6]
# r95 = r95[6:] + r95[0:6]
# r100 = r100[6:] + r100[0:6]
# r105 = r105[6:] + r105[0:6]
# r110 = r110[6:] + r110[0:6]

# # fig20, ax20=plt.subplots()
# # ax20.plot(ra,r90,color='black', linestyle='-', marker='.', markersize=12,  label="$h_{p} = 90$")
# # ax20.plot(ra,r95,color='purple', linestyle='-', marker='.', markersize=12,  label="$h_{p} = 95$")
# # ax20.plot(ra,r100,color='blue', linestyle='-', marker='.', markersize=12,  label="$h_{p}$ = 100")
# # ax20.plot(ra,r105,color='red', linestyle='-', marker='.', markersize=12,  label="$h_{p}$ = 105")
# # ax20.plot(ra,r110,color='green', linestyle='-', marker='.', markersize=12,  label="$h_{p}$ = 110")
# # ax20.set_xlabel("$r_{a}$ (km)",size='x-large')
# # fig20.set_size_inches(10,12)
# # plt.tick_params(labelsize=14)
# # plt.legend(fontsize=16)
# # ax20.grid()


# labels  = np.array(range(5000,41000,1000))

# rp_90 = r90
# rp_95 = r95
# rp_100 = r100
# rp_105 = r105
# rp_110 =  r110
# x = np.arange(len(labels))  # the label locations
# width = 0.15


# fig5, ax5 = plt.subplots(figsize=(14, 8))

# rects1 = ax5.bar(x - 2*width, (rp_90), width, label='$h_p$=90km')
# rects2 = ax5.bar(x - 1*width, (rp_95), width, label='$h_p$=95km')
# rects3 = ax5.bar(x - 0*width, (rp_100), width, label='$h_p$=100km')
# rects4 = ax5.bar(x + 1*width, (rp_105), width, label='$h_p$=105km')
# rects5 = ax5.bar(x + 2*width, (rp_110), width, label='$h_p$=110km')

# # #Plot a line
# # ax5.plot( [-0.6,10.6], -0.6*np.ones(2), linestyle='--' , color="black")

# #print(plt.style.available)
# # ax5.set_ylabel('Final $\ddot{h}$ Error, %', fontsize=18)
# ax5.set_ylabel('Final $v$ Error, m/s', fontsize=18)
# ax5.set_xlabel('Apoapsis (km)', fontsize=18)
# ax5.tick_params(labelsize=16)
# ax5.set_xticks(x)
# ax5.set_xticklabels(labels,fontsize=16)
# ax5.legend(fontsize=18)
# fig5.tight_layout()
# plt.grid()


# #new set of data to analyze

# r90 = []
# r95 = []
# r100 = []
# r105 = []
# r110 = []

# for i in range(len(ra)):
#     r90.append(a_tot[i][0])
#     r95.append(a_tot[i][1])
#     r100.append(a_tot[i][2])
#     r105.append(a_tot[i][3])
#     r110.append(a_tot[i][4])
    

# r90 = r90[9:12] + r90[0:9]
# r95 = r95[9:12] + r95[0:9]
# r100 = r100[9:12] + r100[0:9]
# r105 = r105[9:12] + r105[0:9]
# r110 = r110[9:12] + r110[0:9]


# fig21, ax21=plt.subplots()
# ax21.plot(ra,r90,color='black', linestyle='-', marker='.', markersize=12,  label="$h_{p} = 90$")
# ax21.plot(ra,r95,color='purple', linestyle='-', marker='.', markersize=12,  label="$h_{p} = 95$")
# ax21.plot(ra,r100,color='blue', linestyle='-', marker='.', markersize=12,  label="$h_{p}$ = 100")
# ax21.plot(ra,r105,color='red', linestyle='-', marker='.', markersize=12,  label="$h_{p}$ = 105")
# ax21.plot(ra,r110,color='green', linestyle='-', marker='.', markersize=12,  label="$h_{p}$ = 110")
# ax21.set_ylabel("$a$",size='x-large')
# ax21.set_xlabel("$r_{a}$ (km)",size='x-large')
# fig21.set_size_inches(8,10)
# plt.tick_params(labelsize=14)
# plt.legend(fontsize=16)
# ax21.grid()


# r90 = []
# r95 = []
# r100 = []
# r105 = []
# r110 = []
    

# for i in range(len(ra)):
#     r90.append(b_tot[i][0])
#     r95.append(b_tot[i][1])
#     r100.append(b_tot[i][2])
#     r105.append(b_tot[i][3])
#     r110.append(b_tot[i][4])
    

# r90 = r90[9:12] + r90[0:9]
# r95 = r95[9:12] + r95[0:9]
# r100 = r100[9:12] + r100[0:9]
# r105 = r105[9:12] + r105[0:9]
# r110 = r110[9:12] + r110[0:9]


# fig22, ax22=plt.subplots()
# ax22.plot(ra,r90,color='black', linestyle='-', marker='.', markersize=12,  label="$h_{p} = 90$")
# ax22.plot(ra,r95,color='purple', linestyle='-', marker='.', markersize=12,  label="$h_{p} = 95$")
# ax22.plot(ra,r100,color='blue', linestyle='-', marker='.', markersize=12,  label="$h_{p}$ = 100")
# ax22.plot(ra,r105,color='red', linestyle='-', marker='.', markersize=12,  label="$h_{p}$ = 105")
# ax22.plot(ra,r110,color='green', linestyle='-', marker='.', markersize=12,  label="$h_{p}$ = 110")
# ax22.set_ylabel("$b$",size='x-large')
# ax22.set_xlabel("$r_{a}$ (km)",size='x-large')
# fig22.set_size_inches(8,10)
# plt.tick_params(labelsize=14)
# plt.legend(fontsize=16)
# ax22.grid()

# r90 = []
# r95 = []
# r100 = []
# r105 = []
# r110 = []
    

# for i in range(len(ra)):
#     r90.append(c_tot[i][0])
#     r95.append(c_tot[i][1])
#     r100.append(c_tot[i][2])
#     r105.append(c_tot[i][3])
#     r110.append(c_tot[i][4])
    

# r90 = r90[9:12] + r90[0:9]
# r95 = r95[9:12] + r95[0:9]
# r100 = r100[9:12] + r100[0:9]
# r105 = r105[9:12] + r105[0:9]
# r110 = r110[9:12] + r110[0:9]


# fig23, ax23=plt.subplots()
# ax23.plot(ra,r90,color='black', linestyle='-', marker='.', markersize=12,  label="$h_{p} = 90$")
# ax23.plot(ra,r95,color='purple', linestyle='-', marker='.', markersize=12,  label="$h_{p} = 95$")
# ax23.plot(ra,r100,color='blue', linestyle='-', marker='.', markersize=12,  label="$h_{p}$ = 100")
# ax23.plot(ra,r105,color='red', linestyle='-', marker='.', markersize=12,  label="$h_{p}$ = 105")
# ax23.plot(ra,r110,color='green', linestyle='-', marker='.', markersize=12,  label="$h_{p}$ = 110")
# ax23.set_ylabel("$c$",size='x-large')
# ax23.set_xlabel("$r_{a}$ (km)",size='x-large')
# fig23.set_size_inches(8,10)
# plt.tick_params(labelsize=14)
# plt.legend(fontsize=16)
# ax23.grid()

# import csv

# print(tf_sim_tot)

# header = ['Simulation','hp', 'a','b','c', 'h0', 'v0', 'y0', 'rho0' ,'hddot0','tp_lamb',"tf_sim","Initial drag", "Integral drag", "Periapsis velocity", "Integral drag no v", "Integral hor drag", "Integral ver drag"]
header = ['Simulation','hp', 'a','b','c', 'h0', 'v0', 'y0', 'rho0' ,'hddot0','tp_lamb',"tf_sim","Initial drag", "Integral drag", "Integral hor drag", "Integral ver drag", "Integral drag no v", "Periapsis velocity", "Energy Diff"]

print(np.size(hddot0_tot))

data = np.empty(((36*33)+1,len(header)))

ct = 0

# a_tot = rp_sorting(a_tot)
# b_tot = rp_sorting(b_tot)
# c_tot = rp_sorting(c_tot)

# ICs_tot = rp_sorting(ICs_tot)

# hddot0_tot = rp_sorting(hddot0_tot)
# tp_lamb_tot = rp_sorting(tp_lamb_tot)
# tf_sim_tot = rp_sorting(tf_sim_tot)

# max_D_tot = rp_sorting(max_D_tot)
# drag_integral_tot = rp_sorting(drag_integral_tot)
# drag_integral_hor_tot = rp_sorting(drag_integral_hor_tot)

# drag_integral_ver_tot = rp_sorting(drag_integral_ver_tot)
# drag_integral_no_v_tot = rp_sorting(drag_integral_no_v_tot)
# vp_tot = rp_sorting(vp_tot)
# energy_array = rp_sorting(energy_array)

print(a_tot)


for i in range(len(ra_list)):
    for j in range(len(rp)):
        data[ct][0] = ra_list[i]
        data[ct][1] = rp[j]
        data[ct][2] = a_tot[i][j]
        data[ct][3] = b_tot[i][j]
        data[ct][4] = c_tot[i][j]
        data[ct][5] = ICs_tot[i][j][0]
        data[ct][6] = ICs_tot[i][j][1]
        data[ct][7] = ICs_tot[i][j][2]
        data[ct][8] = ICs_tot[i][j][3]
        data[ct][9] = hddot0_tot[i][j]
        data[ct][10] = tp_lamb_tot[i][j]
        data[ct][11] = tf_sim_tot[i][j]
        data[ct][12] = max_D_tot[i][j]
        data[ct][13] = drag_integral_tot[i][j]
        data[ct][14] = drag_integral_hor_tot[i][j]
        data[ct][15] = drag_integral_ver_tot[i][j]
        data[ct][16] = drag_integral_no_v_tot[i][j]
        data[ct][17] = vp_tot[i][j]
        data[ct][18] = energy_array[i][j]
        
        ct = ct+1

with open('data/data_all_sims.csv', 'w', encoding='UTF8', newline='') as f:
    writer = csv.writer(f)

    # write the header
    writer.writerow(header)

    # write the data
    writer.writerows(data)


# # r90 = []
# # r95 = []
# # r100 = []
# # r105 = []
# # r110 = []
    

# # for i in range(len(ra)):
# #     r90.append(ICs_tot[i][0][1])
# #     r95.append(ICs_tot[i][1][1])
# #     r100.append(ICs_tot[i][2][1])
# #     r105.append(ICs_tot[i][3][1])
# #     r110.append(ICs_tot[i][4][1])
    

# # r90 = r90[9:12] + r90[0:9]
# # r95 = r95[9:12] + r95[0:9]
# # r100 = r100[9:12] + r100[0:9]
# # r105 = r105[9:12] + r105[0:9]
# # r110 = r110[9:12] + r110[0:9]


# # fig20, ax20=plt.subplots()
# # ax20.plot(ra,r90,color='black', linestyle='-', marker='.', markersize=12,  label="$r_{p} = 90$")
# # ax20.plot(ra,r95,color='purple', linestyle='-', marker='.', markersize=12,  label="$r_{p} = 95$")
# # ax20.plot(ra,r100,color='blue', linestyle='-', marker='.', markersize=12,  label="$r_{p}$ = 100")
# # ax20.plot(ra,r105,color='red', linestyle='-', marker='.', markersize=12,  label="$r_{p}$ = 105")
# # ax20.plot(ra,r110,color='green', linestyle='-', marker='.', markersize=12,  label="$r_{p}$ = 110")
# # ax20.set_xlabel("$r_{a}$ (km)",size='x-large')
# # fig20.set_size_inches(10,12)
# # plt.tick_params(labelsize=14)
# # plt.legend(fontsize=16)
# # ax20.grid()




# # r90 = []
# # r95 = []
# # r100 = []
# # r105 = []
# # r110 = []
    

# # for i in range(len(ra)):
# #     r90.append(ICs_tot[i][0][2])
# #     r95.append(ICs_tot[i][1][2])
# #     r100.append(ICs_tot[i][2][2])
# #     r105.append(ICs_tot[i][3][2])
# #     r110.append(ICs_tot[i][4][2])
    

# # r90 = r90[9:12] + r90[0:9]
# # r95 = r95[9:12] + r95[0:9]
# # r100 = r100[9:12] + r100[0:9]
# # r105 = r105[9:12] + r105[0:9]
# # r110 = r110[9:12] + r110[0:9]


# # fig20, ax20=plt.subplots()
# # ax20.plot(ra,r90,color='black', linestyle='-', marker='.', markersize=12,  label="$r_{p} = 90$")
# # ax20.plot(ra,r95,color='purple', linestyle='-', marker='.', markersize=12,  label="$r_{p} = 95$")
# # ax20.plot(ra,r100,color='blue', linestyle='-', marker='.', markersize=12,  label="$r_{p}$ = 100")
# # ax20.plot(ra,r105,color='red', linestyle='-', marker='.', markersize=12,  label="$r_{p}$ = 105")
# # ax20.plot(ra,r110,color='green', linestyle='-', marker='.', markersize=12,  label="$r_{p}$ = 110")
# # ax20.set_xlabel("$r_{a}$ (km)",size='x-large')
# # fig20.set_size_inches(10,12)
# # plt.tick_params(labelsize=14)
# # plt.legend(fontsize=16)
# # ax20.grid()





# # r90 = []
# # r95 = []
# # r100 = []
# # r105 = []
# # r110 = []
    

# # for i in range(len(ra)):
# #     r90.append(tf_lamb_tot[i][0])
# #     r95.append(tf_lamb_tot[i][1])
# #     r100.append(tf_lamb_tot[i][2])
# #     r105.append(tf_lamb_tot[i][3])
# #     r110.append(tf_lamb_tot[i][4])
    

# # r90 = r90[9:12] + r90[0:9]
# # r95 = r95[9:12] + r95[0:9]
# # r100 = r100[9:12] + r100[0:9]
# # r105 = r105[9:12] + r105[0:9]
# # r110 = r110[9:12] + r110[0:9]


# # fig20, ax20=plt.subplots()
# # ax20.plot(ra,r90,color='black', linestyle='-', marker='.', markersize=12,  label="$r_{p} = 90$")
# # ax20.plot(ra,r95,color='purple', linestyle='-', marker='.', markersize=12,  label="$r_{p} = 95$")
# # ax20.plot(ra,r100,color='blue', linestyle='-', marker='.', markersize=12,  label="$r_{p}$ = 100")
# # ax20.plot(ra,r105,color='red', linestyle='-', marker='.', markersize=12,  label="$r_{p}$ = 105")
# # ax20.plot(ra,r110,color='green', linestyle='-', marker='.', markersize=12,  label="$r_{p}$ = 110")
# # ax20.set_xlabel("$r_{a}$ (km)",size='x-large')
# # fig20.set_size_inches(10,12)
# # plt.tick_params(labelsize=14)
# # plt.legend(fontsize=16)
# # ax20.grid() 



# # cr90 = []
# # cr95 = []
# # cr100 = []
# # cr105 = []
# # cr110 = []

# # x90 = []
# # x95 = []
# # x100 = []
# # x105 = []
# # x110 = []

# # t90 = []
# # t95 = []
# # t100 = []
# # t105 = []
# # t110 = []
    
# # coeff_array_m = []

# # coeff_array_b = []

# # for i in range(len(ra)):
# #     cr90.append(a_tot[i][0])
# #     cr95.append(a_tot[i][1])
# #     cr100.append(a_tot[i][2])
# #     cr105.append(a_tot[i][3])
# #     cr110.append(a_tot[i][4])
    
# #     x90.append(max_D_tot[i][0])
# #     x95.append(max_D_tot[i][1])
# #     x100.append(max_D_tot[i][2])
# #     x105.append(max_D_tot[i][3])
# #     x110.append(max_D_tot[i][4])

# #     t90.append(tf_lamb_tot[i][0])
# #     t95.append(tf_lamb_tot[i][1])
# #     t100.append(tf_lamb_tot[i][2])
# #     t105.append(tf_lamb_tot[i][3])
# #     t110.append(tf_lamb_tot[i][4])
    
# #     coeff = np.polyfit( [max_D_tot[i][0],max_D_tot[i][1],max_D_tot[i][2],max_D_tot[i][3],max_D_tot[i][4] ], [a_tot[i][0],a_tot[i][1],a_tot[i][2],a_tot[i][3],a_tot[i][4]], 1)
# #     coeff_array_b.append(coeff[1])
# #     coeff_array_m.append(coeff[0])
    
    

# # cr90 = cr90[9:12] + cr90[0:9]
# # cr95 = cr95[9:12] + cr95[0:9]
# # cr100 = cr100[9:12] + cr100[0:9]
# # cr105 = cr105[9:12] + cr105[0:9]
# # cr110 = cr110[9:12] + cr110[0:9]

# # x90 = x90[9:12] + x90[0:9]
# # x95 = x95[9:12] + x95[0:9]
# # x100 = x100[9:12] + x100[0:9]
# # x105 = x105[9:12] + x105[0:9]
# # x110 = x110[9:12] + x110[0:9]

# # t90 = t90[9:12] + t90[0:9]
# # t95 = t95[9:12] + t95[0:9]
# # t100 = t100[9:12] + t100[0:9]
# # t105 = t105[9:12] + t105[0:9]
# # t110 = t110[9:12] + t110[0:9]

# # coeff_array_b = coeff_array_b[9:12] + coeff_array_b[0:9]
# # coeff_array_m = coeff_array_m[9:12] + coeff_array_m[0:9]

# # y = np.add((coeff_array[1][1]),np.multiply(coeff[0],[ x90[1],x95[1],x100[1],x105[1],x110[1] ]))

# # coeff2 =np.polyfit( [ x90[0],x95[0],x100[0],x105[0],x110[0] ], [ cr90[0],cr95[0],cr100[0],cr105[0],cr110[0] ], 1)

# # y2 = np.add((coeff_array[0][1]),np.multiply(coeff_array[0][0],[ x90[0],x95[0],x100[0],x105[0],x110[0] ]))

# # from scipy import optimize
        
# # def test_func(x, a, c):
# #     return a * np.exp(x) +c

# # params, params_covariance = optimize.curve_fit(test_func, np.array(x90), np.array(cr90))   

# # y90 =  test_func(np.array(x90), params[0], params[1])


# # X = [x90,x95,x100,x105,x110]
# # Y = [cr90,cr95,cr100,cr105,cr110]
# # Z = [t90,t95,t100,t105,t110]

# # print(Z)

# # fig85, ax85 = plt.subplots()
# # CS = ax85.contourf(X,Y,Z,12, cmap = 'viridis')
# # CB = fig85.colorbar(CS)
# # # ax85.clabel(CS,inline=True)
# # ax85.set_ylabel("$a$",size='x-large')
# # ax85.set_xlabel("$D_{max}$ (N)",size='x-large')
# # # ax85.plot(x90,cr90,color='black', markersize=12, label="$h_{p} = 90$")
# # # ax85.plot(x95,cr95,color='purple', markersize=12, label="$h_{p} = 95$")
# # # ax85.plot(x100,cr100,color='blue', markersize=12, label="$h_{p}$ = 100")
# # # ax85.plot(x105,cr105,color='red', markersize=12, label="$h_{p}$ = 105")
# # # ax85.plot(x110,cr110,color='green', markersize=12, label="$h_{p}$ = 110")
# # ax85.scatter(x90,cr90,color='black', label="$h_{p} = 90$")
# # ax85.scatter(x95,cr95,color='purple', label="$h_{p} = 95$")
# # ax85.scatter(x100,cr100,color='blue', label="$h_{p} = 100$")
# # ax85.scatter(x105,cr105,color='red', label="$h_{p} = 105$")
# # ax85.scatter(x110,cr110,color='green', label="$h_{p} = 110$")
# # # plt.legend(fontsize=16)
# # fig85.set_size_inches(10,12)
# # plt.tick_params(labelsize=14)
# # ax85.grid()


# # # print(np.shape(X))

# # x = []
# # y = []
# # z = [] 

# # for i in X:
# #     for jj in i:
# #         x.append(jj)

# # for i in Y:
# #     for jj in i:
# #         y.append(jj)

# # for i in Z:
# #     for jj in i:
# #         z.append(jj)


# # fig84 = plt.figure()
# # ax84 = fig84.add_subplot(projection = '3d')
# # CS = ax84.plot_trisurf(x,y,z, cmap='binary', linewidths = 0.2)
# # CB = fig84.colorbar(CS)
# # ax84.scatter(x90,cr90,t90, color='black', label="$h_{p} = 90$")
# # ax84.scatter(x95,cr95,t95, color='purple', label="$h_{p} = 95$")
# # ax84.scatter(x100,cr100,t100, color='blue', label="$h_{p} = 100$")
# # ax84.scatter(x105,cr105,t105, color='red', label="$h_{p} = 105$")
# # ax84.scatter(x110,cr110, t110, color='green', label="$h_{p} = 110$")
# # ax84.set_ylabel("$a$",size='x-large')
# # ax84.set_xlabel("$D_{max}$ (N)",size='x-large')
# # plt.legend(fontsize=16)
# # fig84.set_size_inches(10,12)
# # plt.tick_params(labelsize=14)
# # ax84.grid()


# # #print(tp_lamb_tot)

# # # fig20, ax20=plt.subplots()
# # # ax20.plot(x90,cr90,color='black', linestyle='-', label="$h_{p} = 90$")
# # # # ax20.plot([ x90[0],x95[0],x100[0],x105[0],x110[0] ] ,y2,color='black', linestyle='--')
# # # # ax20.plot(x90,y90,color='green', linestyle='--')

# # # ax20.plot(x95,cr95,color='purple', linestyle='-',  label="$h_{p} = 95$")
# # # ax20.plot(x100,cr100,color='blue', linestyle='-',  label="$h_{p}$ = 100")
# # # ax20.plot(x105,cr105,color='red', linestyle='-',  label="$h_{p}$ = 105")
# # # ax20.plot(x110,cr110,color='green', linestyle='-',  label="$h_{p}$ = 110")
# # # ax20.set_ylabel("$a$",size='x-large')
# # # ax20.set_xlabel("$D_{max}$ (N)",size='x-large')
# # # fig20.set_size_inches(10,12)
# # # plt.tick_params(labelsize=14)
# # # plt.legend(fontsize=16)
# # # ax20.grid()

# # # ra = [5000, 7000, 9000, 11000, 13000 ,15000, 17000 , 19000 , 21000 , 23000 , 25000 , 27000]
# # # # 
# # # np.polyfit( [ cr90[0],cr90[1],cr90[2],cr90[3],cr90[4] ] , [ x90[0],x90[1],x90[2],x90[3],x90[4] ] , 1)

# # # #    y ≈ exp(-0.401) * exp(0.105 * x) = 0.670 * exp(0.105 * x)
# # # # (^ biased towards small values)


# # # #    y ≈ exp(1.42) * exp(0.0601 * x) = 4.12 * exp(0.0601 * x)
# # # # (^ not so biased)

# # # fig201, ax201=plt.subplots()
# # # ax201.plot(ra,coeff_array_m,color='blue', linestyle='-', marker='.', markersize=12,  label="m")
# # # ax201.plot(ra,coeff_array_b,color='red', linestyle='-', marker='.', markersize=12,  label="b")
# # # fig201.set_size_inches(10,12)
# # # plt.tick_params(labelsize=14)
# # # plt.legend(fontsize=16)
# # # ax201.grid()

# # # ar90 = []
# # # ar95 = []
# # # ar100 = []
# # # ar105 = []
# # # ar110 = []

# # # cr90 = []
# # # cr95 = []
# # # cr100 = []
# # # cr105 = []
# # # cr110 = []

# # # x90 = []
# # # x95 = []
# # # x100 = []
# # # x105 = []
# # # x110 = []
    
# # # coeff_array_m = []

# # # coeff_array_b = []

# # # for i in range(len(ra)):
    
# # #     ar90.append(a_tot[i][0])
# # #     ar95.append(a_tot[i][1])
# # #     ar100.append(a_tot[i][2])
# # #     ar105.append(a_tot[i][3])
# # #     ar110.append(a_tot[i][4])
    
# # #     cr90.append(c_tot[i][0])
# # #     cr95.append(c_tot[i][1])
# # #     cr100.append(c_tot[i][2])
# # #     cr105.append(c_tot[i][3])
# # #     cr110.append(c_tot[i][4])
    
# # #     # x90.append(tp_lamb_tot[i][0])
# # #     # x95.append(tp_lamb_tot[i][1])
# # #     # x100.append(tp_lamb_tot[i][2])
# # #     # x105.append(tp_lamb_tot[i][3])
# # #     # x110.append(tp_lamb_tot[i][4])
    
# # #     x90.append(ICs_tot[i][0][1])
# # #     x95.append(ICs_tot[i][1][1])
# # #     x100.append(ICs_tot[i][2][1])
# # #     x105.append(ICs_tot[i][3][1])
# # #     x110.append(ICs_tot[i][4][1])
    
# # #     # coeff = np.polyfit( [max_D_tot[i][0],max_D_tot[i][1],max_D_tot[i][2],max_D_tot[i][3],max_D_tot[i][4] ], [a_tot[i][0],a_tot[i][1],a_tot[i][2],a_tot[i][3],a_tot[i][4]], 1)
# # #     # coeff_array_b.append(coeff[1])
# # #     # coeff_array_m.append(coeff[0])
    
    

# # # ar90 = ar90[9:12] + ar90[0:9]
# # # ar95 = ar95[9:12] + ar95[0:9]
# # # ar100 = ar100[9:12] + ar100[0:9]
# # # ar105 = ar105[9:12] + ar105[0:9]
# # # ar110 = ar110[9:12] + ar110[0:9]

# # # cr90 = cr90[9:12] + cr90[0:9]
# # # cr95 = cr95[9:12] + cr95[0:9]
# # # cr100 = cr100[9:12] + cr100[0:9]
# # # cr105 = cr105[9:12] + cr105[0:9]
# # # cr110 = cr110[9:12] + cr110[0:9]

# # # x90 = x90[9:12] + x90[0:9]
# # # x95 = x95[9:12] + x95[0:9]
# # # x100 = x100[9:12] + x100[0:9]
# # # x105 = x105[9:12] + x105[0:9]
# # # x110 = x110[9:12] + x110[0:9]

# # # # coeff_array_b = coeff_array_b[9:12] + coeff_array_b[0:9]
# # # # coeff_array_m = coeff_array_m[9:12] + coeff_array_m[0:9]

# # # # y = np.add((coeff_array[1][1]),np.multiply(coeff[0],[ x90[1],x95[1],x100[1],x105[1],x110[1] ]))

# # # # coeff2 =np.polyfit( [ x90[0],x95[0],x100[0],x105[0],x110[0] ], [ cr90[0],cr95[0],cr100[0],cr105[0],cr110[0] ], 1)

# # # # y2 = np.add((coeff_array[0][1]),np.multiply(coeff_array[0][0],[ x90[0],x95[0],x100[0],x105[0],x110[0] ]))


# # # fig20, ax20=plt.subplots()
# # # ax20.plot(x90,cr90,color='black', linestyle='-', marker='.', markersize=12,  label="$r_{p} = 90$")
# # # ax20.plot(x95,cr95,color='purple', linestyle='-', marker='.', markersize=12,  label="$r_{p} = 95$")
# # # ax20.plot(x100,cr100,color='blue', linestyle='-', marker='.', markersize=12,  label="$r_{p}$ = 100")
# # # ax20.plot(x105,cr105,color='red', linestyle='-', marker='.', markersize=12,  label="$r_{p}$ = 105")
# # # ax20.plot(x110,cr110,color='green', linestyle='-', marker='.', markersize=12,  label="$r_{p}$ = 110")
# # # ax20.set_ylabel("$c$",size='x-large')
# # # ax20.set_xlabel("$v_{0}$",size='x-large')
# # # fig20.set_size_inches(10,12)
# # # plt.tick_params(labelsize=14)
# # # plt.legend(fontsize=16)
# # # ax20.grid()


# # # fig21, ax21=plt.subplots()
# # # ax21.plot(ar90,cr90,color='black', linestyle='-', marker='.', markersize=12,  label="$r_{p} = 90$")
# # # ax21.plot(ar95,cr95,color='purple', linestyle='-', marker='.', markersize=12,  label="$r_{p} = 95$")
# # # ax21.plot(ar100,cr100,color='blue', linestyle='-', marker='.', markersize=12,  label="$r_{p}$ = 100")
# # # ax21.plot(ar105,cr105,color='red', linestyle='-', marker='.', markersize=12,  label="$r_{p}$ = 105")
# # # ax21.plot(ar110,cr110,color='green', linestyle='-', marker='.', markersize=12,  label="$r_{p}$ = 110")
# # # ax21.set_ylabel("$c$",size='x-large')
# # # ax21.set_xlabel("$a$",size='x-large')
# # # fig21.set_size_inches(10,12)
# # # plt.tick_params(labelsize=14)
# # # plt.legend(fontsize=16)
# # # ax21.grid()


# # plt.show()