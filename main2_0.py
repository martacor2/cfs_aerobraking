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

diff_array = []
diff_array_2 = []

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
        csv_file=pd.read_csv(os.path.join(directory,file))

    files = data_gathering(csv_file)
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
# not even simplified D and L
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
        
        # rddot = -g + ((rho/rho_0)*v[i]**2)/(Beta*Rp)*(LD*np.cos(y[i]) - np.sin(y[i])) \
        # + v[i]**2/(h[i]+Rp)*(np.cos(y[i]))**2
        
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

hddot_array=[]
time_array=[]
c_sim_array=[]
amplitude_ra=[]
start_ra=[]

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

# SAM_lambert = np.multiply((Rp + h_lambert[0])*v_lambert[0]*np.cos(y_lambert[0]),np.ones(np.size(t_lamb)))

        
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

rddotOM = rddot_OM(h_lambert,SAM_lambert)
rddotRE = rddot_RE(h_lambert,v_lambert,y_lambert)
rddotLAMB = rddot_OM(h_lambert,SAM_lambert)

rddotLAMB_w_LD = rddot_RE(h_lambert,v_lambert,y_lambert)


c_lamb = np.multiply(-v0*y0/tp_lamb,np.ones(np.size(t_lamb)))
alt_cfs=ALT(t_simulation,tp_sim,v0,y0,h0)
vel_km5 = VEL2(t_simulation,tp_sim,alt_cfs,v0,y0,h0,aoa,t_simulation[-1], hddot_app)[0]
# vel_cfs=np.add(vel_km5,v0-vel_km5[0])
y_cfs=FPA(t_simulation,tp_sim,v0,y0,h0,vel_km5)

# diff_array = np.subtract(rddotLAMB,hddot_ff)


diff_array2 = np.subtract(np.subtract(rddotLAMB_w_LD,rddotLAMB_w_LD[0]),np.subtract(hddot_app,hddot_app[0]))

diff_array2 = np.subtract(rddotLAMB_w_LD,hddot_app)

t_diff = np.divide(t_lamb,delta_t_lamb)


target2 = np.min(np.abs(diff_array2[50:]))

index_to_app = np.argmin(np.abs(diff_array2[50:]))

time_target2 = t_diff[50:][index_to_app]

# print(time_target)
print(time_target2)


# fig2, ax2=plt.subplots()
# # # ax2.plot(t_simulation,hddot,color='black', label="$\ddot{h}_{small-angle}$")
# # ax2.plot(t_simulation,hddot_sim,color='blue', label="$\ddot{h}$")
# # ax2.plot(t_diff,np.abs(diff_array),color='red')
# # ax2.plot(time_target,target,'o',color='orange',markersize=10)
# ax2.plot(t_diff,diff_array2,color='blue')
# ax2.plot(time_target2,target2,'o',color='green',markersize=10)
# # ax2.plot(t_simulation, rddotOM, color='green', label="$\ddot{r}_{OM}$")
# # ax2.plot(t_simulation, rddotRE, color='orange', label="$\ddot{r}_{RE}$")
# # ax2.plot(t_lamb, rddotLAMB_w_LD, color='pink', label="$\ddot{r}_{Lambert}$ with LD")
# ax2.set_title("$\ddot{h}$ vs time | "+folder2[15:-13],size='xx-large')
# ax2.set_ylabel("$\ddot{h}$ (m/s)",size='x-large')
# ax2.set_xlabel("Time (sec)",size='x-large')

# ax2.set_ylim(-0.0025,0.0025)

# fig2.set_size_inches(10,5)
# plt.tick_params(labelsize=14)
# ax2.grid()


# fig2, ax2=plt.subplots()
# # # ax2.plot(t_simulation,hddot,color='black', label="$\ddot{h}_{small-angle}$")
# # ax2.plot(t_simulation,hddot_sim,color='blue', label="$\ddot{h}$")
# # ax2.plot(t_simulation, np.subtract(hddot_ff,hddot_ff[0]),color='red', label="$\ddot{h}_{full-form}$")
# ax2.plot(t_diff, np.subtract(hddot_app,hddot_app[0]),color='green', label="$\ddot{h}_{approx}$")
# # ax2.plot(t_simulation, rddotOM, color='green', label="$\ddot{r}_{OM}$")
# # ax2.plot(t_simulation, rddotRE, color='orange', label="$\ddot{r}_{RE}$")
# # ax2.plot(t_lamb, np.subtract(rddotLAMB,rddotLAMB[0]), color='grey', label="$\ddot{r}_{Lambert}$")
# ax2.plot(t_diff, np.subtract(rddotRE,rddotRE[0]), color='orange', label="$\ddot{r}_{RE}$")
# # ax2.plot(t_lamb, rddotLAMB_w_LD, color='pink', label="$\ddot{r}_{Lambert}$ with LD")

# ax2.set_title("$\ddot{h}$ vs time | "+folder2[15:-13],size='xx-large')
# ax2.set_xlim(0)
# ax2.set_ylabel("$\ddot{h}$ (m/s)",size='x-large')
# ax2.set_xlabel("Time (sec)",size='x-large')
# ax2.legend(fontsize=16)
# fig2.set_size_inches(10,5)
# plt.tick_params(labelsize=12)
# ax2.grid()


# h_ddot = np.add(np.multiply(v_rate_sim,np.sin(y_sim)),np.multiply(y_rate_sim,(np.multiply(v_sim,np.cos(y_sim)))))
# #assuming small angle
# h_ddot_small = sim_hddot(CL_0,CD_0,Sref,v_sim,rho_sim,Rp,h_sim,y_sim)


# vel_km2=np.abs(VEL(t_simulation,tp_sim,h_sim,v0,y0,h0,aoa,tf,h_ddot)[0])
# vel_new=np.add(vel_km2,v0-vel_km2[0])


fig4, (ax1, ax2, ax3) = plt.subplots(1, 3)

ax1.plot(t_lamb, h_lambert, color='grey', label="$h_{Lambert}$")
ax1.plot([delta_t_lamb,delta_t_lamb], [np.max(h_lambert), np.min(h_lambert)], "--")
ax1.plot(t_simulation, h_sim, color='red', label="$h_{Sim}$")
ax1.set_title(folder2[15:-13]+" | $h$",size='xx-large')
ax1.set_xlim(0)
ax1.set_xlabel("Time (sec)",size='x-large')
ax1.grid()

ax2.plot(t_lamb, v_lambert, color='grey', label="$h_{Lambert}$")
ax2.plot([delta_t_lamb,delta_t_lamb], [np.max(v_lambert), np.min(v_lambert)], "--")
ax2.plot(t_simulation, v_sim, color='red', label="$h_{Sim}$")
ax2.set_title("$v$",size='xx-large')
ax2.set_xlim(0)
ax2.set_xlabel("Time (sec)",size='x-large')
ax2.grid()

ax3.plot(t_lamb, y_lambert, color='grey', label="$h_{Lambert}$")
ax3.plot([delta_t_lamb,delta_t_lamb], [np.max(y_lambert), np.min(y_lambert)], "--")
ax3.plot(t_simulation, y_sim, color='red', label="$h_{Sim}$")
ax3.set_title("$\gamma$",size='xx-large')
ax3.set_xlim(0)
ax3.set_xlabel("Time (sec)",size='x-large')
ax3.grid()

fig4.set_size_inches(16,5)
# fig4.savefig("single_sim_plots/all_3.png");

def parabola_app(pt1,pt2,pt3,time):
    
    A = np.array([ [pt1[0]**2,pt1[0],1], [pt2[0]**2,pt2[0],1] ,[pt3[0]**2,pt3[0],1] ])
    B =  np.array( [ pt1[1],pt2[1],pt3[1] ] )
    
    a,b,c = np.linalg.solve(A,B)
    
    par_app=[]
    
    for i in range(len(time)):
        par_app.append(a*(time[i]**2) + b*time[i] +c)
        
    return par_app

def vdot_part1(y,v,h,rho):
    
    fin = mu/(Rp+h)**2 * y
    
    return fin

def y_2nd(y,v,h,rho):
    
    r = np.add(h,Rp)
    
    Beta = 2*m/(CD_0*aoa*rho0*Sref*Rp)
    
    LD = CL_0/(aoa*CD_0)
    
    den = 1 + (H/Rp)*((mu/r**2)*Rp/(v**2) - 1)*(1 - rho0/rho )
    
    num = np.cos(y0)+ 0.5*LD*H*rho/Beta * (1 - rho0/rho)
    
    fin = -rho*aoa*CD_0*Sref*(v**2)/(2*m)
    
    return fin
    
    # return (H/Rp)*(np.cos(y)/rho)*((mu/r**2)*Rp/(v**2) - 1)


pt1 = [t0,hddot_app[0]]
pt2 = [delta_t_lamb*time_target2, hddot_app[50:][index_to_app] ]
pt3 = [2*(delta_t_lamb*time_target2),hddot_app[0]]
# pt3 = [delta_t_lamb,hddot_app[-1]]

par_app = parabola_app(pt1,pt2,pt3,t_lamb)


print(np.argmax(drag_sim)) 
print(np.argmax(v_lambert))


# fig3, ax3=plt.subplots()
# ax3.plot(t_simulation,vdot_part1(y_sim,v_sim,h_sim,rho_sim),color='red', label="$y_{sim}$")
# ax3.plot(t_simulation, y_2nd(y_sim,v_sim,h_sim,rho_sim), color='black', label="$y_{sim}$")

# ax3.set_title("FPA",size='x-large')
# ax3.set_xlim(0)
# ax3.set_ylabel("FPA (rad)",size='x-large')
# ax3.set_xlabel("Time (sec)",size='x-large')
# fig3.set_size_inches(10,5)
# plt.tick_params(labelsize=16)
# plt.legend(fontsize=16)
# ax3.grid()

#approximation of hddot error
err = (par_app[-1] - hddot_app[-1])/(hddot_app[-1])*100

tot_error = []

for i in range(len(hddot_app)):
    
    tot_error.append((hddot_app[i] - rddotLAMB_w_LD[i]) / rddotLAMB_w_LD[i])
    
x= np.subtract(t_simulation,delta_t_lamb/2)
y = tot_error


from scipy import optimize

def test_func(x, a, b, c):
    return a * np.tanh(c* x) + b

params, params_covariance = optimize.curve_fit(test_func, x, y, p0=[-0.002, 0.02,0.001])
print(params)

error_app =  test_func(x, params[0], params[1], params[2])

fig1, ax1=plt.subplots()
ax1.plot(t_simulation,tot_error,color='green', label="$Error$")
# ax1.plot([delta_t_lamb,delta_t_lamb], [np.max(tot_error), np.min(tot_error)], "--")
ax1.plot(np.add(x,delta_t_lamb/2), error_app, color='red', label="$Approximation$")

ax1.set_title("Total error vs time | "+folder2[15:-13],size='xx-large')
ax1.set_xlim(0)
ax1.set_ylabel("$(\ddot{h}_{simplified} - \ddot{r}_{RE}) / \ddot{r}_{RE}$ (m/s$^2$)",size='x-large')
ax1.set_xlabel("Time (sec)",size='x-large')
ax1.legend(fontsize=16)
fig1.set_size_inches(10,5)
plt.tick_params(labelsize=14)
ax1.grid()
fig1.savefig("single_sim_plots/hddot_error.png");

hope_app = np.add(np.multiply(error_app, rddotLAMB_w_LD), rddotLAMB_w_LD)


fig2, ax2=plt.subplots()
# # ax2.plot(t_simulation,hddot,color='black', label="$\ddot{h}_{small-angle}$")
# ax2.plot(t_simulation,hddot_sim,color='blue', label="$\ddot{h}$")
ax2.plot(t_simulation,hddot_ff,color='black', label="$\ddot{h}$")
ax2.plot(t_simulation,hddot_app,color='green', label="$\ddot{h}_{simplified}$")
# ax2.plot([delta_t_lamb,delta_t_lamb], [np.max(hddot_app), np.min(hddot_app)], "--")

# ax2.plot(t_simulation,hope_app,color='black', label="tanh error approx")
# ax2.plot(t_simulation,par_app,color='red', label="parabola approx")
# ax2.plot(pt3[0],pt3[1], 'o',color='orange',markersize=10)

ax2.plot(t_simulation, rddotOM, color='cyan', label="$\ddot{r}_{OM}$")
# ax2.plot(t_lamb, rddotLAMB, color='grey', label="$\ddot{r}_{Lambert}$")
ax2.plot(t_lamb, rddotLAMB_w_LD, color='blue', label="$\ddot{r}_{RE}$")
# ax2.plot(pt1[0],pt1[1], 'o',color='orange',markersize=10)
# ax2.plot(pt2[0],pt2[1], 'o',color='orange',markersize=10)
# ax2.plot(pt3[0],pt3[1], 'o',color='orange',markersize=10)
# ax2.plot(tp_sim,hddot_ff[index],'o',color='orange',markersize=10)
ax2.set_title("$\ddot{h}$ vs time | "+folder2[15:-13],size='xx-large')
ax2.set_xlim(0)
ax2.set_ylabel("$\ddot{h}$ (m/s$^2$)",size='x-large')
ax2.set_xlabel("Time (sec)",size='x-large')
ax2.legend(fontsize=16)
fig2.set_size_inches(10,5)
plt.tick_params(labelsize=14)
ax2.grid()
fig2.savefig("single_sim_plots/hddot.png");

#should probably pass into y^2 one
vel_km6 = VEL2(t_simulation,tp_sim,alt_cfs,v0,y0,h0,aoa,t_simulation[-1], hope_app)[0]
#should probably pass into y^2 one
# vel_km6 = VEL2(t_simulation,tp_sim,alt_cfs,v0,y0,h0,aoa,t_simulation[-1], par_app)[0]


rho_exp_model = rho_exp(alt_cfs)

fig3, ax3=plt.subplots()
ax3.plot(t_simulation,v_sim,label="Simulation")
# ax3.plot(t_simulation,vel_km5,label="Closed-Form Solution")
ax3.plot(t_simulation,vel_km6,label="Closed-Form Solution 6", color='red')

ax3.plot(t_lamb, v_lambert, color='grey', label="$v_{Lambert}$")

ax3.set_title("Velocity "+folder2[15:-13],size='x-large')
ax3.set_xlim(0)
ax3.set_ylabel("Velocity (m/s)",size='x-large')
ax3.set_xlabel("Time (sec)",size='x-large')
fig3.set_size_inches(10,5)
plt.tick_params(labelsize=16)
plt.legend(fontsize=16)
ax3.grid()

fig3.savefig("single_sim_plots/velocity.png");



fig3, ax3=plt.subplots()
ax3.plot(t_simulation,rho_sim,label="Simulation")
ax3.plot(t_simulation, rho_exp_model, color='grey', label="$rho_{exp}$")

ax3.set_title("Density "+folder2[15:-13],size='x-large')
ax3.set_xlim(0)
ax3.set_ylabel("Density (kg/m$^3$)",size='x-large')
ax3.set_xlabel("Time (sec)",size='x-large')
fig3.set_size_inches(10,5)
plt.tick_params(labelsize=16)
plt.legend(fontsize=16)
ax3.grid()


fig3, ax3=plt.subplots()
ax3.plot(t_simulation,drag_sim,label="Simulation")
ax3.plot(t_simulation,drag_sim,label="Simulation")
ax3.set_title("Drag "+folder2[15:-13],size='x-large')
ax3.set_xlim(0)
ax3.set_ylabel("Drag (N)",size='x-large')
ax3.set_xlabel("Time (sec)",size='x-large')
fig3.set_size_inches(10,5)
plt.tick_params(labelsize=16)
plt.legend(fontsize=16)
ax3.grid()

Drag = (np.max(v_lambert)**2)*np.max(rho_exp(h_lambert))/2*Sref*CD_0*aoa
print(aoa)