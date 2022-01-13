# -*- coding: utf-8 -*-
"""
Created on Sat Mar 20 18:30:55 2021

@author: marta
"""

import numpy as np

#altitude
def ALT(t,tp,v0,y0,h0):
    altitude=h0+(v0*y0)*(t-(t**2/(2*tp)))
    return altitude

#flight path angle
def FPA(t,tp,v0,y0, h0,v):
    fp_angle = (v0*y0)/v*(1-t/tp)
    return fp_angle

#velocity
def VEL4(t,tp,alt,v0,y0,h0,aoa,tf,c):
    velocity = []
    Rp = 3396.2*1000 #km
    mu = 4.2828e13 # gravitational parameter, m^3/s^2
    m = 461 #kg
    rho_ref = 8.748923102971180e-07 #kg/m^3
    Sref=11 #m^2
    CL0=0.1
    CD0=1.477
    H =6300 #m
    
    first=[]
    second=[]
    third=[]
    fourth=[]
    
    for i in range(len(alt)):
        g = mu/(Rp+alt[i])**2
        g=3.71
        
        rho=rho_ref*np.exp(np.divide(90000-alt[i],H))

        k1 = ((rho*CL0*Sref)/(2*m))+1/(np.add(Rp,alt[i]))
        k2 = (-rho*CD0*aoa*Sref*v0*y0*(1-np.divide(t[i],tp)))/(2*m)
        k3 = -g*(v0**2*y0**2*((1-np.divide(t[i],tp)))**2)
        
        coeff=[k1,k2,(-g-c[i]),0,k3]
        v_alt=np.roots(coeff)
        
        velocity.append(v_alt)
#        rho_array.append(rho)
        
    for i in range(len(velocity)):
        need=velocity[i][0]
        first.append(need)
        need=velocity[i][1]
        second.append(need)
        need=velocity[i][2]
        third.append(need)
        need=velocity[i][3]
        fourth.append(need)
        
    #these are the 4 roots calculated for every time step
    return [first,second,third,fourth]


def VEL2(t,tp,alt,v0,y0,h0,aoa,tf,c):
    velocity = []
    Rp = 3396.2*1000 #km
    mu = 4.2828e13 # gravitational parameter, m^3/s^2
    m = 461 #kg
    rho_ref = 8.748923102971180e-07 #kg/m^3
    Sref=11 #m^2
    CL0=0.1
    CD0=1.477
    H =6300 #m
    
    
    for i in range(len(alt)):
        g = mu/(Rp+alt[i])**2
        
        rho=rho_ref*np.exp(np.divide(90000-alt[i],H))

        k1 = ((rho*CL0*Sref)/(2*m))+1/(np.add(Rp,alt[i]))
        k2 = (-rho*CD0*aoa*Sref*v0*y0*(1-np.divide(t[i],tp)))/(2*m)
        
        
        v_alt = k2/(2*k1) + 1/2*np.sqrt( (k2/k1)**2 + 4*((g+c[i])/k1) )
        
        velocity.append(v_alt)

    return [velocity]