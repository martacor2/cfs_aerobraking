# -*- coding: utf-8 -*-
"""
Created on Tue Jan 19 18:42:11 2021

@author: marta
"""
import numpy as np
from scipy.optimize import fsolve

def Lambert(f0,e,a,mu):
    f_f = -f0
    E_initialstate = 2 * np.arctan(((1 - e) / (1 + e)) ** 0.5 * np.tan((f0) / 2)) # eccentric anomaly
    E_finalstate  = 2 * np.arctan(((1 - e) / (1 + e)) ** 0.5 * np.tan((f_f) / 2)) # eccentric anomaly
    delta_t = (a**3/mu)**0.5*((E_finalstate - e * np.sin(E_finalstate)) - (E_initialstate - e * np.sin(E_initialstate)))
    tp = delta_t/2
    
    return[delta_t, tp]

def Lambert2(nu,e,a,mu,Rp):
    t_array = [0]
    t0 = 0
    h_array = [(a*(1-e**2))/(1 + e*np.cos(nu[0]))-Rp]
    v_array = [np.sqrt(mu * (2/ ((a*(1-e**2))/(1 + e*np.cos(nu[0]))) - 1/a))]
    
    y_array = [ np.arctan( (e*np.sin(nu[0]))/(1+e*np.cos(nu[0]))) ]
    
    for i in np.arange(0,len(nu)-1):
        E_0 = 2 * np.arctan(((1 - e) / (1 + e)) ** 0.5 * np.tan((nu[i]) / 2))
        E_f  = 2 * np.arctan(((1 - e) / (1 + e)) ** 0.5 * np.tan((nu[i+1]) / 2))
    
        dt = (a**3/mu)**0.5*((E_f - e*np.sin(E_f)) - (E_0-e*np.sin(E_0)))
    
        t_array.append(t0+dt)
        
        t0 = t0+dt
    
        h = (a*(1-e**2))/(1 + e*np.cos(nu[i+1]))-Rp
        r = (a*(1-e**2))/(1 + e*np.cos(nu[i+1]))
        
        v = np.sqrt(mu * (2/r - 1/a))
        y = np.arctan( (e*np.sin(nu[i+1]))/(1+e*np.cos(nu[i+1])) )
    
        
        h_array.append(h)
        v_array.append(v)
        y_array.append(y)
    
    return[np.array(h_array), np.array(t_array), np.array(v_array), np.array(y_array)]


def Lambert3(t_array, e, a, mu, Rp, f0):
    
    temp_array = [t_array[0]]
    
    nu_initial = f0
    
    nu_array = [f0]
    
    h_array = [(a*(1-e**2))/(1 + e*np.cos(f0))-Rp]
    
    v_array = [np.sqrt(mu * (2/ ((a*(1-e**2))/(1 + e*np.cos(f0))) - 1/a))]
    
    y_array = [ np.arctan((e*np.sin(f0))/(1+e*np.cos(f0))) ]
    
    for i in np.arange(0,len(t_array)-1):
        
        dt = t_array[i+1]-t_array[i]
        
        E0 = 2 * np.arctan(((1 - e) / (1 + e)) ** 0.5 * np.tan((f0) / 2))
    
        data = (E0, a, e, mu, dt)
        Ef_guess = 0
        
        Ef = fsolve(func, Ef_guess, args = data)[0]
        
        nu_f = 2*np.arctan( np.tan(Ef/2) * ( (1-e)/(1+e) )**(-1/2) )
    
        h = (a*(1-e**2))/(1 + e*np.cos(nu_f))-Rp
        r = (a*(1-e**2))/(1 + e*np.cos(nu_f))
        
        v = np.sqrt(mu * (2/r - 1/a))
        y = np.arctan( (e*np.sin(nu_f))/(1+e*np.cos(nu_f)) )
    
        
        h_array.append(h)
        v_array.append(v)
        y_array.append(y)
        nu_array.append(nu_f)
        temp_array.append(t_array[i+1])
        
        # if nu_f > -nu_initial:
        #     break
        
        f0 = nu_f
    
    return[np.array(h_array), np.array(v_array), np.array(y_array),np.array(temp_array),np.array(nu_array)]



def func(x,*data):
    
    E0, a, e, mu, dt = data
    
    return (dt*(a**3/mu)**(-1/2)) + ( E0 - e*np.sin(E0) ) - ( x - e*np.sin(x) ) 
    
    
    

