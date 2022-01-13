# -*- coding: utf-8 -*-
"""
Created on Wed Jan 27 23:49:22 2021

@author: marta
"""

import numpy as np
from numpy.linalg import inv
from numpy.linalg import norm

def data_gathering(csv_file):
    h_sim = np.array(csv_file["alt"])
    v_sim = np.array(csv_file["vel_ii_mag"])
    y_sim = np.array(csv_file["gamma_ii"])
    t_sim = np.array(csv_file['time'])
    f_sim = np.array(csv_file["vi"])
    #semi-major axis
    a_sim = np.array(csv_file["a"])
    #eccentricity
    e_sim = np.array(csv_file["e"])
    r_sim = np.array(csv_file['pos_ii_mag'])
    rho_sim = np.array(csv_file['rho'])
    
    #begin data collection at h0=160 km
    data=np.where(h_sim<=160000)[0].tolist()
    initial=data[0]
    
    if initial != 0:
        w0=init(h_sim,v_sim,y_sim,f_sim,a_sim,e_sim,r_sim,t_sim,rho_sim)
        h0=w0[0]
        v0=w0[1]
        y0=w0[2]
        f0=w0[3]
        a0=w0[4]
        e0=w0[5]
        r0=w0[6]
        t0=w0[7]
        rho0=w0[8]
        
        h_sim = np.insert(h_sim[initial:],0,h0)
        v_sim = np.insert(v_sim[initial:],0,v0)
        y_sim = np.insert(y_sim[initial:],0,y0)
        f_sim = np.insert(f_sim[initial:],0,f0)
        a_sim = np.insert(a_sim[initial:],0,a0)
        e_sim = np.insert(e_sim[initial:],0,e0)
        r_sim = np.insert(r_sim[initial:],0,r0)
        t_sim = np.insert(t_sim[initial:],0,t0)
        rho_sim = np.insert(rho_sim[initial:],0,rho0)
        
    else:
        h0=h_sim[0]
        v0=v_sim[0]
        y0=y_sim[0]
        f0=f_sim[0]
        t0=t_sim[0]
        a0=a_sim[0]
        e0=e_sim[0]
        r0=r_sim[0]
        rho0=rho_sim[0]
        
    initial_conditions=[h0,v0,y0,f0,a0,e0,r0,t0,rho0]
    
    #data collection ends at 160 km    
    data=np.where(h_sim<=160000)[0].tolist()
    x=np.size(h_sim)-1
    final=data[-1]    
        
    if final != x:
        wf=fin(h_sim,v_sim,y_sim,f_sim,a_sim,e_sim,r_sim,t_sim,rho_sim)
        hf=wf[0]
        vf=wf[1]
        yf=wf[2]
        ff=wf[3]
        af=wf[4]
        ef=wf[5]
        rf=wf[6]
        tf=wf[7]
        rhof=wf[8]
        
        h_sim = np.insert(h_sim[:final+1],np.size(h_sim[:final+1]),hf)
        v_sim = np.insert(v_sim[:final+1],np.size(v_sim[:final+1]),vf)
        y_sim = np.insert(y_sim[:final+1],np.size(y_sim[:final+1]),yf)
        f_sim = np.insert(f_sim[:final+1],np.size(f_sim[:final+1]),ff)
        a_sim = np.insert(a_sim[:final+1],np.size(a_sim[:final+1]),af)
        e_sim = np.insert(e_sim[:final+1],np.size(e_sim[:final+1]),ef)
        r_sim = np.insert(r_sim[:final+1],np.size(r_sim[:final+1]),rf)
        t_sim = np.insert(t_sim[:final+1],np.size(t_sim[:final+1]),tf)
        rho_sim = np.insert(rho_sim[:final+1],np.size(rho_sim[:final+1]),rhof)
        
    else:
        hf=h_sim[-1]
        vf=v_sim[-1]
        yf=y_sim[-1]
        ff=f_sim[-1]
        tf=t_sim[-1]
        af=a_sim[-1]
        ef=e_sim[-1]
        rf=r_sim[-1]
        rhof=rho_sim[-1]
        
    final_conditions=[hf,vf,yf,ff,af,ef,rf,tf,rhof]
    
    return(h_sim,v_sim,y_sim,f_sim,a_sim,e_sim,r_sim,t_sim,rho_sim,initial_conditions,final_conditions)
           
    
def init(h,v,y,f,a,e,r,time,rho):
    
    data=np.where(h<=160000)[0].tolist()
    ind1=data[0]-1
    ind2=data[0]
    
    ph=np.polyfit([time[ind1],time[ind2]],[h[ind1],h[ind2]],deg=1)
    h0=160000
    t0=(h0-ph[1])/ph[0]
    
    pv=np.polyfit([time[ind1],time[ind2]],[v[ind1],v[ind2]],deg=1)
    v0=t0*pv[0]+pv[1]
    
    py=np.polyfit([time[ind1],time[ind2]],[y[ind1],y[ind2]],deg=1)
    y0=t0*py[0]+py[1]
    
    pf=np.polyfit([time[ind1],time[ind2]],[f[ind1],f[ind2]],deg=1)
    f0=t0*pf[0]+pf[1]
    
    pe=np.polyfit([time[ind1],time[ind2]],[e[ind1],e[ind2]],deg=1)
    e0=t0*pe[0]+pe[1]
    
    pa=np.polyfit([time[ind1],time[ind2]],[a[ind1],a[ind2]],deg=1)
    a0=t0*pa[0]+pa[1]
    
    pr=np.polyfit([time[ind1],time[ind2]],[r[ind1],r[ind2]],deg=1)
    r0=t0*pr[0]+pr[1]
    
    prho=np.polyfit([time[ind1],time[ind2]],[rho[ind1],rho[ind2]],deg=1)
    rho0=t0*prho[0]+prho[1]
    
    return [h0,v0,y0,f0,a0,e0,r0,t0,rho0]


def fin(h,v,y,f,a,e,r,time,rho):
    
    data=np.where(h<=160000)[0].tolist()
    ind1=data[-1]
    ind2=data[-1]+1
    
    ph=np.polyfit([time[ind1],time[ind2]],[h[ind1],h[ind2]],deg=1)
    hf=160000
    tf=(hf-ph[1])/ph[0]
    
    pv=np.polyfit([time[ind1],time[ind2]],[v[ind1],v[ind2]],deg=1)
    vf=tf*pv[0]+pv[1]
    
    py=np.polyfit([time[ind1],time[ind2]],[y[ind1],y[ind2]],deg=1)
    yf=tf*py[0]+py[1]
    
    pf=np.polyfit([time[ind1],time[ind2]],[f[ind1],f[ind2]],deg=1)
    ff=tf*pf[0]+pf[1]
    
    pe=np.polyfit([time[ind1],time[ind2]],[e[ind1],e[ind2]],deg=1)
    ef=tf*pe[0]+pe[1]
    
    pa=np.polyfit([time[ind1],time[ind2]],[a[ind1],a[ind2]],deg=1)
    af=tf*pa[0]+pa[1]
    
    pr=np.polyfit([time[ind1],time[ind2]],[r[ind1],r[ind2]],deg=1)
    rf=tf*pr[0]+pr[1]
    
    prho=np.polyfit([time[ind1],time[ind2]],[rho[ind1],rho[ind2]],deg=1)
    rhof=tf*prho[0]+prho[1]
    
    return [hf,vf,yf,ff,af,ef,rf,tf,rhof]

def perturbation_data(csv_file):
    h_sim = np.array(csv_file["alt"])
    t_sim = np.array(csv_file['time'])
    #velocity vector components
    Grav1=np.array(csv_file["gravity_ii[0]"])
    Grav2=np.array(csv_file["gravity_ii[1]"])
    Grav3=np.array(csv_file["gravity_ii[2]"])
    #drag
    Drag1=np.array(csv_file["drag_ii[0]"])
    Drag2=np.array(csv_file["drag_ii[1]"])
    Drag3=np.array(csv_file["drag_ii[2]"])
    #lift
    Lift1=np.array(csv_file["lift_ii[0]"])
    Lift2=np.array(csv_file["lift_ii[1]"])
    Lift3=np.array(csv_file["lift_ii[2]"])
    
    #more orbital elements
    Omega=np.array(csv_file["OMEGA"])
    w=np.array(csv_file["omega"])
    SAM=np.array(csv_file["h_ii_mag"])
    
    #begin data collection at h0=160 km
    data=np.where(h_sim<=160000)[0].tolist()
    initial=data[0]
    
    if initial != 0:
        w0=pert_init(h_sim,Grav1,Grav2,Grav3,Drag1,Drag2,Drag3,Lift1,Lift2,Lift3,Omega,w,SAM,t_sim)
        
        h0=160000
        grav1_0 = w0[0]
        grav2_0 = w0[1]
        grav3_0 = w0[2]
        drag1_0 = w0[3]
        drag2_0 = w0[4]
        drag3_0 = w0[5]
        lift1_0 = w0[6]
        lift2_0 = w0[7]
        lift3_0 = w0[8]
        Omega0 = w0[9]
        w0_1=w0[10]
        SAM0=w0[11]
        t0=w0[12]
        
        h_sim = np.insert(h_sim[initial:],0,h0)    
        Grav1 = np.insert(Grav1[initial:],0,grav1_0)
        Grav2 = np.insert(Grav2[initial:],0,grav2_0)
        Grav3 = np.insert(Grav3[initial:],0,grav3_0)
        Drag1 = np.insert(Drag1[initial:],0,drag1_0)
        Drag2 = np.insert(Drag2[initial:],0,drag2_0)
        Drag3 = np.insert(Drag3[initial:],0,drag3_0)     
        Lift1 = np.insert(Lift1[initial:],0,lift1_0)
        Lift2 = np.insert(Lift2[initial:],0,lift2_0)
        Lift3 = np.insert(Lift3[initial:],0,lift3_0)
        Omega = np.insert(Omega[initial:],0,Omega0)
        w = np.insert(w[initial:],0,w0_1)
        SAM = np.insert(SAM[initial:],0,SAM0)
        t_sim = np.insert(t_sim[initial:],0,t0)
        
    else:
        h0=h_sim[0]
        grav1_0 = Grav1[0]
        grav2_0 = Grav2[0]
        grav3_0 = Grav3[0]
        drag1_0 = Drag1[0]
        drag2_0 = Drag2[0]
        drag3_0 = Drag3[0]
        lift1_0 = Lift1[0]
        lift2_0 = Lift2[0]
        lift3_0 = Lift3[0]
        Omega0=Omega[0]
        w0_1=w[0]
        SAM0=SAM[0]
        t0=t_sim[0]
        
    initial_conditions=[grav1_0,grav2_0,grav3_0,drag1_0,drag2_0,drag3_0,lift1_0,lift2_0,lift3_0,Omega0,w0_1,SAM0]
    
    #data collection ends at 160 km    
    data=np.where(h_sim<=160000)[0].tolist()
    x=np.size(h_sim)-1
    final=data[-1]    
        
    if final != x:
        wf=pert_final(h_sim,Grav1,Grav2,Grav3,Drag1,Drag2,Drag3,Lift1,Lift2,Lift3,Omega,w,SAM,t_sim)
        
        hf=160000
        grav1_f = wf[0]
        grav2_f = wf[1]
        grav3_f = wf[2]
        drag1_f = wf[3]
        drag2_f = wf[4]
        drag3_f = wf[5]
        lift1_f = wf[6]
        lift2_f = wf[7]
        lift3_f = wf[8]
        Omegaf = wf[9]
        wf_1=wf[10]
        SAM_f=wf[11]
        tf=wf[12]

        h_sim = np.insert(h_sim[:final+1],np.size(h_sim[:final+1]),hf)

        Drag1 = np.insert(Drag1[:final+1],np.size(Drag1[:final+1]),drag1_f)
        Drag2 = np.insert(Drag2[:final+1],np.size(Drag2[:final+1]),drag2_f)
        Drag3 = np.insert(Drag3[:final+1],np.size(Drag3[:final+1]),drag3_f)
        Omega = np.insert(Omega[:final+1],np.size(Omega[:final+1]),Omegaf)
        w = np.insert(w[:final+1],np.size(w[:final+1]),wf_1)
        SAM = np.insert(SAM[:final+1],np.size(SAM[:final+1]),SAM_f)
        t_sim = np.insert(t_sim[:final+1],np.size(t_sim[:final+1]),tf)
        
    else:
        hf=h_sim[-1]
        grav1_f = Grav1[-1]
        grav2_f = Grav2[-1]
        grav3_f = Grav3[-1]
        drag1_f = Drag1[-1]
        drag2_f = Drag2[-1]
        drag3_f = Drag3[-1]
        lift1_f = Lift1[-1]
        lift2_f = Lift2[-1]
        lift3_f = Lift3[-1]
        Omegaf=Omega[-1]
        wf_1=w[-1]
        SAM_f=SAM[-1]
        tf=t_sim[-1]

        
    final_conditions=[grav1_f,grav2_f,grav3_f,drag1_f,drag2_f,drag3_f,lift1_f,lift2_f,lift3_f,Omegaf,wf_1,SAM_f]
    
    return(Grav1,Grav2,Grav3,Drag1,Drag2,Drag3,Lift1,Lift2,Lift3,Omega,w,SAM,initial_conditions,final_conditions)


def pert_init(h,Grav1,Grav2,Grav3,Drag1,Drag2,Drag3,Lift1,Lift2,Lift3,Omega,w,i,time):
    
    data=np.where(h<=160000)[0].tolist()
    ind1=data[0]-1
    ind2=data[0]
    
    ph=np.polyfit([time[ind1],time[ind2]],[h[ind1],h[ind2]],deg=1)
    h0=160000
    t0=(h0-ph[1])/ph[0]
    
    pGrav1=np.polyfit([time[ind1],time[ind2]],[Grav1[ind1],Grav1[ind2]],deg=1)
    Grav1_0=t0*pGrav1[0]+pGrav1[1]
    
    pGrav2=np.polyfit([time[ind1],time[ind2]],[Grav2[ind1],Grav2[ind2]],deg=1)
    Grav2_0=t0*pGrav2[0]+pGrav2[1]
    
    pGrav3=np.polyfit([time[ind1],time[ind2]],[Grav3[ind1],Grav3[ind2]],deg=1)
    Grav3_0=t0*pGrav3[0]+pGrav3[1]
    
    pdrag1=np.polyfit([time[ind1],time[ind2]],[Drag1[ind1],Drag1[ind2]],deg=1)
    drag1_0=t0*pdrag1[0]+pdrag1[1]
    
    pdrag2=np.polyfit([time[ind1],time[ind2]],[Drag2[ind1],Drag2[ind2]],deg=1)
    drag2_0=t0*pdrag2[0]+pdrag2[1]
    
    pdrag3=np.polyfit([time[ind1],time[ind2]],[Drag3[ind1],Drag3[ind2]],deg=1)
    drag3_0=t0*pdrag3[0]+pdrag3[1]
    
    pLift1=np.polyfit([time[ind1],time[ind2]],[Lift1[ind1],Lift1[ind2]],deg=1)
    Lift1_0=t0*pLift1[0]+pLift1[1]
    
    pLift2=np.polyfit([time[ind1],time[ind2]],[Lift2[ind1],Lift2[ind2]],deg=1)
    Lift2_0=t0*pLift2[0]+pLift2[1]
    
    pLift3=np.polyfit([time[ind1],time[ind2]],[Lift3[ind1],Lift3[ind2]],deg=1)
    Lift3_0=t0*pLift3[0]+pLift3[1]
    
    pomega=np.polyfit([time[ind1],time[ind2]],[Omega[ind1],Omega[ind2]],deg=1)
    Omega0=t0*pomega[0]+pomega[1]
    
    pw=np.polyfit([time[ind1],time[ind2]],[w[ind1],w[ind2]],deg=1)
    w0=t0*pw[0]+pw[1]
    
    pi=np.polyfit([time[ind1],time[ind2]],[i[ind1],i[ind2]],deg=1)
    i0=t0*pi[0]+pi[1]
    
    return [Grav1_0,Grav2_0,Grav3_0,drag1_0,drag2_0,drag3_0,Lift1_0,Lift2_0,Lift3_0,Omega0,w0,i0,t0]
    
def pert_final(h,Grav1,Grav2,Grav3,Drag1,Drag2,Drag3,Lift1,Lift2,Lift3,Omega,w,i,time):
    
    data=np.where(h<=160000)[0].tolist()
    ind1=data[-1]
    ind2=data[-1]+1
    
    ph=np.polyfit([time[ind1],time[ind2]],[h[ind1],h[ind2]],deg=1)
    hf=160000
    tf=(hf-ph[1])/ph[0]
    
    pGrav1=np.polyfit([time[ind1],time[ind2]],[Grav1[ind1],Grav1[ind2]],deg=1)
    Grav1_f=tf*pGrav1[0]+pGrav1[1]
    
    pGrav2=np.polyfit([time[ind1],time[ind2]],[Grav2[ind1],Grav2[ind2]],deg=1)
    Grav2_f=tf*pGrav2[0]+pGrav2[1]
    
    pGrav3=np.polyfit([time[ind1],time[ind2]],[Grav3[ind1],Grav3[ind2]],deg=1)
    Grav3_f=tf*pGrav3[0]+pGrav3[1]
    
    pdrag1=np.polyfit([time[ind1],time[ind2]],[Drag1[ind1],Drag1[ind2]],deg=1)
    drag1_f=tf*pdrag1[0]+pdrag1[1]
    
    pdrag2=np.polyfit([time[ind1],time[ind2]],[Drag2[ind1],Drag2[ind2]],deg=1)
    drag2_f=tf*pdrag2[0]+pdrag2[1]
    
    pdrag3=np.polyfit([time[ind1],time[ind2]],[Drag3[ind1],Drag3[ind2]],deg=1)
    drag3_f=tf*pdrag3[0]+pdrag3[1]
    
    pLift1=np.polyfit([time[ind1],time[ind2]],[Lift1[ind1],Lift1[ind2]],deg=1)
    Lift1_f=tf*pLift1[0]+pLift1[1]
    
    pLift2=np.polyfit([time[ind1],time[ind2]],[Lift2[ind1],Lift2[ind2]],deg=1)
    Lift2_f=tf*pLift2[0]+pLift2[1]
    
    pLift3=np.polyfit([time[ind1],time[ind2]],[Lift3[ind1],Lift3[ind2]],deg=1)
    Lift3_f=tf*pLift3[0]+pLift3[1]

    pomega=np.polyfit([time[ind1],time[ind2]],[Omega[ind1],Omega[ind2]],deg=1)
    Omegaf=tf*pomega[0]+pomega[1]
    
    pw=np.polyfit([time[ind1],time[ind2]],[w[ind1],w[ind2]],deg=1)
    wf=tf*pw[0]+pw[1]
    
    pi=np.polyfit([time[ind1],time[ind2]],[i[ind1],i[ind2]],deg=1)
    i_f=tf*pi[0]+pi[1]
    
    return [Grav1_f,Grav2_f,Grav3_f,drag1_f,drag2_f,drag3_f,Lift1_f,Lift2_f,Lift3_f,Omegaf,wf,i_f,tf]  
    
def rotate_basis(vel_0, theta, Omega0, i0):
    #theta=true anomaly + argument of periapsis
    #build rotational matrix
    ct=np.cos(theta)
    st=np.sin(theta)
    co=np.cos(Omega0)
    so=np.sin(Omega0)
    ci=np.cos(i0)
    si=np.sin(i0)
    R1=np.array([(co*ct)-(so*st*ci),(-co*st)-(so*ct*ci),so*si])
    R2=np.array([(so*ct)+(co*st*ci),(-so*st)+(co*ct*ci),(-co*si)])
    R3=np.array([(st*si),(ct*si),ci])
    #transformation matrix (this would be to get inertial base)
    R=np.array([[R1[0],R1[1],R1[2]],[R2[0],R2[1],R2[2]],[R3[0],R3[1],R3[2]]])
    vRTN=np.matmul(inv(R),vel_0)
    unit_vec=np.divide(vRTN,norm(vRTN))
    
    return(unit_vec,R)
    
#def average_rho
#could use the definition of altitude to begin with, (Lambert's time) get 
#average density from an exponential density model, and go from there?
    
def cfs_fin(h,v,y,time):
    
    data=np.where(h<=160000)[0].tolist()[-1]
    
    if data == np.where(h==h[-1])[0].tolist()[0]:
        return [h,v,y,time]
    
    else:
        ind1=data
        ind2=data+1
        
        ph=np.polyfit([time[ind1],time[ind2]],[h[ind1],h[ind2]],deg=1)
        hf=160000
        tf=(hf-ph[1])/ph[0]
        
        pv=np.polyfit([time[ind1],time[ind2]],[v[ind1],v[ind2]],deg=1)
        vf=tf*pv[0]+pv[1]
        
        py=np.polyfit([time[ind1],time[ind2]],[y[ind1],y[ind2]],deg=1)
        yf=tf*py[0]+py[1]
        
        h = np.insert(h[:data],np.size(h[:data]),hf)
        v = np.insert(v[:data],np.size(v[:data]),vf)
        y = np.insert(y[:data],np.size(y[:data]),yf)
        time = np.insert(time[:data],np.size(time[:data]),tf)
        
        
        return [h,v,y,time]

    
def velocity_IJK(mu, h, theta, Omega, omega, i, e):
    #theta=true anomaly + argument of periapsis
    #build rotational matrix
    ct=np.cos(theta)
    st=np.sin(theta)
    
    co=np.cos(omega)
    so=np.sin(omega)
    
    cO=np.cos(Omega)
    sO=np.sin(Omega)
    
    ci=np.cos(i)
    si=np.sin(i)
    
    vx=(-mu/h)*(cO*(st+e*so) + sO*(ct+e*co)*ci)
    vy=(-mu/h)*(sO*(st+e*so) - cO*(ct+e*co)*ci)
    vz=(mu/h)*(ct + e*co)*si
    #transformation matrix (this would be to get inertial base)
    v_vec= np.array([vx,vy,vz]) 
    v_mag=norm(v_vec)
    
    unit_vec = np.divide(v_vec,v_mag)
    
    #build rotational matrix
    ct=np.cos(theta)
    st=np.sin(theta)
    co=np.cos(Omega)
    so=np.sin(Omega)
    ci=np.cos(i)
    si=np.sin(i)
    R1=np.array([(co*ct)-(so*st*ci),(-co*st)-(so*ct*ci),so*si])
    R2=np.array([(so*ct)+(co*st*ci),(-so*st)+(co*ct*ci),(-co*si)])
    R3=np.array([(st*si),(ct*si),ci])
    #transformation matrix (this would be to get inertial base)
    R=np.array([[R1[0],R1[1],R1[2]],[R2[0],R2[1],R2[2]],[R3[0],R3[1],R3[2]]])
    
    return(v_mag,unit_vec,R)


def numerical_integration_fin(r,v,time,r0):
    
    data=np.where(r<=r0)[0].tolist()[-1]
    
    if data == np.where(r==r[-1])[0].tolist()[0]:
        return [r,v,time]
    
    else:
        ind1=data
        ind2=data+1
        
        ph=np.polyfit([time[ind1],time[ind2]],[r[ind1],r[ind2]],deg=1)
        rf=r0
        tf=(rf-ph[1])/ph[0]
        
        pv=np.polyfit([time[ind1],time[ind2]],[v[ind1],v[ind2]],deg=1)
        vf=tf*pv[0]+pv[1]
        
        r = np.insert(r[:data],np.size(r[:data]),rf)
        v = np.insert(v[:data],np.size(v[:data]),vf)
        time = np.insert(time[:data],np.size(time[:data]),tf)
        
        
        return [r,v,time]






