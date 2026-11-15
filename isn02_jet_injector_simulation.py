#!/usr/bin/env python3
"""
================================================================================
ISN-JETPORT: TRANSDERMAL LORENTZ-FORCE MICRO-JET INJECTOR SIMULATION (v2.0)
Authors: Kallu Krishna Sumanth & Shiek Zaheer Khalander
================================================================================
"""
import numpy as np
import matplotlib.pyplot as plt

def simulate_transdermal_jet_injection(save_fig_path='ISN_JETPORT_Figure_IEEE.png'):
    B_field=1.25
    L_coil=9.8
    piston_mass=4e-3
    piston_damping=35.0
    piston_stiffness=1200.0
    D_piston=3e-3
    A_piston=np.pi*(D_piston/2)**2
    d_orifice=50e-6
    A_orifice=np.pi*(d_orifice/2)**2
    rho=1025.0
    mu=1.25e-3
    bulk_modulus=2.2e9
    chamber_volume_initial=100e-9
    dt=5e-6
    t=np.arange(0,45e-3,dt)
    n=len(t)
    current=np.where(t<=2e-3,12-750*t,2.1*np.exp(-(t-2e-3)/0.05))
    x=np.zeros(n);v=np.zeros(n);a=np.zeros(n)
    P=np.zeros(n);P[0]=101325
    jet=np.zeros(n);vol=np.zeros(n);depth=np.zeros(n)
    for i in range(1,n):
        F=B_field*current[i]*L_coil
        Ff=P[i-1]*A_piston
        a[i]=(F-piston_damping*v[i-1]-piston_stiffness*x[i-1]-Ff)/piston_mass
        v[i]=v[i-1]+a[i]*dt
        x[i]=x[i-1]+v[i]*dt
        Qp=max(v[i]*A_piston,0)
        dP=max(P[i-1]-101325,0)
        jet[i]=0.88*np.sqrt(2*dP/rho)
        Qn=A_orifice*jet[i]
        V=max(20e-9,chamber_volume_initial-A_piston*x[i])
        P[i]=max(101325,P[i-1]+(bulk_modulus/V)*(Qp-Qn)*dt)
        vol[i]=vol[i-1]+Qn*dt*1e9
        Rt=5e5*np.exp(1.8*depth[i-1])
        depth[i]=depth[i-1]+((rho*A_orifice*jet[i]**2)/Rt)*dt*1000
    plt.plot(t*1000,P/1e6)
    plt.xlabel('Time (ms)')
    plt.ylabel('Pressure (MPa)')
    plt.savefig(save_fig_path,dpi=300)
    print('Peak pressure MPa:',np.max(P)/1e6)

if __name__=='__main__':
    simulate_transdermal_jet_injection()
