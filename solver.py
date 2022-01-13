import numpy as np
import sympy as sym
from sympy import integrate, solve


P = [40.3180628310673, 0.0102170093134411, 551.964398658762]
Q = [16.8521775970777, 0.0138647056191996, 606.325100627082]
R = [9.89943743641637,  0.0012283955806910564 ,392.01854536843433]

pq = np.subtract(Q , P)
pr = np.subtract(R, P)

n = np.cross(pq,pr)

print(n)

x = sym.Symbol('x'); y = sym.Symbol('y'); z = sym.Symbol('z'); 

fun = n[0]*(x-P[0]) + n[1]*(y-P[1]) + n[2]*(z-P[2]) 

print(fun)

def plane_eqn(drag, tof):
    a_coeff = (118.604093540174 - 0.321883685743695*tof + 0.0948065459341457*drag)/(-5406.84886632796)
    return a_coeff

print(plane_eqn(6.711196070437156, 788.3896888659153))