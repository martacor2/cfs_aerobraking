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

csv_file=pd.read_csv("data/data_all_sims.csv")

name = np.array(csv_file["Simulation"])
hp = np.array(csv_file["hp"])
a_coeff = np.array(csv_file["a"])
b_coeff = np.array(csv_file["b"])
c_coeff = np.array(csv_file["c"])
h0 = np.array(csv_file["h0"])
v0 = np.array(csv_file["v0"])
y0 = np.array(csv_file["y0"])
hddot0 = np.array(csv_file["hddot0"])
tf_lamb = np.array(csv_file["tf_lamb"])


fig_acoeff = plt.figure("a_coeff", dpi=500, figsize=[6,4])
plt.figure("a_coeff")
plt.scatter(v0, a_coeff, color='green', marker='o')   
# plt.scatter(y0, a_coeff, color='blue', marker='o')    
plt.legend();plt.grid()
plt.xlabel("eh")
fig_acoeff.savefig("all_sims_figures/a_ceoff.png")
