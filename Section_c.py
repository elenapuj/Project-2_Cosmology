import numpy as np
import matplotlib.pyplot as plt




#First we define some constants that we will need through this section

G = 6.67e-8  #cm³/gs²

H0 = 2.72e-18  #1/s

kB = 1.38e-16  #cm²kg/sK

T0 = 2.725  #K

hbar = 1.055e-27  #cm²g/s

c = 3e10  #cm/s

Neff = 3




#We will calculate the value of Omega_r0 here


Or = ( (8*(np.pi)**3) / 45 ) * ( G / (H0**2) ) * ( (kB*T0)**4 / ( (hbar**3) * (c**5) ) ) * ( 1 + Neff * (7/8) * (4/11)**(4/3) )

print("O_r =", Or)
