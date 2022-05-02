import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp


#First, we define some constants that we will need through this section

G = 6.67e-8  #cm³/gs²

H0 = 2.72e-18  #1/s

kB = 1.38e-16  #cm²kg/sK

T0 = 2.725  #K

hbar = 1.055e-27  #cm²g/s

c = 3e10  #cm/s

Neff = 3

tau = 1700  #s

q = 2.53

Z = 5.93e9

Or = ( (8*(np.pi)**3) / 45 ) * ( G / (H0**2) ) * ( (kB*T0)**4 / ( (hbar**3) * (c**5) ) ) * ( 1 + Neff * (7/8) * (4/11)**(4/3) )

Yn0 = 1 / ( 1 + np.exp( q * Z / (100e9) ) )

Yp0 = 1 - Yn0




#Then, we will introduce some functions in order to calculate the differential equation using solve_ivp

def I(x, t, q):

	T = np.exp(t)

	Tv = ( (4/11)**(1/3) ) * T

	I1 = ( (x+q)**2 * (x**2 - 1)**(1/2) * x ) / ( (1 + np.exp( x*Z/T )) * (1 + np.exp( -(x+q)*Z/Tv) ) )

	I2 = ( (x-q)**2 * (x**2 - 1)**(1/2) * x ) / ( (1 + np.exp( -x*Z/T )) * (1 + np.exp( (x-q)*Z/Tv )) )

	return((I1+I2)/tau)



def Int(t, q):

	x = np.linspace(1, 100, 1000)
	dx = x[1] - x[0]
	integral = np.trapz(I(x, t, q), dx=dx)

	return(integral)




#After that we introduce the actual differential equation we want to calculate

def Y(t, y, h0, OR, Q, t0):

	H = H0*np.sqrt(Or)*np.exp(2*t)/(T0**2)

	return[ -(1/H) * ( y[1]*Int(t, -q) - y[0]*Int(t, q) ), -(1/H) * ( y[0]*Int(t, q)  - y[1]*Int(t, -q) ) ]



#And we actually calculate the differential equation

D = solve_ivp(Y, [np.log(100e9), np.log(0.1e9)], [Yn0, Yp0], args = (H0, Or, q, T0) ,method = 'Radau', rtol = 1e-12, atol = 1e-12)





#Finally, we produce the plot

T_ = np.exp(D.t)


Yn = 1 / ( 1 + np.exp( q * Z / T_ ) )
Yp = 1 - Yn


plt.plot(T_, D.y[0,:], label = 'n', color = 'orange')
plt.plot(T_, D.y[1,:], label = 'p', color = 'blue')
plt.plot(T_, Yn, ':', color = 'orange')
plt.plot(T_, Yp, ':', color = 'blue')
plt.xlabel('T[K]')
plt.ylabel(r'$Y_i$')
plt.xlim([100e9, 0.1e9])
plt.ylim([1e-3, 2])
plt.xscale('log')
plt.yscale('log')
plt.legend()
plt.title('Relative number density')
plt.savefig('Figure 1.pdf')
