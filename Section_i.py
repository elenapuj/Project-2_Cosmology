import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp


#First, we define some constants that we will need through this section

G = 6.67e-8  #cm³/gs²

H0 = 2.72e-18  #1/s

pc0 = 9.2e-30  #g/cm³

Ob0 = 0.05

kB = 1.38e-16  #cm²g/sK

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

YD0 = YT0 = YHe30 = YHe40 = YLi70 = YBe70 = 0




#Then, we will introduce some functions in order to calculate the differential equation using solve_ivp

def I(x, t, q):

        T = np.exp(t)

        Tv = ( (4/11)**(1/3) ) * T

        I1 = ( (x+q)**2 * (x**2 - 1)**(1/2) * x ) / ( (1 + np.exp( x*Z/T )) * (1 + np.exp( -(x+q)*Z/Tv) ) )

        I2 = ( (x-q)**2 * (x**2 - 1)**(1/2) * x ) / ( (1 + np.exp( -x*Z/T )) * (1 + np.exp( (x-q)*Z/Tv )) )

        return((I1+I2)/tau)



def Lw(t, q):

        x = np.linspace(1, 100, 1000)
        dx = x[1] - x[0]
        integral = np.trapz(I(x, t, q), dx=dx)

        return(integral)



def pn(t):

	PN = 2.5e4 * Ob0 * pc0 * np.exp(3*t)/(T0**3)

	return(PN)



def pD(t):

	T = np.exp(t)

	T9 = T / (1e9)

	pd = 2.23e3 * Ob0 * pc0 * np.exp(3*t)/(T0**3) * T9**(-2/3) * np.exp(-3.72 * T9**(-1/3) ) * ( 1 + 0.112 * T9**(1/3)  + 3.38 * T9**(2/3) + 2.65 * T9 )

	return(pd)



def nD(t):

	nd = Ob0 * pc0 * np.exp(3*t)/(T0**3) * ( 75.5 + 1250 * ( np.exp(t) / (1e9) ) )

	return(nd)



def nHe3_p(t):

	nhe3 = 7.06e8 * Ob0 * pc0 * np.exp(3*t)/(T0**3)

	return(nhe3)



def pT_n(t):

	pt = nHe3_p(t) * np.exp(-8.864e9 / np.exp(t))

	return(pt)



def pT_g(t):

	T = np.exp(t)

	T9 = T / (1e9)

	pt = 2.87e4 * Ob0 * pc0 * np.exp(3*t)/(T0**3) * T9**(-2/3) * np.exp(-3.87 * T9**(-1/3)) * ( 1 + 0.108 * T9**(1/3) + 0.466 * T9**(2/3) + 0.352 * T9 + 0.3 * T9**(4/3) + 0.576 * T9**(5/3) )

	return(pt)



def nHe3_g(t):

	nhe3 = 6e3 * Ob0 * pc0 * np.exp(3*t)/(T0**3) * ( np.exp(t) / (1e9) )

	return(nhe3)



def DD_n(t):

	T = np.exp(t)

	T9 = T / (1e9)

	dd = 3.9e8 * Ob0 * pc0 * np.exp(3*t)/(T0**3) * T9**(-2/3) * np.exp(-4.26 * T9**(-1/3)) * ( 1 + 0.0979 * T9**(1/3) + 0.642 * T9**(2/3) + 0.44 * T9 )

	return(dd)



def nHe3_D(t):

	nhe = 1.73 * DD_n(t) * np.exp(-37.94e9 / np.exp(t))

	return(nhe)



def pT_D(t):

	pt = 1.73 * DD_n(t) * np.exp(-46.80e9 / np.exp(t))

	return(pt)



def DD_g(t):

	T = np.exp(t)

	T9 = T / (1e9)

	dd = 24.1 * Ob0 * pc0 * np.exp(3*t)/(T0**3) * T9**(-2/3) * np.exp(-4.26 * T9**(-1/3)) * ( T9**(2/3) + 0.685 * T9 + 0.152 * T9**(4/3) + 0.265 * T9**(5/3) )

	return(dd)



def DHe3(t):

	T = np.exp(t)

	T9 = T / (1e9)

	dhe = 2.6e9 * Ob0 * pc0 * np.exp(3*t)/(T0**3) * T**(-3/2) * np.exp(-2.99 / T9)

	return(dhe)



def He4p(t):

	hep = 5.5 * DHe3(t) * np.exp(-213e9 / np.exp(t))

	return(hep)



def DT(t):

	T = np.exp(t)

	T9 = T / (1e9)

	dt = 1.38e9 * Ob0 * pc0 * np.exp(3*t)/(T0**3) * T9**(-3/2) * np.exp(-0.745 / T9)

	return(dt)




def He4n(t):

	hen = 5.5 * DT(t) * np.exp(-204.1e9 / np.exp(t))

	return(hen)



def He3T_D(t):

	T = np.exp(t)

	T9 = T / (1e9)

	het = 3.88e9 * Ob0 * pc0 * np.exp(3*t)/(T0**3) * T9**(-2/3) * np.exp(-7.72 * T9**(-1/3)) * ( 1 + 0.054 * T**(1/3) )

	return(het)



def He4D(t):

	hed = 1.59 * He3T_D(t) * np.exp(-166.2e9 / np.exp(t))

	return(hed)



def He3He4(t):

	T = np.exp(t)

	T9 = T / (1e9)

	he = 4.8e6 * Ob0 * pc0 * np.exp(3*t)/(T0**3) * T9**(-2/3) * np.exp(-12.8 * T9**(-1/3)) * ( 1 + 0.0326 * T9**(1/3) - 0.219 * T9**(2/3) - 0.0499 * T9 + 0.0258 * T9**(4/3) + 0.015 * T9**(5/3) )

	return(he)



def THe4(t):

	T = np.exp(t)

	T9 = T / (1e9)

	the = 5.28e5 *  Ob0 * pc0 * np.exp(3*t)/(T0**3) * T9**(-2/3) * np.exp(-8.08 * T9**(-1/3)) * ( 1 + 0.0516 * T9**(1/3))

	return(the)



def nBe7_p(t):

	nbe = 6.74e9 * Ob0 * pc0 * np.exp(3*t)/(T0**3)

	return(nbe)



def pLi7_n(t):

	pli = nBe7_p(t) * np.exp(-19.07e9 / np.exp(t))

	return(pli)



def pLi7_He4(t):

	T = np.exp(t)

	T9 = T / (1e9)

	pli = 1.42e9 * Ob0 * pc0 * np.exp(3*t)/(T0**3) * T9**(-2/3) * np.exp(-8.47 * T9**(-1/3)) * ( 1 + 0.0493 * T9**(1/3) )

	return(pli)



def He4He4_p(t):

	he = 4.64 * pLi7_He4(t) * np.exp(-201.3e9 / np.exp(t))

	return(he)



def nBe7_He4(t):

	nbe = 1.2e7 * Ob0 * pc0 * np.exp(3*t)/(T0**3) * np.exp(t) / (1e9)

	return(nbe)



def He4He4_n(t):

	he = 4.64 * nBe7_He4(t) * np.exp(-220.4e9 / np.exp(t) )

	return(he)




def LD(t):

	T = np.exp(t)

	ld = 4.68e9 * 2.5e4  * ( T / (1e9) )**(3/2) * np.exp(-25.82e9/T)

	return(ld)



def LHe3(t):

	T = np.exp(t)

	T9 = T / (1e9)

	lhe = 1.63e10 * pD(t) * ( 1 / Ob0 * pc0 * np.exp(3*t)/(T0**3) )  * T9**(3/2) * np.exp(-63.75e9 / T)

	return(lhe)



def LT(t):

	T = np.exp(t)

	T9 = T / (1e9)

	lt = 1.63e10 * nD(t) * ( 1 / Ob0 * pc0 * np.exp(3*t)/(T0**3) ) * T9**(3/2) * np.exp(-72.62 / T9)

	return(lt)



def LHe4_p(t):

	T = np.exp(t)

	T9 = T / (1e9)

	lhe = 2.59e10 * pT_g(t) * ( 1 / Ob0 * pc0 * np.exp(3*t)/(T0**3) ) * T9**(3/2) * np.exp(-229.9 / T9)

	return(lhe)



def LHe4_n(t):

	T = np.exp(t)

	T9 = T / (1e9)

	lhe = 2.6e10 * nHe3_g(t) * ( 1 / Ob0 * pc0 * np.exp(3*t)/(T0**3) ) * T9**(3/2) * np.exp(-238.8 / T9 )

	return(lhe)



def LHe4_D(t):

	T = np.exp(t)

	T9 = T / (1e9)

	lhe = 4.5e10 * DD_g(t) * ( 1 / Ob0 * pc0 * np.exp(3*t)/(T0**3) ) * T9**(3/2) * np.exp(-276.7 / T9)

	return(lhe)



def LBe7(t):

	T = np.exp(t)

	T9 = T / (1e9)

	lbe = 1.12e10 * He3He4(t) * ( 1 / Ob0 * pc0 * np.exp(3*t)/(T0**3) ) * T9**(3/2) * np.exp(-18.42 / T9)

	return(lbe)



def LLi7(t):

	T = np.exp(t)

	T9 = T / (1e9)

	lli = 1.12e10 * THe4(t) * ( 1 / Ob0 * pc0 * np.exp(3*t)/(T0**3) ) * T9**(3/2) * np.exp(-28.63 / T9)

	return(lli)




#After that we introduce the actual differential equation we want to calculate

def Y(t, y, h0, OR, Q, t0):

	H = H0*np.sqrt(Or)*np.exp(2*t)/(T0**2)

	Yn = -(1/H) * ( Lw(t, -q)*y[1] - Lw(t, q)*y[0] + LD(t) * y[2] - pn(t) * y[0] * y[1] - nD(t) * y[0] * y[2] + LT(t) * y[3] - nHe3_p(t) * y[0] * y[4] + pT_n(t) * y[1] * y[3] - nHe3_g(t) * y[0] * y[4] + LHe4_n(t) * y[5] + 0.5*DD_n(t) * y[2]**2 - nHe3_D(t) * y[0] * y[4] + DT(t) * y[2] * y[3] - He4n(t) * y[5] * y[0] - nBe7_p(t) * y[0] * y[7] + pLi7_n(t) * y[1] * y[6] - nBe7_He4(t) * y[0] * y[7] + 0.5*He4He4_n(t) * y[5]**2 )

	Yp = -(1/H) * ( Lw(t, q)*y[0] - Lw(t, -q)*y[1] + LD(t) * y[2] - pn(t) * y[0] * y[1] - pD(t) * y[1] * y[2] + LHe3(t) * y[4] + nHe3_p(t) * y[0] * y[4] - pT_n(t) * y[1] * y[3] - pT_g(t) * y[1] * y[3] + LHe4_p(t) * y[5] + 0.5*DD_n(t) * y[2]**2 - pT_D(t) * y[1] * y[3] + DHe3(t) * y[2] * y[4] - He4p(t) * y[5] * y[1] + nBe7_p(t) * y[0] * y[7] - pLi7_n(t) * y[1] * y[6] - pLi7_He4(t) * y[1] * y[6] + 0.5*He4He4_p(t) * y[5]**2 )

	YD = -(1/H) * ( -LD(t) * y[2] + pn(t) * y[0] * y[1] - y[1] * y[2] * pD(t) + y[4] * LHe3(t) - nD(t) * y[0] * y[2] + LT(t) * y[3] - DD_n(t) * y[2]**2 + 2*nHe3_D(t) * y[0] * y[4] - DD_n(t) * y[2]**2 + 2*pT_D(t) * y[1] * y[3] - DD_g(t) * y[2]**2 + 2*LHe4_D(t) * y[5] - DHe3(t) * y[2] * y[4] + He4p(t) * y[5] * y[1] - DT(t) * y[2] * y[3] + He4n(t) * y[5] * y[0] + He3T_D(t) * y[4] * y[3] - He4D(t) * y[5] * y[2] )

	YT = -(1/H) * ( nD(t) * y[0] * y[2] - LT(t) * y[3] + nHe3_p(t) * y[0] * y[4] - pT_n(t) * y[1] * y[3] - pT_g(t) * y[1] * y[3] + LHe4_p(t) * y[5] + 0.5*DD_n(t) * y[2]**2 - pT_D(t) * y[1] * y[3] - DT(t) * y[2] * y[3] + He4n(t) * y[5] * y[0] - He3T_D(t) * y[4] * y[3] + He4D(t) * y[5] * y[2] - THe4(t) * y[3] * y [5] + LLi7(t) * y[6] )

	YHe3 = - (1/H) * ( y[1] * y[2] * pD(t) - y[4] * LHe3(t) - nHe3_p(t) * y[0] * y[4] + pT_n(t) * y[1] * y[3] - nHe3_g(t) * y[0] * y[4] + LHe4_n(t) * y[5] + 0.5*DD_n(t) * y[2]**2 - nHe3_D(t) * y[0] * y[4] - DHe3(t) * y[2] * y[4] + He4p(t) * y[5] * y[1] - He3T_D(t) * y[4] * y[3] + He4D(t) * y[5] * y[2] - He3He4(t) * y[4] * y[5] + LBe7(t) * y[7] )

	YHe4 = -(1/H) * ( pT_g(t) * y[1] * y[3] - LHe4_p(t) * y[5] + nHe3_g(t) * y[0] * y[4] - LHe4_n(t) * y[5] + 0.5*DD_g(t) * y[2]**2 - LHe4_D(t) * y[5] + DHe3(t) * y[2] * y[4] - He4p(t) * y[5] * y[1] + DT(t) * y[2] * y[3] - He4n(t) * y[5] * y[0] + He3T_D(t) * y[4] * y[3] - He4D(t) * y[4] * y[2] - He3He4(t) * y[4] * y[5] + LBe7(t) * y[7] - THe4(t) * y[3] * y[5] + LLi7(t) * y[6] + 2*pLi7_He4(t) * y[1] * y[6] - He4He4_p(t) * y[5]**2 + 2*nBe7_He4(t) * y[0] * y[7] - He4He4_n(t) * y[5]**2 )

	YLi7 = -(1/H) * ( THe4(t) * y[3] * y[5] - LLi7(t) * y[6] + nBe7_p(t) * y[0] * y[7] - pLi7_n(t) * y[1] * y[6] - pLi7_He4(t) * y[1] * y[6] + 0.5*He4He4_p(t) * y[5]**2 )

	YBe7 = -(1/H) * ( He3He4(t) * y[4] * y[5] - LBe7(t) * y[7] - nBe7_p(t) * y[0] * y[7] + pLi7_n(t) * y[1] * y[6] - nBe7_He4(t) * y[0] * y[7] + 0.5*He4He4_n(t) * y[5]**2 )

	return[ Yn, Yp, YD, YT, YHe3, YHe4, YLi7, YBe7 ]




#And we actually calculate the differential equation

D = solve_ivp( Y, [np.log(100e9), np.log(0.01e9)], [Yn0, Yp0, YD0, YT0, YHe30, YHe40, YLi70, YBe70], args = (H0, Or, q, T0), method = 'Radau', rtol = 1e-12, atol = 1e-12 )




#Finally, we produce the plot

T_ = np.exp(D.t)


plt.plot(T_, D.y[0,:], label = 'n', color = 'orange')
plt.plot(T_, D.y[1,:], label = 'p', color = 'blue')
plt.plot(T_, 2*D.y[2,:], label = 'D', color = 'mediumseagreen')
plt.plot(T_, 3*D.y[3,:], label = 'T', color = 'red')
plt.plot(T_, 3*D.y[4,:], label = 'He³', color = 'gold')
plt.plot(T_, 4*D.y[5,:], label = 'He⁴', color = 'deeppink')
plt.plot(T_, 7*D.y[6,:], label = 'Li⁷', color = 'mediumpurple')
plt.plot(T_, 7*D.y[2,:], label = 'Be⁷', color = 'midnightblue')
plt.xlabel('T[K]')
plt.ylabel(r'$A_iY_i$')
plt.xlim([100e9, 0.01e9])
plt.ylim([1e-11, 1e1])
plt.xscale('log')
plt.yscale('log')
plt.legend()
plt.title('Mass fraction')
plt.savefig('Figure 3.pdf')
