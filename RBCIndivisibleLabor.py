# -*- coding: utf-8 -*-
"""
Created on Mon Feb  3 15:46:06 2020

@author: salvador.ortigueira
"""

# Basic RBC model with indivisible labor (log utility on consumption)
#
# Salvador Ortigueira
# February 3, 2020

# 0. Initialization
import numpy as np
import math
import matplotlib.pyplot as plt

from numpy.linalg import inv



    ##  1. Calibration

aalpha = 0.39    
bbeta  = 0.95       
ppsi = 1.25
ddelta = 0.09
rrho = 0.9


    ## 2. Steady State 
laborSS         = (((1-aalpha)/ppsi)*((1/bbeta)-1+ddelta))/(-ddelta*aalpha+((1/bbeta)-1+ddelta))        
capitalSS       = ((1/aalpha)*((1/bbeta)-1+ddelta)*laborSS**(aalpha-1))**(1/(aalpha-1))
outputSS        = capitalSS**aalpha
consumptionSS   = ((1/aalpha)*((1/bbeta)-1+ddelta) -ddelta)*capitalSS

print ("Output = ", outputSS, " Capital = ", capitalSS, " Consumption = ", consumptionSS, "Labor =", laborSS) 
    
    
    ## Log-linearized sytem of equations: The first two equations are for z and k; and the third for consumption 
    
a11 =  rrho
a12 = 0
a13 = 0 
    
a21 = capitalSS**(aalpha-1)*laborSS**(1-aalpha) + (ppsi*consumptionSS*laborSS)/(aalpha*capitalSS)
a22 = (1/bbeta) + ((ppsi*consumptionSS*laborSS)/(capitalSS))*capitalSS**(aalpha-1)
a23 = (ppsi*consumptionSS*laborSS)/(aalpha*capitalSS) + (consumptionSS/capitalSS)
    
a31 = ((rrho-(1-ddelta)*bbeta*rrho))/(1-(1-aalpha)*(1-ddelta)*bbeta)
a32 = 0
a33 = (aalpha)/(1-(1-aalpha)*(1-ddelta)*bbeta)
    
mA = np.array([[a11, a12, a13],
                  [a21, a22, a23],
                  [a31, a32, a33]],float)
    
print ("Matrix Log-linearized system = ", mA)    

    ## Diagonalize matrix

eigenvalues, eigenvectors = np.linalg.eig(mA)


idx = eigenvalues.argsort()
print("index",idx)
eigenvalues = eigenvalues[idx]
V=eigenvectors[:,idx]
D=np.diag(eigenvalues)
print(D)
print(V)



## Select the submatrixes of D and V 

D_1    = D[0:2,0:2]    
V_11   = V[0:2,0:2]
V_21   = V[2:, 0:2]

##  Coefficients in the solution of C 

VC =  V_21 @ inv(V_11)  ## Vector of coef in consumption function

C_z = VC[0:1, 0:1]        ## Coef on z in consumption function
C_k = VC[0:1, 1:]         ## Coef on k in consumption function

##  Coefficients in the solution of X 

VX = V_11 @ D_1 @ inv(V_11)

Z_z = VX[0:1,0:1]
Z_k = VX[0:1,1:]

K_z = VX[1:,0:1]
K_k = VX[1:,1:]


## Generate transformed technology levels (percentage deviation from steady-state value)
 
        ## Generate shocks from N(0,sigma)

mu, sigma = 0, 0.01
et = np.random.normal(mu, sigma, 1000)

count, bins, ignored = plt.hist(et, 30, normed=True)
plt.plot(bins, 1/(sigma*np.sqrt(2*np.pi))*np.exp(-(bins-mu)**2/(2*sigma**2)), linewidth = 2, color = 'r')
plt.show()


index = range(1000)

zt = np.zeros(len(index))

zt0 = 0
zt[0] = zt0   

for n in index[1:]:
    zt[n] = Z_z * zt[n-1] + et[n]
    
plt.plot(index, zt, linewidth = 2, color = 'b') 
plt.title('Technology (percentage deviation from steady-state value)')
plt.show()  
 
## Generate transformed levels of physical capital (percentage deviation from steady-state value)

kt = np.zeros(len(index))  
 
kt0 = 0
kt[0] = kt0

for n in index[1:]:
    kt[n] = K_z * zt[n-1] +  K_k * kt[n-1] 
    
plt.plot(index, kt, linewidth = 2, color = 'b')
plt.title('Physical Capital (percentage deviation from steady-state value)')
plt.show()

## Generate transformed levels of consumption (percentage deviation from steady-state value)
 
ct = np.zeros(len(index)) 

for n in index[0:]:
    ct[n] = C_z * zt[n] + C_k * kt[n]

plt.plot(index, ct, linewidth = 2, color = 'b')
plt.title('Consumption (percentage deviation from steady-state value)')
plt.show()    

## Generate the transformed levels of labor  (percentage deviation from steady-state value)

nt = (1/aalpha)*zt + (capitalSS**(aalpha-1))*kt - (1/aalpha)*ct  

plt.plot(index, nt, linewidth = 2, color = 'b')
plt.title('Labor (percentage deviation from steady-state value)')
plt.show()    


##  Generate levels of technology

z = np.exp(zt)

plt.plot(index, z, linewidth = 2, color = 'b')
plt.title('Technology')
plt.show() 

## Generate levels of capital

k = (1 + kt)*capitalSS 

plt.plot(index, k, linewidth = 2, color = 'b')
plt.title('Physical Capital')
plt.show() 

## Generate levels of consumption

c = (1 + ct)*consumptionSS 

plt.plot(index, c, linewidth = 2, color = 'b')
plt.title('Consumption')
plt.show()

## Generate levels of labor 

n =  (1 + nt)*laborSS 

plt.plot(index, n, linewidth = 2, color = 'b')
plt.title('Labor')
plt.show()

## Generate levels of output

y = z*(k**aalpha)*(n**(1-aalpha))

plt.plot(index, y, linewidth = 2, color = 'b')
plt.title('Output')
plt.show()


## Generate cycle by applying a Hoddrick-Prescott filter

import statsmodels.api as sm

cycle, trend = sm.tsa.filters.hpfilter(y, 1600)

plt.plot(index, y, linewidth = 2, color = 'b', label='Output')
plt.plot(index, trend, linewidth = 2, color = 'r', label='Trend')
plt.title('Output and its Trend')
plt.legend()

plt.plot(index, cycle, linewidth = 2, color = 'b')
plt.title('Business Cycle, Output')
plt.show()






 
    
 