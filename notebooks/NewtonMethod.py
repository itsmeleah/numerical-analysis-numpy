#!/usr/bin/env python
# coding: utf-8
from math import *
import matplotlib.pyplot as plt
import numpy as np
from decimal import Decimal
import random


#***************************************************************************
# Simulation for Administering True Positive Results for Coronavirus Test **
#***************************************************************************

def ConditionalProb():
    NUM_RUNS = 100000
    total_test_positive = 0
    total_test_pos_and_disease_present = 0

    population = 1000
    infected = int(population * .05)

    for i in range(NUM_RUNS):
        # A list of random integers representing the 5% of the population with the disease
        disease_list = random.sample(range(int(population)), infected)
        # Randomly select an integer in the range of the population to represent selected person
        person_selected = np.random.choice(range(int(population)))
        #test this person. In a range from 1-100 any number can represent the one percent chance the test is negative
        test_result = np.random.choice(range(1,100))
        #check to see if this person is in the list of the sick population
        if person_selected in disease_list:
        
            # If the random number isnt the one percent then increment the totals
            if test_result != 1:
                total_test_positive += 1
                total_test_pos_and_disease_present += 1
        # if the person selected is not among the diseased
        else: 
        
        # if the healthy person recieves a positive test, 
            if test_result == 1:
                total_test_positive += 1

    # divide the total number of true positives by the total number of positives
    prob = total_test_pos_and_disease_present/ total_test_positive

    print("-----------------------------------------------------")
    print("| Number of positive tests ", total_test_positive)
    print("| Test positive and has disease ", total_test_pos_and_disease_present) 
    print("| The probability of a person who tested positive to actualy have the disease: %.6f" % prob)
    print("------------------------------------------------------")

ConditionalProb()


#*********************************************************************
# Evaluating a 10-dimensional integral using Monte Carlo simulation **
#*********************************************************************

print("\n***********************")
print("Section 2, Problem 1")
print("***********************\n")

# Define the function that we are integrating
def fx(x1, x2, x3, x4, x5, x6, x7, x8, x9, x10):
        return x1 * x2 * x3 * x4 * x5 * x6 * x7 * x8 * x9 * x10

def section2Problem1(N):
    # Generate N 10-tuples of uniform random values between 0 and 2
    points = np.random.uniform(0, 2, (N, 10))
    
    # Store the volume of the 10-D region of integration
    vol = 2 ** 10
    
    # Accumulators for the values squares of the values of the function to calculate the variance
    fxVal = 0
    fxValSq = 0
    
    # Loop to accumulate the values for each point
    for point in points:
        fxVal += fx(*point)
        fxValSq += fx(*point) ** 2
    
    # Divide by the number of points to get the average value, multiply by the volume of the region
    # to get our estimated value of the integration of our function over this region
    value = (fxVal / N) * vol
    
    # Variance: (expected value of fxValSq - square of fxVal) divided by N and multiplied by
    # the square of the volume of the region
    var = (1 / N) * ((fxValSq / N) - (fxVal / N) ** 2) * vol ** 2
    
    # Standard deviation in the value
    std = np.sqrt(var)
    
    # Number of standard deviations from the actual value, 1024
    num_deviations = abs(1024 - value) / std

    print("For %d points: " % (N))
    print("Estimated value: %.6f" % (value))
    print("Standard deviation: %.6f" % (std))
    print("Number of deviations from actual value: %.6f \n" % (num_deviations))
    

numPoints = [10000, 40000, 160000]

for N in numPoints:
    section2Problem1(N)


#********************
# Newton's Method ***
#********************

print("\n***********************")
print("Section 3, Part a")
print("***********************\n")

# Define function for Newton method
def newton(f, fprime, x0, delta):
    i = 0
    c = x0
    Ci = [c]
    fc = f(x0)
    # Plot the initial step for Newton method
    if x0 == 1.08:
        x = np.arange(-3,3, 1e-6)
    else:
        x = np.arange(-3,15, 1e-6) 
    fig, ax = plt.subplots(figsize=(14,12))  # width = 14, height = 12
    ax.plot(x,np.tanh(x),color = "red") # plot the graph of function y = tanh(x)
    ax.axhline(0) # x-axis
    ax.set_title('Newton Method for tanh(x) at x0 = %.2f' % (x0))
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    bbox = dict(boxstyle="round", fc="0.8")
    arrowprops = dict(arrowstyle="<-", connectionstyle="angle,angleA=90,angleB=0,rad=5")
    offset = 20
    offset1 = 70
    # Initial guess
    print("Initial guess: x = %.5e, fx = %.5e, error = %.5e" % (c, fc, abs(2-c))) # output initial guess 
    ax.scatter(c, fc, alpha=.8, marker ="o") # scatter a dot for initial guess 
    ax.axvline(c, alpha=0.2, color='black', linestyle='--') # vertical guide line
    ax.annotate('Initial guess: x=%.2e' % c ,(c,fc),xytext=(-2*offset1, offset1), textcoords='offset points',bbox=bbox, arrowprops=arrowprops)
    # Check to see if the initial guess satisfies the convergence critirion
    if abs(fc) <= delta:
        return
    while abs(fc) > delta:
        fpc = fprime(c)
        i += 1
        # If f prime = 0, abort
        if fpc == 0:
            print("fprime is 0. Aborted.") # print the error and exists
            return
        else:
            c = c - fc/fpc # Newton's step
            fc = f(c)
            Ci.append(c)
            print("Step %d: x = %.5e, fx = %.5e, error = %.5e" % (i,c,fc,abs(0-c))) # Output from step 2 onward  
            ax.scatter(c, fc, alpha=.8, marker ="o") # scatter points for each root
            ax.plot([c,Ci[i-1]], [0,f(Ci[i-1])], alpha=.8) # tangent lines
            ax.axvline(c, alpha=0.2, color='black', linestyle='--') # vertical guide lines
        if c < 0:          
            ax.annotate('Step %d: x=%.2e'%(i,Ci[i]),(Ci[i], f(Ci[i])),xytext=(-2*(offset+20*i), offset+20*i), textcoords='offset points', bbox=bbox, arrowprops=arrowprops)
        else:
            ax.annotate('Step %d: x=%.2e'%(i,Ci[i]),(Ci[i], f(Ci[i])),xytext=(0.5*(offset+30*i),-offset-30*i), textcoords='offset points',bbox=bbox, arrowprops=arrowprops)
        #Uncomment the 2 lines below to see how the method behaves before breaking
        #if i == 6:
            #break
fx = lambda x: tanh(x)
fprime_x = lambda x: 1-(tanh(x))**2
print("Newton method for tanh(x) at x0 = 1.08")
newton(fx,fprime_x,1.08,1e-6)
print("\n")
print("Newton method for tanh(x) at x0 = 1.09")
newton(fx,fprime_x,1.09,1e-6) 



#********************
# Secant Method *****
#********************

from math import *
def secant(x0, x1, f, TOL, Nmax):
    f_x0 = f(x0)
    f_x1 = f(x1)
    itc = 0
    dx = 0.0
    while abs(f_x1) > TOL and itc < Nmax:
        try:
            dx = (f_x1-f_x0)/(x1-x0)
            x = x1 - f_x1/dx
        except ValueError as e:
            print (e)  
        x0 = x1
        x1 = x
        f_x0 = f_x1
        f_x1 = f(x1)
        itc += 1
        print("Step %d: x = %.5e, f(x) = %.5e" % (itc,x,f_x1))
    return f_x1
    y = x1
    if abs(f_x1) > TOL:
        itc -= 1
fx = lambda x: tanh(x)
print("Secant Method for the interval [1.08, 1.09]")
secant(1.08,1.09,fx,1e-6,8)
print("\nSecant Method for the interval [1.09, 1.1]")
secant(1.09,1.10,fx,1e-6,8)  
print("\nSecant Method for the interval [1, 2.3]")
secant(1,2.3,fx,1e-6,8) 
print("\nSecant Method for the interval [1, 2.4]")
secant(1,2.4,fx,1e-6,8)  



#********************
# Bisection Method **
#********************


from math import *
import numpy as np
def bisect(a, b, f, delta):
    fa = f(a)
    fb = f(b)
    if  np.sign(fa) == np.sign(fb):
        raise ValueError('f must have different signs at the endpoints a and b. Aborted.')
    else:
        print("Step 1: a = %.2e, b=%.5e, fa=%.5e, fb=%.5e" % (a,b,fa,fb))   
    i = 1
    while(abs(b-a) > 2*delta):
        c = (b + a)/2
        fc = f(c)
        i += 1
        if np.sign(fc) != np.sign(fb):
            a = c
            fa = fc
        else:
            b = c
            fb = fc
        print("Step %d: a = %.2e, b = %.5e, fa = %.5e, fb=%.5e" % (i,a,b,fa,fb))
    return c
fx = lambda x: tanh(x)
bisect(-5, 3, fx, 1e-6)

