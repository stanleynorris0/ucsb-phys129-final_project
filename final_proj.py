#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 26 12:06:07 2020

@author: stanleynorris
"""

# import relevant libraries and modules
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
import math
import numpy as np
import matplotlib.pyplot as plt
import requests
import scipy
from scipy.integrate import odeint
import numpy.random as random
import matplotlib.animation as anim
from matplotlib.pyplot import figure
import time
import sys
import os
import csv
import requests
# clear terminal
os.system('cls||clear')
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
# program starts with intro message 
print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
print("WELCOME TO THE CORONAVIRUS EPIDEMIC SIMULATOR")
print("This program will simulate the spread of the novel COVID-19 virus")
print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
print("")
# import and organise the data
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
# import the data from GBplaces.csv file which contains name, type, population, latitude and longitude of 100 most populated cities and towns in Great Britain
# there is a header to the file so the data starts from the 2nd row
# the csv file is found by accessing a github repository created specifically for this final project
csv_url = 'https://raw.githubusercontent.com/stanleynorris0/ucsb-phys129-final_project/master/GBplaces.csv'
img_url = 'https://raw.githubusercontent.com/stanleynorris0/ucsb-phys129-final_project/master/GBmap.png' 
organised_data = []
header = 0
print("Fetching data...")
with requests.Session() as s:
    download = s.get(csv_url)
    decoded_content = download.content.decode('utf-8')
    cr = csv.reader(decoded_content.splitlines(), delimiter=',')
    my_list = list(cr)
    for row in my_list:
        if (header > 0):
            temp_array = [ float(row[2]), float(row[3]), float(row[4]) ]
            organised_data.append(temp_array)
        header += 1 
print("Data succesfully retrieved:")
print("    ->GBPlaces.csv: file with information on 100 most populous cities in Great Britain")
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
# now the organised_data array contains the information required to begin the simulation
# each element of allS, allI and allR will contain an array representing results of a simulation in a given city/town
# each array in allS represents the number of susceptible people over time for a given city, and the same applies for allI and allR except that I represents the number of infected people and R the number of receovered people
# AllCoords will take the data representing the coordinates of the places, in the form of (latitude, longitude)
allS = []
allI = []
allR = []
AllCoords = []
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#


# run SIR simulation using data
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
    # run the classic SIR model on all 100 places, place the solutions to the differential equations in allS, allI and allR
print("")
print("The differential equations which model coronavirus are characterised by the infection and recovery rate.")
print("The infection rate determines how fast subjects in a population go from Susceptible to Infected")
print("The recovery rate determines how fast subjects in a population go from Infected to Recovered")
print("The categories are:")
print("S (Susceptible)")
print("I (Infected)")
print("R (Recovered)")
    # Enable interactivity to set initial parameters of differential equations
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

while True:
    try:
        print("You may select the infection and recovery rates:")
        print("The values must both be between 0 and 1")
        print("beta must be greater than gamma")
        print("*if not the default values will be set*")
        print("|Default is:         |")
        print("|beta [1/days] = 0.3 |")
        print("|gamma [1/days] = 0.1|")
        beta = float(input("Enter the rate of infection, beta, to be used in model: "))
        gamma = float(input("Enter the rate of recovery, gamma, to be used in model: "))
        break
    except ValueError:
        print("Oops! That was not a valid input.")
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
    # beta is the rate of infection, [1/days]
    # gamma is the rate of recovery, [1/days]
if beta < gamma:
    beta = 0.3
    gamma = 0.1
if beta < 0 or beta > 1:
    beta = 0.3
if gamma < 0 or gamma > 1:
    gamma = 0.1
    # display the values used in simulation
print("beta = " + str(beta))
print("gamma = " + str(gamma))
print("")

for j in range(0,100):
    # population of city_j is N.
    this_sim_data = organised_data[j] 
    N = this_sim_data[0]
    AllCoords.append((this_sim_data[1],this_sim_data[2]))
    
    # initialise the number of infected and recovered people
    # setting I_0 to zero is an assumption to be discussed in the paper
    I_0, R_0 = 1, 0
    # if an individual is not infected or recovered, then they are susceptible
    S_0 = N - I_0 - R_0
    # t is the time over which the simulation is to be ran, [days]
    t = np.linspace(0, 250, 250)

    # the SIR differential equations are returned by this function
    def diffs(y, t, N, beta, gamma):
        S, I, R = y
        dSdt = -beta * S * I / N
        dIdt = beta * S * I / N - gamma * I
        dRdt = gamma * I
        return dSdt, dIdt, dRdt
    # initial conditions to be 1st argument to diff eqns
    y_0 = S_0, I_0, R_0
    # integrate the SIR derivatives over time interval t
    ret = odeint(diffs, y_0, t, args=(N, beta, gamma))
    S, I, R = ret.T
    allS.append(S)
    allI.append(I)
    allR.append(R)
print("Differential equations solved!")
print("")
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#


# determine nature of peak
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
# data tracking the spread of covid-19 for every region, over 250 days, is done
# need to use this data to calculate the average peak in Great Britain
# this peak is categorized by the assumption that the time at which the most equipment is needed is when the total number of infected people on the island is the highest
global I_max
I_max = 0
global which_array
which_array = 0
for i in range(0,200):
    I_tot = 0
    for j in range(0,100):
        I_tot += allI[j][i]
    if I_tot > I_max:
        I_max = I_tot
        which_array = i
# output a message indicating at which day disease is at its worst
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
      
      
# function which calculates the distance (as though travelled over the surface of Earth) between coordinates c1 and c2, taking into account curvature of Earth
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
def dist_calc(c1, c2):
    R = 6372800  # Earth radius in meters
    lat1, lon1 = c1
    lat2, lon2 = c2
    A1, A2 = math.radians(lat1), math.radians(lat2) 
    dA = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dA/2)**2 + math.cos(A1)*math.cos(A2)*math.sin(dlambda/2)**2
    return 2*R*math.atan2(math.sqrt(a), math.sqrt(1 - a))
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#


#function which displays large integers with commas separating 3 orders mag.
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
def dec_display(number):
    s = '%d' % number
    groups = []
    while s and s[-1].isdigit():
        groups.append(s[-3:])
        s = s[:-3]
    return s + ','.join(reversed(groups))
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#


# output a message indicating at which day disease is at its worst
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
print("The peak of the disease will be reached on day " + str(which_array) + ", at which point " + str(dec_display(int(round(I_max)))) + " people in Great Britain will be infected")
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

# optimization part, determines most efficient place to build a factory to manufacture and ship hand sanitizer during an epidemic
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
# initialise parameters for determination of maxima/minima
R = 3958.75
countahh = 0
dx=0
dy=0
glx=0
gly=0
step = 0.001
globalMax = 0
wSum = 0
newwSum = 0
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

# user gets to choose between 4 major ports in the UK from which to receive the ethanol required in the manufacturing of the hand sanitizer
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
Immingham_docks = (53.631323, -0.193303)
London_docks = (51.443588, 0.395973)
Southampton_docks = (50.901540, -1.417160)
Liverpool_docks = (53.398179, -2.993227)
ports = (Immingham_docks, London_docks, Southampton_docks, Liverpool_docks)

# begin selection
print("~~~~~~~~~~~~~~~~~~~~")
print("Options:")
print("1. Immingham docks")
print("2. London docks")
print("3. Southampton docks")
print("4. Liverpool docks")
print("~~~~~~~~~~~~~~~~~~~~")
print("A hand sanitizer manufacturing factory must be built in Great Britain, to ship the necessary product to the people of the island")
print("The program will calculate the most optimised location of the factory based on the data imported")
print("There is a choice of port to which the ethanol, used in the synthesis of hand sanitizer, must be shipped to")
response = None
while response not in {"1", "2", "3", "4"}:
    response = input("Please choose a port to partner with from options 1-4 above by entering {1,2,3,4}: ")
for g in range(1,5):
    elm = int(response)
    if g == elm:
        port_loc = ports[g-1]
# port_loc has the coordinates of the chosen port

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

#function that creates a random location [degrees] between upper and lower, converts it to radians and returns that value
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
def randum_numbah(upper,lower):
    d = (upper-lower) * np.random.random() + lower   
    q = d * np.pi/180
    return q
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
 
# begin simulation
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

# find data corresponding to the number of infected people in each place at the time of the peak
# conv_data will hold this information
conv_data = []
for i in range(0,100):
    global whicharray
    GPEAK_INFECTIONS = allI[i][which_array]
    c_one = organised_data[i][1]*(np.pi/180)
    c_two = organised_data[i][2]*(np.pi/180)
    temp_data = [GPEAK_INFECTIONS, c_one, c_two]
    conv_data.append(temp_data)

north = (60, -1.5)
south = (50, -1.5)
# for every j a new iteration will begin in which a random point is created in the region of Great Britain, at which point the weighting function is evaluated
# the weighting function is one that essentially takes into account the number of infected people in each city/town and the distance of those from the random point generated
# the most efficient place (with linear weighting, more discussion in report) will be somewhere as close as possible to as many populated places as possible
# this is a very simplistic version of what really happens, and where the best place is to put the factory
print("Commencing calculation of factory location, please wait...")
for i in range(0,200):
    if i == 50:
        print("25% complete...")
    if i == 100:
        print("50% complete...")
    if i == 150:
        print("75% complete...")
    if i == 190:
        print("finishing up...")
    latit = randum_numbah(50,54)
    longit = randum_numbah(-0.5,2)
    # coor is the random location
    coor = (latit, longit)
    wwSum = 0

    # these calculations and comparisons will evaluate the value of the weighting function and try to minimise it by maximise its inverse and only saving new values if they are greater than the previous ones
    # the loops in k,l,m calculate the value of the weighting function at areas surrounding the random location generated, to search for local maxima in the inverse of the weighting fucntion
    for j in range(0,100):
        coor_prime = (conv_data[j][1], conv_data[j][2])
        value = dist_calc(coor, coor_prime) * conv_data[j][0]
        wwSum += value
    wwSum = wwSum + dist_calc(coor, port_loc) * 1000000
    wSum = 1/wwSum
    oldwSum = 0
    while (wSum>oldwSum):
        oldwSum = wSum
        for k in range(-1,2):
             for l in range(-1,2):
                 if (k==l):
                     pass
                 else:
                     newwSum = 0
                     for m in range(0,100):
                         cwor = (latit + step*k, longit + step*l)
                         cwor2 = (conv_data[m][1], conv_data[m][2])
                         newValue = dist_calc(cwor, cwor2)*conv_data[m][0]
                         newwSum += newValue
                     newwSum = newwSum + dist_calc(cwor, port_loc) * 1000000
                     newWSum = 1/newwSum
                     if (newWSum >= wSum):
                         dx = k
                         dy = l
                         wSum = newWSum
        latit += step*dx
        longit += step*dy
    if (wSum > globalMax):
        globalMax = wSum
        glx = latit*180/(np.pi)
        gly = longit*180/(np.pi)

print("The coordinates of the manufacturing factory are given by: ")
print("(", glx,", ",gly,")")
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#


# animation for user to visualise how disease spreads across region
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
extrems = (-7.94, 1.80, 49.9, 60.9)
x1 = np.zeros(100)
y1 = np.zeros(100)
for i in range(0,100):
    x1[i] = AllCoords[i][1]
    y1[i] = AllCoords[i][0]
global counter
counter = 0
def plot_gr(xmax):
    fig, ax = plt.subplots()
    ax.set_ylim(extrems[2],extrems[3])
    ax.set_xlim(extrems[0],extrems[1])
    def update(i):
        global counter
        ax.clear()
        temp = []
        for i in range(0,100):
            temp.append(0.003*allI[i][counter])
        ax.scatter(x1, y1, zorder=1, alpha = 0.6, c='r', s = temp)
        fac = plt.scatter(gly, glx, zorder=1, alpha = 1, c='g', s = 10)
        ax.annotate("Factory Location",
                    xy=(gly, glx), xycoords='data',
                    xytext=(0.9, 0.90), textcoords='axes fraction',
                    arrowprops=dict(facecolor='black', shrink=0.02, width = 1),
                    horizontalalignment='right', verticalalignment='top')
                    #ax.annotate("Factory Location", (gly, glx), fontsize = 10)
        ax.legend([fac], [str(counter+1) + " days since start of epidemic"])
        ax.imshow(im, zorder = 0, extent = extrems, aspect = 'equal')
        counter += 1
        if counter == 250:
            input("End of simulation: Press Enter")
            exit()
    
    a = anim.FuncAnimation(fig, update, frames = xmax, repeat=False)
    plt.show()


# import background of plot, wihch is a map of Great Britain created with Basemap on python2.7
# image file is found on github repository specifically made for this final project
im = plt.imread(img_url)
# begin animation
print("Animation starting...")
plot_gr(len(allI[0]))
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#


#plot showing the solutions to all differential equations, useful for debugging
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#for k in range (0,100):
# Plot the data on three separate curves for S(t), I(t) and R(t)
 #  plt.plot(t, allS[k]/200000, 'b', alpha=0.5, lw=2, label='Susceptible')
  # plt.plot(t, allI[k]/200000, 'r', alpha=0.5, lw=2, label='Infected')
   #plt.plot(t, allR[k]/200000, 'g', alpha=0.5, lw=2, label='Recovered with immunity')

#plt.show()
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
