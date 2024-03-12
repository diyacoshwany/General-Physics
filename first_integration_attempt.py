# -*- coding: utf-8 -*-
"""
The following code solves the first set of equations which describes a cooling
cup of coffee. I will use the RK4 method to numerically integrate the coupled
differential equationsand plot a temperature against time graph of the coffee
and cup
"""

import numpy as np
import matplotlib.pyplot as plt

COFFEE_AREA = np.pi * 0.05**2
CUP_AREA = 2*0.05*np.pi*0.10 + COFFEE_AREA
INITIAL_COFFEE_TEMP = 95 + 273.15
INITIAL_CUP_TEMP = 298.15 + 60
ROOM_TEMP = 298.15
COFFEE_HEAT_CAP = 4200 * 0.2
CUP_HEAT_CAP = 900 * 0.5
COFFEE_EMMI = 0.97
CUP_EMMI = 0.9
ROOM_EMMI = 1
K_COFFEE = 0.598
K_CUP = 3.8
STEF_BOLTZ = 5.67 * 10 ** -8 
COFFEE_HEAT_TRANSFER = 500 
CUP_HEAT_TRANSFER = 100 
LENGTH = CUP_AREA
TIME_STEP = 0.001
ITERATIONS = int(100000*1.5)

def convective_law(contact_area, heat_transfer_co, surround_temp, obj_temp):
    return contact_area * heat_transfer_co * (obj_temp - surround_temp)

def conductive_law(heat_conduction, length_scale, surround_temp, obj_temp):
    return -heat_conduction * length_scale * (surround_temp - obj_temp)

def radiative_law(contact_area, STEF_BOLTZ, emmi_obj, emmi_surr, obj_temp,
                  surr_temp):
    return contact_area * STEF_BOLTZ * (emmi_obj * obj_temp**4 - emmi_surr
                                        * surr_temp**4)

def coffee_equation(coffee_temp, cup_temp):
    
    temp_change = -(1/COFFEE_HEAT_CAP)*(convective_law(COFFEE_AREA, COFFEE_HEAT_TRANSFER,
                                    ROOM_TEMP, coffee_temp) +
                      conductive_law(K_COFFEE,
                                     LENGTH,
                                     cup_temp, coffee_temp)
                        + radiative_law(COFFEE_AREA, STEF_BOLTZ,
                                        COFFEE_EMMI, CUP_EMMI, coffee_temp, ROOM_TEMP))
    
    return temp_change

def cup_equation(coffee_temp, cup_temp):
    temp_change = -(1/CUP_HEAT_CAP)*(convective_law(CUP_AREA, CUP_HEAT_TRANSFER,
                                    ROOM_TEMP, cup_temp) -
                      conductive_law(K_CUP,
                                     LENGTH,
                                     cup_temp, coffee_temp)
                        + radiative_law(CUP_AREA, STEF_BOLTZ,
                                        CUP_EMMI, ROOM_EMMI, cup_temp, ROOM_TEMP))
    return temp_change

def k1_coffee(coffee_temp, cup_temp):
    
    return coffee_equation(coffee_temp, cup_temp)

def k1_cup(coffee_temp, cup_temp):
    
    return cup_equation(coffee_temp, cup_temp)

def k2_coffee(coffee_temp, cup_temp, k1_cof, k1_cu):
    
    return coffee_equation(coffee_temp + TIME_STEP*0.5*k1_cof,
                           cup_temp + TIME_STEP * 0.5 * k1_cu)

def k2_cup(coffee_temp, cup_temp, k1_cof, k1_cu):
    
    return cup_equation(coffee_temp + TIME_STEP*0.5*k1_cof,
                        cup_temp + TIME_STEP*0.5*k1_cu)
def k3_coffee(coffee_temp, cup_temp, k2_cof, k2_cu):
    
    return coffee_equation(coffee_temp + TIME_STEP*0.5*k2_cof,
                           cup_temp + TIME_STEP * 0.5 * k2_cu)

def k3_cup(coffee_temp, cup_temp, k2_cof, k2_cu):
    
    return cup_equation(coffee_temp + TIME_STEP*0.5 * k2_cof,
                           cup_temp + TIME_STEP * 0.5 * k2_cu)
def k4_coffee(coffee_temp, cup_temp, k3_cof, k3_cu):
    
    return coffee_equation(coffee_temp + TIME_STEP*k3_cof,
                           cup_temp + TIME_STEP *  k3_cu)

def k4_cup(coffee_temp, cup_temp, k3_cof, k3_cu):
    
    return cup_equation(coffee_temp + TIME_STEP*k3_cof,
                           cup_temp + TIME_STEP * k3_cu)

def RK4(temp_n, k1, k2, k3, k4, TIME_STEP):
    
    return temp_n + TIME_STEP*(k1 + 2*k2 + 2*k3 + k4)


def main():
    coffee_temp = INITIAL_COFFEE_TEMP
    cup_temp = INITIAL_CUP_TEMP
    k1_array = np.zeros((ITERATIONS, 2))
    k2_array = np.zeros((ITERATIONS, 2))
    k3_array = np.zeros((ITERATIONS, 2))
    k4_array = np.zeros((ITERATIONS, 2))
    temp_array = np.zeros((ITERATIONS + 1, 2))
    time_array = np.zeros(ITERATIONS + 1)
    temp_array[0, 0] = coffee_temp
    temp_array[0, 1] = cup_temp
    print(temp_array[0, 0])
    
    for i in range(ITERATIONS):
        k1_array[i, 0] = k1_coffee(temp_array[i, 0], temp_array[i, 1])
        k1_array[i, 1] = k1_cup(temp_array[i, 0], temp_array[i, 1])
        k2_array[i, 0] = k2_coffee(temp_array[i, 0], temp_array[i, 1], k1_array[i, 0], k1_array[i, 1])
        k2_array[i, 1] = k2_cup(temp_array[i, 0], temp_array[i, 1], k1_array[i, 0], k1_array[i, 1])
        k3_array[i, 0] = k3_coffee(temp_array[i, 0], temp_array[i, 1], k2_array[i, 0], k2_array[i, 1])
        k3_array[i, 1] = k3_cup(temp_array[i, 0], temp_array[i, 1], k2_array[i, 0], k2_array[i, 1])
        k4_array[i, 0] = k4_coffee(temp_array[i, 0], temp_array[i, 1], k3_array[i, 0], k2_array[i, 1])
        k4_array[i, 1] = k4_cup(temp_array[i, 0], temp_array[i, 1], k3_array[i, 0], k3_array[i, 1])
        temp_array[i + 1, 0] = RK4(temp_array[i, 0], k1_array[i, 0], k2_array[i, 0],
                                      k3_array[i, 0], k4_array[i, 0], TIME_STEP)
        temp_array[i + 1, 1] = RK4(temp_array[i, 1], k1_array[i, 1], k2_array[i, 1],
                                      k3_array[i, 1], k4_array[i, 1], TIME_STEP)        
        time_array[i + 1] = TIME_STEP * (i + 1)
        
    print(temp_array[ITERATIONS - 10:ITERATIONS - 1, 0])
    print(temp_array[ITERATIONS - 10:ITERATIONS - 1, 1])
        
    plt.xlabel("time/s")
    plt.ylabel("temperature/K")
    plt.plot(time_array, temp_array[:,0], label = "coffee")
    plt.plot(time_array, temp_array[:,1], label = "cup")
    plt.legend()
    plt.show()
    return 0

main()