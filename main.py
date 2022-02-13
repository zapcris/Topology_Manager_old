# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import array as arr
import numpy as np
import math
from sys import stdout
from typing import List, Any
from itertools import combinations
from random import sample
from mip import Model, xsum, BINARY, INTEGER
import pandas as pd


class workstation:
    def __init__(self, name, x, y, a):
        self.x = x
        self.y = y
        self.name = name
        self.active = a

    def __repr__(self):
        return f'("{self.name}",{self.x},{self.y}, {self.active})'


# Instantiate workstations
w1 = workstation('workstation1', 0, 0, True)
w2 = workstation('workstation2', 0, 0, True)
w3 = workstation('workstation3', 0, 0, True)
w4 = workstation('workstation4', 0, 0, False)
w5 = workstation('workstation5', 0, 0, True)
w6 = workstation('workstation6', 0, 0, True)
w7 = workstation('workstation7', 0, 0, False)
w8 = workstation('workstation8', 0, 0, True)
w9 = workstation('workstation9', 0, 0, True)
w10 = workstation('workstation10', 0, 0, False)
w11 = workstation('workstation11', 0, 0, True)
w12 = workstation('workstation12', 0, 0, False)
w13 = workstation('workstation13', 0, 0, True)
w14 = workstation('workstation14', 0, 0, False)
w15 = workstation('workstation15', 9, 9, True)

Workstations2 = [w1, w2, w3, w4, w5, w6, w7, w8, w9, w10, w11, w12, w13, w14, w15]
print(Workstations2)
Total_w_production = 0
Total_cost = 0
active_topology: list[workstation | Any] = []
base_topology = []
Total_topology = []
ran_pos = []
ran_topologies = []
ran_topologies.clear()
# for i in Workstations :
#   print (Workstations[i],end = "")
print(active_topology)
#  print(i)
w1.x, w1.y = (0, 0)
w15.x, w15.y = (9, 9)
rows, cols = (10, 10)

Grid = [[0 for i in range(cols)] for j in range(rows)]  # initiate a grid

Grid[0][0] = w1.name
Grid[9][9] = w15.name

for rows in Grid:
    print(rows)


def distance(x1, y1, x2, y2):
    return math.sqrt(math.pow(x2 - x1, 2) +
                     math.pow(y2 - y1, 2) * 1.0)


for x in range(len(Workstations2)):

    if Workstations2[x].active:
        Total_w_production = Total_w_production + 1
        active_topology.append(Workstations2[x])
        base_topology.append(Workstations2[x])
        # print(Workstations2[x].name)

print("Total workstations in production :", Total_w_production)  # total active workstations
print("Active topology :", active_topology)
print("%.0f" % distance(w1.x, w1.y, w15.x, w15.y))

# for x in range(len(active_topology)):
#    print(active_topology[x].x)

d = []
# def total_cost():

# d1 = "%.0f"%distance(active_topology[0].x, active_topology[0].y,active_topology[1].x, active_topology[1].y)
# d2 = "%.0f"%distance(active_topology[1].x, active_topology[1].y,active_topology[2].x, active_topology[2].y)

cost_length = len(active_topology) - 1

# print(cost_length)
# Calculate total cost of the topology

for x in range(cost_length):
    d1 = distance(active_topology[x].x, active_topology[x].y, active_topology[x + 1].x, active_topology[x + 1].y)
    d.append(round(d1))

print("Total cost of production of base topology:", sum(d))  # total cost of production

n = Total_w_production
print("total active workstations", n)
Map = Model()

z = [Map.add_var(var_type=INTEGER) for i in range(n)]  # cost function based on euclidean travel distance

print(sum(z))

# Randomize placement of workstations

for i in range(n):

    if (i & 1) == 1 and i != 0 and i != n - 1:
        active_topology[i].x = 6
        active_topology[i].y = active_topology[i - 1].y + 0

    elif (i & 1) != 1 and i != 0 and i != n - 1:
        active_topology[i].x = active_topology[i - 1].x
        active_topology[i].y = active_topology[i - 1].y + 2

for x in range(len(active_topology)):
    print("position of workstation", active_topology[x].name, active_topology[x].x, active_topology[x].y)

for x in range(cost_length):
    d1 = distance(active_topology[x].x, active_topology[x].y, active_topology[x + 1].x, active_topology[x + 1].y)
    d.append(round(d1))
    print(round(d1))
print("Total cost of production_1st topology:", sum(d))  # total cost of production

Total_topology.append(sum(d))

print(Total_topology)


# Function to grid placement and travel cost estimation

def Grid_placing(current_topology, x_offset, y_offset):
    c1 = []
    r = len(current_topology) - 1
    for i in range(0, len(current_topology)):
        if i != 0 and i != n - 1:
            current_topology[i].x = x_offset[i]
            current_topology[i].y = y_offset[i]
            # print(current_topology[i].x, current_topology[i].y)
        # elif (i & 1) != 1 and i != 0 and i != len(current_topology):
        #    current_topology[i].x = 0
        #   current_topology[i].y = current_topology[i - 1].y + y_offset

    return current_topology


def cal_cost(input_topology, ran):  # Function to calculate cumulative travel cost in topology
    total_dist = [[0]]
    cum = []
    for i in range(ran):
        for j in range(n - 1):
            # dist = distance(input_topology[x].x, input_topology[x].y, input_topology[x + 1].x, input_topology[x + 1].y)
            total_dist[i][j] = math.sqrt(math.pow(input_topology[i][j + 1].x - input_topology[i][j].x, 2) + math.pow(
                input_topology[i][j + 1].y - input_topology[i][j].y, 2) * 1.0)
            print(total_dist[i][j])

    return cum


# T2 = Grid_placing(base_topology, 6, 2)

# for x in range(len(T2)):
#    print(T2[x].x,T2[x].y)

ran_sample = 3
for i in range(ran_sample):  # generate random position values
    ran_pos.append(sample(list(combinations(range(0, n), 2)), 10))

print("Random positions:", ran_pos)
print("Total random topologies:", len(ran_pos))
print(ran_pos[0][0][0],ran_pos[0][1][0],ran_pos[0][2][0],ran_pos[0][3][0],ran_pos[0][4][0],ran_pos[0][5][0])
print(ran_pos[1][0][0],ran_pos[1][1][0],ran_pos[1][2][0],ran_pos[1][3][0],ran_pos[1][4][0],ran_pos[1][5][0])
# for i in range(len(ran_pos)):  # generate topologies based on random positions
# for i in range(n):
#   ran_topologies.append(Grid_placing(base_topology, ran_pos[i][i][0], ran_pos[i][i][1]))

# for i in range(ran_sample):
for i in range(ran_sample):  # create datastructure for Random/collective topologies
    ran_topologies.append(base_topology)

print("length of empty random topologies:", len(ran_topologies))

print(ran_topologies)


# for i in range(0, len(ran_topologies)):  # populate random topologies with random positions

#    for j in range(0, len(ran_topologies[i])):

#      if j != 0 and j != n - 1:
#        ran_topologies[i][j].x = ran_pos[i][j][0]
#        ran_topologies[i][j].y = ran_pos[i][j][1]

#def fill_ranpos(topology, a):
for j in range(0, len(ran_topologies[0])):
    if j != 0 and j != n - 1:
        ran_topologies[0][j].x = ran_pos[0][j][0]
        ran_topologies[0][j].y = ran_pos[0][j][1]
        print("random_position index 0:", ran_pos[0][j][0],ran_pos[0][j][1])
    # print("Workstations random position and name", ran_topologies[i][j].x,
    #     ran_topologies[i][j].y, ran_pos[i][j][0], ran_pos[i][j][1], ran_topologies[i][j].name)
random_topology = []
#for j in range(0, len(ran_topologies[1])):
  #  ran_topo = []
   # if j != 0 and j != n - 1:
    #    ran_topo[j] = ran_pos[0][j][0]
     #   ran_topo[j] = ran_pos[0][j][1]
      #  print("NEw random topo:", ran_topo[1][j][0], ran_topo[1][j][1])
print(ran_topologies[0][0].x, ran_topologies[0][0].y, ran_topologies[0][1].x, ran_topologies[0][1].y)
print(ran_topologies)
# print(enumerate(ran_topologies)
print(len(ran_topologies))
for i in enumerate(ran_topologies[i]):
    print(i)
# print(len(ran_topologies[i])-1)
# def travel_cost(a):
arr_cost = []
print(ran_topologies[0])
print(ran_topologies[1])
for i in range(len(ran_topologies)):
    total_cost = []
    for j in range(len(ran_topologies[i]) - 1):
        cost = (math.sqrt(math.pow(ran_topologies[i][j + 1].x - ran_topologies[i][j].x, 2) +
                          math.pow(ran_topologies[i][j + 1].y - ran_topologies[i][j].y, 2) * 1.0))
        total_cost.append(cost)
    arr_cost.append(sum(total_cost))
print(arr_cost)
    # return total_cost, sum(total_cost)
