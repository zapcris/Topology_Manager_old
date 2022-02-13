# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

from email.mime import base
import math
from msilib.schema import Class
from multiprocessing import allow_connection_pickling
from sys import stdout
from random import sample
import random
from itertools import combinations

import numpy as np
import pandas as pd
from mip import Model, xsum, BINARY, INTEGER
from dataclasses import dataclass

random.seed(1314141)

# @dataclass
# class work_station_instance:
#     num: int
#     x: float = 0
#     y: float = 0
#     active: bool = False

# ws1 = work_station_instance(num=1, x=500)

# def make_ws_array()


# class topology():
#     def __init__(self, work_stations=10) -> None:
#         self.workstations = []
#         for index, i in enumerate(range(work_stations)):
#             self.workstations.append(work_station_instance(num=index+1, active=ran))
    
#     def print_work_stations(self):
#         for ws in self.workstations:
#             print(ws)

#     def turn_off_ws(self, idx):
#         self.workstations[idx].active = False

#     def edit_pos(self, idx, x, y):
#         self.workstations[idx].x = x
#         self.workstations[idx].y = y


# print("At FIRST")
# tplg = topology(15)
# tplg.print_work_stations()

# print("AND NOW")
# tplg.turn_off_ws(0)
# tplg.print_work_stations()

#-------- FROM HERE ------------------------------------------------------------#
# External parameters:
# activity = [random.choice([True, False]) for x in range(15)]
activity = [True, True, False, False, True, True, False]
max_topologies = 5

sequence = [3,2,1,5,28,12,16,9]

@dataclass
class workstation:
    num: int
    active: bool

@dataclass
class config:
    x: float
    y: float

all_workstations = []
# making workstations
# for num_ws, active in zip(range(len(activity)), activity):
#     all_workstations.append(workstation(num=num_ws+1, active=active))

for num_ws in sequence:
    all_workstations.append(workstation(num=num_ws, active=True))

# base_station means base topology
base_stations = [ws for ws in all_workstations if ws.active]
for bs in base_stations:
    print(bs)


class topology():
    def __init__(self, ws_list, config_list):
        self.ws_list = ws_list
        self.configs = config_list
    
    def display(self):
        for ws, cfg in zip(self.ws_list, self.configs):
            print(f"Workstation number: {ws.num} Coordinates x={cfg.x} y={cfg.y} active={ws.active}")

    def calculate_distance(self):
        total_ws = len(self.ws_list) # 5
        dist = 0.0
        for i in range(total_ws-1):
            dist +=  math.sqrt(math.pow(self.configs[i+1].x - self.configs[i].x, 2) + math.pow(
                self.configs[i+1].y - self.configs[i].y, 2) * 1.0)
        return dist




all_topologies, config_list = [], []

for num in range(max_topologies):
    config_list.append([config(random.randint(0,9), random.randint(0,9)) for i in base_stations])

for num in range(max_topologies):
    all_topologies.append(topology(base_stations, config_list[num]))

for top in all_topologies:
    top.display()
    print(top.calculate_distance())

import sys
sys.exit()


#-------- TO HERE ------------------------------------------------------------#





class workstation:
    def __init__(self, name, x, y, a):
        self.x = x
        self.y = y
        self.name = name
        self.active = a

    def __repr__(self):
        return f'("{self.name}",{self.x},{self.y}, {self.active})'


def distance(x1, y1, x2, y2):
    return math.sqrt(math.pow(x2 - x1, 2) +
                     math.pow(y2 - y1, 2) * 1.0)

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

Total_w_production = 0
Total_cost = 0
active_topology = []
base_topology = []
Total_topology = []
ran_pos = []
ran_topologies = []
ran_topologies.clear()

w1.x, w1.y = (0, 0)
w15.x, w15.y = (9, 9)
rows, cols = (10, 10)

# initiate a grid
Grid = [[0 for i in range(cols)] for j in range(rows)]
Grid[0][0] = w1.name
Grid[9][9] = w15.name

# Check
# for rows in Grid:
    # print(rows)

for x in range(len(Workstations2)):
    if Workstations2[x].active:
        Total_w_production = Total_w_production + 1
        active_topology.append(Workstations2[x])
        base_topology.append(Workstations2[x])

# Logging
print("Total workstations in production :", Total_w_production)
print("Active topology :", active_topology)
print("%.0f" % distance(w1.x, w1.y, w15.x, w15.y))

d = []
cost_length = len(active_topology) - 1

# Calculate total cost of the topology
for x in range(cost_length):
    d1 = distance(active_topology[x].x, active_topology[x].y, active_topology[x + 1].x, active_topology[x + 1].y)
    d.append(round(d1))

print("Total cost of production of base topology:", sum(d))  # total cost of production

n = Total_w_production
print("total active workstations", n)

# Randomize placement of workstations

for x in range(len(active_topology)):
    print("position of workstation", active_topology[x].name, active_topology[x].x, active_topology[x].y)

for x in range(cost_length):
    d1 = distance(active_topology[x].x, active_topology[x].y, active_topology[x + 1].x, active_topology[x + 1].y)
    d.append(round(d1))
    print(round(d1))
print("Total cost of production_1st topology:", sum(d))  # total cost of production

Total_topology.append(sum(d))

print(Total_topology)

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
    ran_topologies[0][j].x = ran_pos[0][j][0]
    ran_topologies[0][j].y = ran_pos[0][j][1]
    print("random_position index 0:", ran_pos[0][j][0],ran_pos[0][j][1])

for tplg in ran_topologies[0]:
    print(tplg)

for j in range(0, len(ran_topologies[1])):
    ran_topologies[1][j].x = ran_pos[1][j][0]
    ran_topologies[1][j].y = ran_pos[1][j][1]
    print("random_position index 1:", ran_pos[1][j][0], ran_pos[1][j][1])

    
print(ran_topologies[0][0].x, ran_topologies[0][0].y, ran_topologies[0][1].x, ran_topologies[0][1].y)
print(ran_topologies)
# print(enumerate(ran_topologies)
print(len(ran_topologies))
for i in enumerate(ran_topologies[i]):
    print(i)
# print(len(ran_topologies[i])-1)
# def travel_cost(a):

# for tplg in ran_topologies[0]:
#     print(tplg)
for tplg in ran_topologies[1]:
    print(tplg)

# print(np.array(ran_pos[0]).shape)
# print(np.array(ran_pos[1]).shape)
# print(np.array(ran_pos[0]) == np.array(ran_pos[1]))
print(np.array(ran_pos[0]))
print(np.array(ran_pos[1]))


        