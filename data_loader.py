from decimal import setcontext
import numpy as np
import json
import math
from haversine import haversine

data_dict = {
    "jun": 
        {
            "data_num": 17, 
            "data_name": "jun", 
            "trains":400
        }
    }

def gps_distance(lat1, lng1, lat2, lng2):
    return haversine((lat1, lng1), (lat2, lng2), unit='m')

def discretize_gps(lat1, lng1, lat2, lng2):
    return int(gps_distance(lat1, lng1, lat2, lng2) / 50)

def get_data(line, idx, data_name):
    f = open("data.txt".format(data_name), 'r')
    done = False
    data = f.readline()
    for i in range(line):
        data = f.readline()

    dt = json.loads(data)

    if dt['time_gap'][-1] == dt['time_gap'][idx-1]:
        done = True
        next_t = dt['time_gap'][-1]
    else:
        next_t = dt['time_gap'][idx]

    last_gps = [dt['lats'][-1], dt['lngs'][-1]]
    init_gps = [dt['lats'][0], dt['lngs'][0]]

    min_lng = 126.75
    min_lat = 37.53

    lngs = dt["lngs"]
    dt['lngs'] = lngs[:idx]

    lats = dt["lats"]
    dt['lats'] = lats[:idx]

    time_gap = dt["time_gap"]
    dt['time_gap'] = time_gap[:idx]


    lng = discretize_gps(dt['lats'][-1], min_lng, dt['lats'][-1], dt['lngs'][-1]) / 330
    lat = discretize_gps(min_lat, dt['lngs'][-1], dt['lats'][-1], dt['lngs'][-1]) / 400
    
    lng = round(lng, 2)
    lat = round(lat, 2)
    
    remaining_time = (dt['time'] - dt['time_gap'][-1]) / dt['time']
    current_t = (dt['timeID']*60 + dt['time_gap'][-1]) / (3600*24)

    remaining_dist = 0
    if done:
        remaining_dist = 0
    else:
        remaining_dist = gps_distance(dt['lats'][-1], dt['lngs'][-1], last_gps[0], last_gps[1])/4700

    return current_t, next_t, lat, lng, done, remaining_dist

def get_reward(disc_temp, done, energy, bestE, preAction, currAction):
    reward = 0
    if done:
        if 0.1 < disc_temp < 0.5:
            if energy < bestE:
                reward += 200
        else:
            reward -= 200
    else:
        if preAction != currAction:
            reward -= 1
        if 0.1 < disc_temp < 0.5:
            reward += 2

    return reward
