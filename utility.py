import numpy as np

def true_zones(lst):
    first = lst[0]
    in_zone = lst[0]
    start = None
    zones = []
    
    for i in range(len(lst)):
        if lst[i]:
            if not in_zone:
                in_zone = True
                start = i
        else:
            if in_zone:
                in_zone = False
                if first:
                    first = False
                else:
                    zones.append([start, i])
                    
    return zones         