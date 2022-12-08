import csv
edgeFile = 'edges.csv'

from queue import PriorityQueue

def ucs(start, end):
    # Begin your code (Part 3)
    adjacency={} # create an adjacency list
    with open(edgeFile, newline='') as csvfile: # open the csvfile
        rows = csv.DictReader(csvfile) # open with "dictionary" data structure
        for row in rows:
            key = int(row['start']) # "key" represents the start node 
            value = [int(row['end']),float(row['distance'])] # "value" represents the list of end node and distance
            if key not in adjacency.keys():
                adjacency[key] = [] # encouter the key at the first time -> create a new list for the key
            adjacency[key].append(value) # update the adjacency list
    # setup initial values
    parent = {}
    queue = []
    visited =[]
    dist = 0.0
    now =  start
    while now != end: # if 'now' not equal to 'end', we continue the process
        if (now in adjacency) and (now not in visited):
            visited.append(now)
            for ad in adjacency[now]: # a loop for every adjacency node
                if ad[0] not in visited: # ad[0]: the adjacency node, ad[1]: the distance from 'now' to ad[0]
                    # dist(recorded at last iteration) + dist(from 'now' to ad[0]) = total distance from 'start' to ad[0]
                    queue.append([ dist + ad[1], ad[0] ]) 
                    parent[ ad[0] ] = now # record the parent of this adjacency node
        queue = sorted(queue) # sort to find the minimum distance 
        now =  queue[0][1] # record the node, and expend road using its adjacency nodes at next iteration
        dist = queue[0][0] # record the distance, it will use when next iteration
        queue.pop(0)
    path = [end] # finish the while loop, and evaluate the result
    while path[-1] != start:
        path.append(parent[path[-1]]) # find the path from 'end' to 'start'
    path.reverse()
    return path, dist, len(visited)
    # End your code (Part 3)


if __name__ == '__main__':
    path, dist, num_visited = ucs(2270143902, 1079387396)
    print(f'The number of path nodes: {len(path)}')
    print(f'Total distance of path: {dist}')
    print(f'The number of visited nodes: {num_visited}')
