import csv
edgeFile = 'edges.csv'
heuristicFile = 'heuristic.csv'


def astar_time(start, end):
    # Begin your code (Part 6)
    adjacency={} # create an adjacency list
    with open(edgeFile, newline='') as csvfile: # open the csvfile
        rows = csv.DictReader(csvfile) # open with "dictionary" data structure
        for row in rows:
            key = int(row['start']) # "key" represents the start node 
            value = [int(row['end']),float(row['distance']),float(row['speed limit'])] # "value" represents the list of end node, distance, and speed
            if key not in adjacency.keys():
                adjacency[key] = [] # encouter the key at the first time -> create a new list for the key
            adjacency[key].append(value) # update the adjacency list
    heuristic = {}
    with open(heuristicFile, newline='') as hfile:
        rows = csv.DictReader(hfile) # open with "dictionary" data structure
        for row in rows:
            key = int(row['node']) # "key" represents the node
            value = float(row[str(end)]) #"value" represents the distance from this node to 'end'
            heuristic[key] = value
    # setup initial values
    parent = {}
    time = {}
    queue = []
    visited =[]
    now =  start
    time[now] = 0.0
    while now != end: # if 'now' not equal to 'end', we continue the process
        if (now in adjacency) and (now not in visited):
            visited.append(now)
            for ad in adjacency[now]:
                if ad[0] not in visited:
                    queue.append([ time[now] + (ad[1]/ad[2])*3.6 + heuristic[ad[0]]/60*3.6 , ad[0] ]) 
                    parent[ ad[0] ] = now # record the parent of this adjacency node
                    time[ ad[0] ] = time[now] + (ad[1]/ad[2])*3.6 # record the time from 'start' to ad[0]
        queue = sorted(queue) # sort to find the minimum (distance + heuristic function value)
        now =  queue[0][1] # record the node, and expend road using its adjacency nodes at next iteration
        queue.pop(0)
    path = [end] # finish the while loop, and evaluate the result
    while path[-1] != start:
        path.append(parent[path[-1]]) # find the path from 'end' to 'start'
    path.reverse()
    return path, time[now], len(visited)
    # End your code (Part 6)



if __name__ == '__main__':
    path, time, num_visited = astar_time(2270143902, 1079387396)
    print(f'The number of path nodes: {len(path)}')
    print(f'Total second of path: {time}')
    print(f'The number of visited nodes: {num_visited}')
