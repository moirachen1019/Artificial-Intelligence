import csv
edgeFile = 'edges.csv'
heuristicFile = 'heuristic.csv'

def astar(start, end):
    # Begin your code (Part 4)
    adjacency={} # create an adjacency list
    with open(edgeFile, newline='') as csvfile: # open the csvfile
        rows = csv.DictReader(csvfile) # open with "dictionary" data structure
        for row in rows:
            key = int(row['start']) # "key" represents the start node 
            value = [int(row['end']),float(row['distance'])] # "value" represents the list of end node and distance
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
    queue = []
    visited =[]
    visited.append(start)
    for ad in adjacency[start]:
        queue.append([ad[1]+heuristic[ad[0]], ad[1], start, ad[0]])
    while 1:
        queue = sorted(queue) # sort to find the minimum (distance + heuristic function value)
        now =  queue[0][3] # record the node, and expend road using its adjacency nodes at next iteration
        dist = queue[0][1] # record the distance from 'start' to 'now'
        par = queue[0][2] # record the parent of 'now'
        queue.pop(0)
        if now in adjacency and now not in visited:
            visited.append(now)
            parent[now] = par
            for ad in adjacency[now]:
                if ad[0] not in visited:
                # it's like ucs search, but considering the heuristic function value
                # record distance + heuristic,  distance only, parent, adjacency node at the same time
                    queue.append([ dist + ad[1] + heuristic[ad[0]] ,dist + ad[1], now, ad[0] ]) 
        if now == end:  # if 'now' not equal to 'end', we continue the process
            break
    path = [end] # finish the while loop, and evaluate the result
    while path[-1] != start:
        path.append(parent[path[-1]]) # find the path from 'end' to 'start'
    path.reverse()
    return path, dist, len(visited)
    # End your code (Part 4)


if __name__ == '__main__':
    path, dist, num_visited = astar(2270143902, 1079387396)
    print(f'The number of path nodes: {len(path)}')
    print(f'Total distance of path: {dist}')
    print(f'The number of visited nodes: {num_visited}')



