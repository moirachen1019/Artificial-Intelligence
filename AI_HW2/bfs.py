import csv
edgeFile = 'edges.csv'


def bfs(start, end):
    # Begin your code (Part 1)
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
    totaldist = 0.0
    visited = []
    parent = {}
    dist = {}
    queue = []
    queue.append(start)
    visited.append(start)
    while queue:
        top = queue.pop(0) # get the first one of the queue
        if top == end: # if 'top' equal to 'end', we started to evaluate the result.
            path = [end]
            while path[-1] != start:
                totaldist += dist[path[-1]] # calculate the total distance
                path.append(parent[path[-1]]) # find the path from 'end' to 'start'
            path.reverse()
            return path, totaldist, len(visited)
        if top in adjacency:
            for ad in adjacency[top]: # a loop for every adjacency node
                if ad[0] not in visited: # only handle the node which hasn't visited
                    queue.append(ad[0]) # add into the queue
                    parent[ ad[0] ] = top # record the parent of this adjacency node
                    dist[ ad[0] ] = ad[1] #record the distance from 'top' to this adjacency node
                    visited.append(ad[0])
    # End your code (Part 1)

if __name__ == '__main__':
    path, dist, num_visited = bfs(2270143902, 1079387396)
    print(f'The number of path nodes: {len(path)}')
    print(f'Total distance of path: {dist}')
    print(f'The number of visited nodes: {num_visited}')