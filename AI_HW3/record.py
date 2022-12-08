'''
Pacman emerges victorious! Score: 1347
Pacman emerges victorious! Score: 1324
Pacman emerges victorious! Score: 1339
Pacman emerges victorious! Score: 1329
Pacman emerges victorious! Score: 1354
Pacman emerges victorious! Score: 1169
Pacman emerges victorious! Score: 1152
Pacman emerges victorious! Score: 1325
Pacman emerges victorious! Score: 1381
Pacman emerges victorious! Score: 1345
Average Score: 1306.5
考慮三個最近的食物、膠囊
'''
    INF = 1000000000000.0
    WEIGHT_FOOD = 5.0 
    WEIGHT_CAP = 10.0
    WEIGHT_GHOST = -10.0
    WEIGHT_SCARED_GHOST = 150.0 
    '''
    adjust_score = currentGameState.getScore() # adjust score is modified from base score
    distancesToFoodList = [util.manhattanDistance(newPos, foodPos) for foodPos in newFood.asList()]
    count = 0
    #if len(distancesToFoodList) > 0:
        #adjust_score += WEIGHT_FOOD / min(distancesToFoodList)
    while len(distancesToFoodList) > 0 and count < 3:
        adjust_score += WEIGHT_FOOD / min(distancesToFoodList)
        distancesToFoodList.remove( min(distancesToFoodList) )
        #WEIGHT_FOOD -= 3
        count += 1
    #else:
        #adjust_score += WEIGHT_FOOD
    distancesToCapList = []
    for ii in range(len(newCapsules)):
        distancesToCapList.append(abs(newPos[0] - newCapsules[ii][0]) + abs(newPos[1] - newCapsules[ii][1]))
        if len(distancesToCapList) > 0:
                adjust_score += WEIGHT_CAP / min(distancesToCapList)
            

    for ghost in newGhostStates:
        distance = manhattanDistance(newPos, ghost.getPosition())
        if distance > 0:
            if ghost.scaredTimer > 0:
                adjust_score += WEIGHT_SCARED_GHOST / distance
            else:
                adjust_score += WEIGHT_GHOST / distance
        else:
            return -INF
    return adjust_score
'''

