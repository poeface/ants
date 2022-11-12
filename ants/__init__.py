
import numpy as np
import matplotlib.pyplot as plt #I'm just using matplotlib because I know how to do that already


def normalize(X):
    return X / np.sqrt(np.sum(X**2,0))

def proximity(X, Y):
    #compute the distance of each X to each Y
    diffs = np.transpose(np.expand_dims(Y,2), (1,2,0)) - np.transpose(np.expand_dims(X,2), (2,1,0)) #pairwise differences
    return np.sqrt(np.sum(diffs**2,2)) #matrix of distance between X and Y

def attraction(X, Y, sigma):
    #computes a normalized vector for each X pointed toward targets Y. Range of attraction is governed by gaussians with width sigma
    diffs = np.transpose(np.expand_dims(Y,2), (1,2,0)) - np.transpose(np.expand_dims(X,2), (2,1,0)) #pairwise differences
    dists = np.sqrt(np.sum(diffs**2,2)) + 1e-10 #matrix of distance between X and Y
    return np.sum((diffs / np.expand_dims(dists,2)) * (1/sigma) * np.exp(-(np.expand_dims(dists,2)/sigma)**2),0).T #gaussian falloff for all


def main():

    #general setup
    dt = 0.05 #delta time between steps
    N = 10 #number of ants
    lb = np.array([[0],[0]]) + 1e-5 #lower bounds, meters
    ub = np.array([[1],[1]]) #upper bounds, meters
    #X = np.random.rand(2,N) * (ub - lb) + lb #initial ant positions in 2D are RANDOM
    X = ((ub + lb) / 2) * np.ones((2,N)) #initial ant positions in 2D are CENTERED
    V = np.random.rand(2,N) * 2 - 1 #initial ant velocities
    mode = np.zeros((1,N)) #modes are 0=searching, 1=sourcing
    P_duration = 120 #seconds, how long the pheromones stick around once dropped
    P_search = np.zeros((2,0)) #searching pheromones
    P_search_timer = np.zeros((1,0)) #search pheromone times
    P_source = np.zeros((2,0)) #sourcing pheromones
    P_source_timer = np.zeros((1,0)) #search pheromone times

    #ant characteristics
    speed = 0.05 #meters / second, ants always move by this amount (unless hitting boundary)
    #speed = 0.1 #meters / second, ants always move by this amount (unless hitting boundary)
    wander = 0.5 #intensity of wandering effect (unitless)
    food_affinity = 5 #unitless, measure of how intensly ants seek out food when they are near it
    search_pheromone_affinity = -0.25 #unitless, measure of interest in search pheromones
    pheromone_drop_period = 1 #seconds / drop
    search_pheromone_timer = np.zeros((1,N)) #time since last pheromone drop
    source_pheromone_timer = np.zeros((1,N)) #time since last source pheromone drop

    #pheromone characteristics
    #radius = speed / 2 - 1e-5 #meters, distance within which a pheromone is detected by an ant
    pheromone_sigma = 0.05 #meters, distance within which a pheromone is detected by an ant

    #environment
    food = 0.25 * np.ones((2,1)) #locations of food sources
    food_sigma = 0.05 #meters, std dev of food gaussian
    food_radius = 0.05 #meters

    #simulate
    #dt is a measurement of time that has passed since last frame.
    fig, ax = plt.subplots()
    while True:

        #drop search pheromones (on timer)
        search_pheromone_timer += dt
        P_search = np.concatenate((P_search, X[:,
            np.logical_and(
                search_pheromone_timer[0,:] > pheromone_drop_period,
                mode[0,:] == 0)
            ]), axis=1) #drops a pheromone ifthe timer for an ant has passed the trigger period
        P_search_timer = np.concatenate((P_search_timer, np.zeros((1,X[:,search_pheromone_timer[0,:] > pheromone_drop_period].shape[1]))), axis=1) #add in corresponding timers too
        search_pheromone_timer[search_pheromone_timer > pheromone_drop_period] = 0 #resets the timer for each ant

        #drop source pheromones (on timer)
        #TODO

        #remove old search pheromones
        print(P_search.size)
        if P_search.size > 0:
            P_search_timer += dt
            print(P_search[:,P_search_timer[0,:] < P_duration].shape)
            P_search = P_search[:,P_search_timer[0,:] < P_duration]
            P_search_timer = P_search_timer[:,P_search_timer[0,:] < P_duration]

        #remove old source pheromones
        #TODO

        #check for food collection. When that happens, switch to sourcing mode
        mode[:, np.any(proximity(X, food) < food_radius, axis=0)] = 1

        #compute update vector:

            #add wandering vector
        rand_angle = 2 * np.pi * np.random.rand(1,N)
        V = V + wander * np.concatenate((np.cos(rand_angle), np.sin(rand_angle)), axis=0) #add wandering vector

            #seek food
        V = V + food_affinity * attraction(X, food, food_sigma) #gaussian falloff for all

            #interact with search pheromones
        if P_search.size > 0:
            V = V + search_pheromone_affinity * attraction(X, P_search, pheromone_sigma) #gaussian falloff for all

        #TODO #if searching, avoid search pheromones and seek source pheromones
            #interact with source pheromones
        #TODO #if sourcing, avoid source pheromones and seek search pheromones

            #normalize vector
        V = normalize(V) #normalize velocity vectors each update

        #update positions according to velocity
        X = X + speed * V * dt #real time

        #clamp inside domain
        X = np.minimum(np.maximum(X, lb), ub)

        #report
        #print(f'X = ({X[0,0]}, {X[1,0]})')
        #print(P_search.shape)

        #render
        ax.clear()
        ax.scatter(food[0,:], food[1,:], color=[0,1,0], s=30)
        ax.scatter(P_search[0,:], P_search[1,:], color='r', s=5)
        ax.scatter(X[0,:], X[1,:], color='b', s=10)
        ax.axis('square')
        ax.set_xlim(lb[0], ub[0])
        ax.set_ylim(lb[1], ub[1])
        ax.grid(True)
        plt.pause(dt)

    plt.show()

if __name__ == '__main__':
    main()
