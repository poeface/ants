
import numpy as np
import matplotlib.pyplot as plt #I'm just using matplotlib because I know how to do that already


def normalize(X):
    return X / np.sqrt(np.sum(X**2,0))

def main():

    #general setup
    dt = 0.05 #delta time between steps
    N = 5 #number of ants
    lb = np.array([[0],[0]]) + 1e-5 #lower bounds, meters
    ub = np.array([[1],[1]]) #upper bounds, meters
    #X = np.random.rand(2,N) * (ub - lb) + lb #initial ant positions in 2D are RANDOM
    X = ((ub + lb) / 2) * np.ones((2,N)) #initial ant positions in 2D are CENTERED
    V = np.random.rand(2,N) * 2 - 1 #initial ant velocities
    mode = np.zeros((1,N)) #modes are 0=searching, 1=sourcing
    P_search = np.zeros((2,0)) #searching pheromones
    P_source = np.zeros((2,0)) #sourcing pheromones

    #ant characteristics
    speed = 0.05 #meters / second, ants always move by this amount (unless hitting boundary)
    #speed = 0.1 #meters / second, ants always move by this amount (unless hitting boundary)
    wander = 2 #intensity of wandering effect (unitless)
    avoidance = 1 #unitless, measure of avoidance of certain pheromones
    pheromone_drop_period = 1 #seconds / drop
    search_pheromone_timer = np.zeros((1,N)) #time since last pheromone drop
    source_pheromone_timer = np.zeros((1,N)) #time since last source pheromone drop

    #pheromone characteristics
    #radius = speed / 2 - 1e-5 #meters, distance within which a pheromone is detected by an ant
    radius = 0.05 #meters, distance within which a pheromone is detected by an ant

    #environment
    #TODO

    #simulate
    #dt is a measurement of time that has passed since last frame.
    fig, ax = plt.subplots()
    while True:
        #compute position updates:
            #add wandering vector
        rand_angle = 2 * np.pi * np.random.rand(1,N)
        V = V + wander * np.concatenate((np.cos(rand_angle), np.sin(rand_angle)), axis=0) #add wandering vector
            #interact with search pheromones
        if P_search.size > 0:
            #print('before diff assigned:', X.shape)
            #print(np.transpose(np.expand_dims(P_search,2), (1,2,0)).shape)
            #print(np.transpose(np.expand_dims(X,2), (2,1,0)).shape)
            diffs = np.transpose(np.expand_dims(P_search,2), (1,2,0)) - np.transpose(np.expand_dims(X,2), (2,1,0)) #pairwise differences
            dists = np.sqrt(np.sum(diffs**2,2)) + 1e-10 #matrix of differences between pheromones and ants
            mode_mask = mode == 0
            #print(X.shape)
            #print(diffs.shape)
            #print(dists.shape)
            #print((diffs * np.exp(-(np.expand_dims(dists,2)/radius)**2)).shape)
            V = V - avoidance * normalize(np.sum((diffs / np.expand_dims(dists,2)) * (1/radius) * np.exp(-(np.expand_dims(dists,2)/radius)**2),0).T) #gaussian falloff for all

        #TODO #if searching, avoid search pheromones and seek source pheromones
            #interact with source pheromones
        #TODO #if sourcing, avoid source pheromones and see search pheromones
            #normalize vector
        #print(V.shape)
        V = normalize(V) #normalize velocity vectors each update

        #drop search pheromones (on timer)
        search_pheromone_timer += dt
        P_search = np.concatenate((P_search, X[:,search_pheromone_timer[0,:] > pheromone_drop_period]), axis=1) #drops a pheromone ifthe timer for an ant has passed the trigger period
        search_pheromone_timer[search_pheromone_timer > pheromone_drop_period] = 0 #resets the timer for each ant

        #drop search pheromones (on timer)

        #update positions
        X = X + speed * V * dt #real time

        #clamp to domain
        X = np.minimum(np.maximum(X, lb), ub)
        #print('after X assigned:', X.shape)

        #report
        #print(f'X = ({X[0,0]}, {X[1,0]})')
        #print(P_search.shape)

        #render
        ax.clear()
        ax.scatter(P_search[0,:], P_search[1,:], color='r', s=5)
        ax.scatter(X[0,:], X[1,:], color='b', s=10)
        ax.set_xlim(lb[0], ub[0])
        ax.set_ylim(lb[1], ub[1])
        ax.grid(True)
        plt.pause(0.05)

    plt.show()

if __name__ == '__main__':
    main()
