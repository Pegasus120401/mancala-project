import mancala_game
import pickle
import mancala_players
import matplotlib.pyplot as plt
import numpy as np

num_episodes = 20
y = []
print("starting...")
ql_vs_ql_game = mancala_game.Game("ql", "ql", verbose=False)
ql_vs_human_game = mancala_game.Game("ql", "human", verbose=False)
human_vs_ql_game = mancala_game.Game("human", "ql", verbose=False)
ql_vs_rand_game = mancala_game.Game("ql", "random", verbose=False)
for j in range(num_episodes):
    #result counters

    one = 0
    two = 0
    tie = 0
    game = human_vs_ql_game
    for i in range(20):
        #play game
        game.play()
        game.re_initialize()

    game = ql_vs_rand_game
    for i in range(5):
        #play game and count wins

        cnt = game.play()
        game.re_initialize()
        if cnt == 0:
            tie += 1
        elif cnt == 1:
            one += 1
        else:
            two += 1

    #save neural network data
    pickle_file = open("NN_10_10.pickle", "wb")
    pickle.dump(mancala_players.Q_Learning_Player.net, pickle_file)
    pickle_file.close()
    #later used for graph
    y.append(one)
    print(str(j+1) + ") One: " + str(one) + " Two: " + str(two) + " Tie: " + str(tie))



#plot graph
x = np.arange(0, num_episodes, 1)
y = np.array(y)
plt.ion()
fig = plt.figure()
ax = fig.add_subplot(111)
line1, = ax.plot(x, y, 'b-')
print("average: " + str(sum(y)/len(y)))
while 1:
    plt.pause(1)
    fig.canvas.draw()

