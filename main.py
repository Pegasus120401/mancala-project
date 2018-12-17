import mancala_game as mg
import pickle
import mancala_players as mp
import matplotlib.pyplot as plt
import numpy as np
import random

num_episodes = 3
PI_vs_VI_stats = []
print("starting...")
PI_player1 = mp.Policy_iter_agent()
PI_player2 = mp.Policy_iter_agent()
VI_player1 = mp.Value_iter_agent()
VI_player2 = mp.Value_iter_agent()
random_player = mp.Random_Player()
human_player = mp.Human_Player()
human_vs_PI_game = mg.Game(human_player, PI_player2)
human_vs_VI_game = mg.Game(human_player, VI_player2)
PI_vs_PI_game = mg.Game(PI_player1, PI_player2)
VI_vs_VI_game = mg.Game(VI_player1, VI_player2)
PI_vs_VI_game = mg.Game(PI_player2, VI_player1)
PI_vs_random_game = mg.Game(PI_player1, random_player)
VI_vs_random_game = mg.Game(VI_player2, random_player)
for j in range(num_episodes):
    #result counters



    for i in range(10):
        #play game

        game = PI_vs_PI_game
        game.play()
        game.re_initialize()
        game = VI_vs_VI_game
        game.play()
        game.re_initialize()

    # game = human_vs_VI_game
    # game.play()
    # game.re_initialize()

    # TESTS

    one = 0
    two = 0
    tie = 0
    for i in range(5):
        #play game and count wins

        game = PI_vs_random_game
        game.test_mode()
        cnt = game.play()
        game.re_initialize()
        if cnt == 0:
            tie += 1
        elif cnt == 1:
            one += 1
        else:
            two += 1

    print(str(j + 1) + ") PI: " + str(one) + " random: " + str(two) + " Tie: " + str(tie))



    one = 0
    two = 0
    tie = 0
    for i in range(5):
        # play game and count wins

        game = VI_vs_random_game
        game.test_mode()
        cnt = game.play()
        game.re_initialize()
        if cnt == 0:
            tie += 1
        elif cnt == 1:
            one += 1
        else:
            two += 1

    print(str(j + 1) + ") VI: " + str(one) + " random: " + str(two) + " Tie: " + str(tie))


    one = 0
    two = 0
    tie = 0
    for i in range(10):
        #play game and count wins
        game = PI_vs_VI_game
        game.test_mode()
        cnt = game.play()
        game.re_initialize()
        if cnt == 0:
            tie += 1
        elif cnt == 1:
            one += 1
        else:
            two += 1

    #save neural network data
    mp.Policy_iter_agent.store_net()
    mp.Value_iter_agent.store_net()
    #later used for graph
    PI_vs_VI_stats.append((one, two, tie))
    print(str(j+1) + ") PI: " + str(one) + " VI: " + str(two) + " Tie: " + str(tie))


game = human_vs_PI_game
game.test_mode()
cnt = game.play()
game.re_initialize()
game = human_vs_VI_game
game.test_mode()
cnt = game.play()
game.re_initialize()
#plot graph
x = np.arange(0, num_episodes, 1)
PI_win = np.array([i[0] for i in PI_vs_VI_stats])
VI_win = np.array([i[1] for i in PI_vs_VI_stats])
plt.ion()
fig = plt.figure()
ax = fig.add_subplot(111)
line1, = ax.plot(x, PI_win, 'b-')
line2, = ax.plot(x, VI_win, 'g-')
while 1:
    plt.pause(1)
    fig.canvas.draw()

