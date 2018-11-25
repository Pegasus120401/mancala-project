import random
import numpy as np
import torch
import neural_network, mancala_game
import pickle


#basic player class
class Player():
    def __init__(self, stones, game, player):
        self.stones = stones
        self.holes = [self.stones for i in range(6)]
        self.pit = 0
        self.game = game
        self.player = player
    
    def re_initialize(self):
        self.holes = [self.stones for i in range(6)]
        self.pit = 0

#human player class
class Human_Player(Player):
    def __init__(self, stones, game, player):
        super().__init__(stones, game, player)

    def do_turn(self):
        #get input from user
        choice = int(input("Choose hole (1-6): "))
        return choice-1

#random player class
class Random_Player(Player):
    def __init__(self, stones, game, player):
        super().__init__(stones, game, player)

    def do_turn(self):
        choice = random.randint(0, 5)

        # time.sleep(1)
        return choice




#rl agent
class Q_Learning_Player(Player):
    exp_replay = []
    exp_replay_max_size = 100
    #net is class attribute - same for all agents, allowing agents to learn from playing against each other.
    try:
        #load neural net from file
        pickle_file_name = "NN_10_10.pickle"
        pickle_file = open(pickle_file_name, "rb")
        net = pickle.load(pickle_file)
        pickle_file.close()
    except:
        #create new neural net
        net = neural_network.NN([12, 10, 10, 6])

    def __init__(self, stones, game, player, gamma=0.9, epsilon=0.2, final_eps=0.05, num_actions=6):
        super().__init__(stones, game, player)
        self.pit_before = self.pit
        #player number
        self.player = player
        self.verbose = game.verbose
        self.last_action = 0
        self.reward = 0
        self.num_actions = num_actions
        self.steps = 0
        self.GAMMA = gamma
        self.EPSILON = epsilon
        self.EPS_END = final_eps
        self.minibatch_size = 3


    def re_initialize(self):
        super().re_initialize()
        self.last_action = 0
        self.reward = 0
        self.steps = 0


    def policy(self, available_actions, actions_values):
        #e-greedy policy only
        if (np.random.random_sample() < self.EPSILON):
            #random action
            if self.verbose: print("random choice")
            return available_actions[np.random.randint(0, len(available_actions))]

        else:
            #action with max value
            return available_actions[np.argmax([actions_values[i] for i in available_actions])]


    def choose_action(self):
        #all actions where there are stones in hole
        available_actions = [5 - i for i in range(self.num_actions) if self.holes[i] != 0]
        #forward with 1/16 normalization factor
        actions_values = self.net.forward(self.state/16)
        choice = self.policy(available_actions, actions_values)
        return choice




    def optimize(self, state, action, reward, next_state):
        self.net.zero_grad()
        #q-values of previous state
        Qpred = self.net.forward(state/16).to(self.net.device)
        #q-values of current state
        Qnext = self.net.forward(next_state/16).to(self.net.device)
        #available actions
        available_actions = [5 - i for i in range(self.num_actions) if self.holes[i] != 0]
        #q-value of available actions in current state
        Q_available = torch.Tensor([Qnext[i] for i in available_actions]).to(self.net.device)
        Qtarget = Qpred
        #array of 0 with a 1 in action number
        id = torch.eye(self.num_actions).to(self.net.device)[action:action+1].view(self.num_actions)
        #Qtarget[action] = reward + self.GAMMA * torch.max(Q_available)
        #target q-values
        n_reward = float(np.tanh(reward))
        if (self.game.is_terminal(next_state)):
            Qtarget = Qtarget * (1 - id) + id * (n_reward)
        else:
            Qtarget = Qtarget * (1 - id) + float(n_reward + self.GAMMA * torch.max(Q_available))*id
        #print(Qtarget, reward, n_reward, self.GAMMA * torch.max(Q_available))


        if self.EPSILON - 1e-3 > self.EPS_END:
            #epsilon decay
            self.EPSILON -= 1e-3

        if self.net.alpha - 1e-8 > self.net.final_alpha:
            self.net.alpha -= 1e-8
            self.net.optimizer.lr = self.net.alpha

        #calculate loss
        loss = self.net.loss(Qpred, Qtarget).to(self.net.device)
        #backpropagate error
        loss.backward()
        #gradient descent step
        self.net.optimizer.step()


    def learn(self, state, action, reward, next_state):
        if len(self.exp_replay) < self.exp_replay_max_size:
            self.exp_replay.append((state, action, reward, next_state))
            #print(len(self.exp_replay))
            r = random.randint(0, len(self.exp_replay) - min(len(self.exp_replay), self.minibatch_size))
            for i in range(r, r+min(len(self.exp_replay), self.minibatch_size)):
                self.optimize(*self.exp_replay[i])

        else:
            try:
                self.exp_replay.pop(0)
                self.exp_replay.append((state, action, reward, next_state))
                r = random.randint(0, len(self.exp_replay) - self.minibatch_size)
                for i in range(r, r+self.minibatch_size):
                    self.optimize(*self.exp_replay[i])

            except Exception as e:
                print(e)
                #print(Q_Learning_Player.rep_idx)





    def get_reward(self):
        #difference in points minus difference in opponent points
        return self.pit - self.pit_before - (self.enemy.pit - self.enemy_pit_before)

    def do_turn(self):
        if self.steps == 0:
            self.enemy = self.game.players[self.player - 1]
            self.enemy_pit_before = self.enemy.pit
            self.state = self.game.get_state(self.player)




        self.reward = self.get_reward()
        #learn Q(state, last_action) =  Q(state, last_action) + alpha*(reward + gamma*max(Q(state_now,actions))- Q(state, last_action))
        if self.steps: self.learn(self.state, self.last_action, self.reward, self.game.get_state(self.player))
        self.state = self.game.get_state(self.player)
        self.pit_before = self.pit
        self.enemy_pit_before = self.enemy.pit
        #choice between 0 and 5
        choice = self.choose_action()
        self.last_action = choice
        self.steps += 1
        return choice





# class Minimax_Player(Player):
#     def __init__(self, stones, game, player):
#         super().__init__(stones, game, player)
#         self.simulation_game = mancala_game.Game()
#     def do_turn(self):
#         pass


