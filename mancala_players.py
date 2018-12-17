import random
import numpy as np
import torch
import neural_network as nn
import mancala_game
import pickle


#basic player class
class Player():
    def __init__(self):
        self.stones = 4
        self.holes = [self.stones for i in range(6)]
        self.pit = 0
        self.game = None
        self.player = None
    
    def re_initialize(self):
        self.holes = [self.stones for i in range(6)]
        self.pit = 0

    def set_player_number(self, num):
        self.player = num


    def set_game(self, game):
        self.game = game


    def test_mode(self):
        pass

#human player class
class Human_Player(Player):
    def __init__(self):
        super().__init__()

    def do_turn(self):
        #get input from user
        choice = int(input("Choose hole (1-6): "))
        return choice-1

#random player class
class Random_Player(Player):
    def __init__(self):
        super().__init__()

    def do_turn(self):
        choice = random.randint(0, 5)

        # time.sleep(1)
        return choice




#rl agent
class Deep_Q_Learning_Player(Player):
    # exp_replay = []
    # exp_replay_max_size = 100
    # #net is class attribute - same for all agents, allowing agents to learn from playing against each other.
    # try:
    #     #load neural net from file
    #     pickle_file_name = "NN_10_10.pickle"
    #     pickle_file = open(pickle_file_name, "rb")
    #     net = pickle.load(pickle_file)
    #     pickle_file.close()
    # except:
    #     #create new neural net
    #     net = nn.NN([12, 10, 10, 6])
    @classmethod
    def store_net(cls):
        with open(cls.pickle_file_name, 'wb+') as file:
            pickle.dump(cls.moving_net, file)



    def __init__(self, double_net=False, gamma=0.9, epsilon=0.2, final_eps=0.05):
        super().__init__()
        self.pit_before = self.pit
        #player number

        self.last_action = 0
        self.reward = 0
        self.num_actions = 6
        self.steps = 0
        self.gamma = gamma
        self.epsilon = epsilon
        self.eps_end = final_eps
        self.minibatch_size = 10
        self.tanh = torch.nn.Tanh()
        self.norm = lambda x: self.tanh(x/8)
        self.double_net = double_net
        self.is_test = False

    def test_mode(self):
        self.is_test = True

    def learn_mode(self):
        self.is_test = False

    def re_initialize(self):
        super().re_initialize()
        self.last_action = 0
        self.reward = 0
        self.steps = 0
        self.target_net = self.moving_net




    def policy(self, available_actions, actions_values):
        #e-greedy policy only
        if (np.random.random_sample() < self.epsilon) and not self.is_test:
            #random action
            if self.epsilon - 1e-5 > self.eps_end:
                # epsilon decay
                self.epsilon -= 1e-5
            if self.verbose: print("random choice")
            return available_actions[np.random.randint(0, len(available_actions))]

        else:
            #action with max value
            return available_actions[np.argmax([actions_values[i] for i in available_actions])]


    def choose_action(self):
        #all actions where there are stones in hole
        available_actions = [5 - i for i in range(self.num_actions) if self.holes[i] != 0]
        #forward with tanh normalization
        actions_values = self.moving_net.forward(self.norm(self.state))
        choice = self.policy(available_actions, actions_values)
        return choice




    def optimize(self, minibatch):
        loss = 0
        for sars in minibatch:
            self.moving_net.zero_grad()
            #q-values of previous state
            Qpred = self.moving_net.forward(self.norm(sars[0])).to(nn.NN.device)
            #q-values of current state
            if self.double_net: Qnext = self.target_net.forward(self.norm(sars[3])).to(nn.NN.device)
            else: Qnext = self.moving_net.forward(self.norm(sars[3])).to(nn.NN.device)
            #available actions
            available_actions = [5 - i for i in range(self.num_actions) if self.holes[i] != 0]
            #q-value of available actions in current state
            Q_available = torch.Tensor([Qnext[i] for i in available_actions]).to(nn.NN.device)
            Qtarget = Qpred
            #array of 0 with a 1 in action number
            id = torch.eye(self.num_actions).to(nn.NN.device)[sars[1]:sars[1]+1].view(self.num_actions)
            #Qtarget[action] = reward + self.GAMMA * torch.max(Q_available)
            #target q-values
            n_reward = float(self.norm(sars[2]).item())
            if (self.game.is_terminal(sars[3])):
                Qtarget = Qtarget * (1 - id) + id * (n_reward)
            else:
                Qtarget = Qtarget * (1 - id) + id * float(n_reward + self.gamma * torch.max(Q_available))
            #print(Qtarget, reward, n_reward, self.GAMMA * torch.max(Q_available))






            #calculate loss
            loss += self.moving_net.loss(Qpred, Qtarget).to(nn.NN.device)
        #backpropagate error
        loss.backward()
        #gradient descent step
        self.moving_net.optimizer.step()

        if self.moving_net.alpha - 1e-10 > self.moving_net.final_alpha:
            self.moving_net.alpha -= 1e-10
            self.moving_net.optimizer.lr = self.moving_net.alpha


    def learn(self, state, action, reward, next_state):
        minibatch = []
        if len(self.exp_replay) < self.exp_replay_max_size:
            self.exp_replay.append((state, action, reward, next_state))
            #print(len(self.exp_replay))

            for i in range(self.minibatch_size):
                r = random.randint(0, len(self.exp_replay)-1)
                minibatch.append(self.exp_replay[r])




        else:
            self.exp_replay.pop(0)
            self.exp_replay.append((state, action, reward, next_state))
            for i in range(self.minibatch_size):
                r = random.randint(0, len(self.exp_replay)-1)
                minibatch.append(self.exp_replay[r])



        self.optimize(minibatch)






    def get_reward(self):
        #difference in points minus difference in opponent points
        return torch.Tensor([self.pit - self.pit_before - (self.enemy.pit - self.enemy_pit_before)])

    def do_turn(self):
        if self.steps == 0:
            self.enemy = self.game.players[self.player - 1]
            self.enemy_pit_before = self.enemy.pit
            self.state = self.game.get_state(self.player)
            self.verbose = self.game.verbose




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
#     def __init__(self):
#         super().__init__()
#         self.simulation_game = mancala_game.Game()
#     def do_turn(self):
#         pass


class Policy_iter_agent(Deep_Q_Learning_Player):
    exp_replay = []
    exp_replay_max_size = 100
    #nets are class attributes - same for all agents, allowing agents to learn from playing against each other.
    try:
        #load neural net from file
        pickle_file_name = "NN_10_10_PI.pickle"
        with open(pickle_file_name, 'rb') as file:
            target_net = pickle.load(file)
        moving_net = target_net
    except:
        #create new neural net
        target_net = nn.NN([12, 10, 10, 6])
        moving_net = target_net

    def __init__(self, gamma=0.9, epsilon=0.2, final_eps=0.05):
        super().__init__(True, gamma, epsilon, final_eps)




class Value_iter_agent(Deep_Q_Learning_Player):
    exp_replay = []
    exp_replay_max_size = 100
    # net is class attribute - same for all agents, allowing agents to learn from playing against each other.
    try:
        # load neural net from file
        pickle_file_name = "NN_10_10_VI.pickle"
        with open(pickle_file_name, 'rb') as file:
            moving_net = pickle.load(file)

    except:
        # create new neural net
        moving_net = nn.NN([12, 10, 10, 6])


    def __init__(self, gamma=0.9, epsilon=0.2, final_eps=0.05):
         super().__init__(False, gamma, epsilon, final_eps)