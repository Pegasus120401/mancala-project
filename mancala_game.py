import time
import mancala_players as mp
import numpy as np
import torch, os, random




class Game():
    """The main mancala game class"""
    #player_dict = {"PI": mancala_players.Policy_iter_agent, "human":mancala_players.Human_Player, "random":mancala_players.Random_Player, "ql":mancala_players.Deep_Q_Learning_Player}
    def __init__(self, player1, player2, stones=4, verbose=False):
        self.verbose = verbose
        if isinstance(player1, mp.Human_Player) or isinstance(player2, mp.Human_Player):
            self.verbose = True
        #player objects
        # self.player1 = self.player_dict[player1](self, 0)
        # self.player2 = self.player_dict[player2](self, 1)
        self.player1 = player1
        self.player1.set_player_number(0)
        self.player1.set_game(self)
        self.player2 = player2
        self.player2.set_player_number(1)
        self.player2.set_game(self)
        self.players = (self.player1, self.player2)
        self.turn = random.randint(0,1)

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


    def re_initialize(self):
        self.turn = random.randint(0,1)
        for p in self.players:
            p.re_initialize()


    def test_mode(self):
        for p in self.players: p.test_mode()


    def eq_states(self, s1, s2):
        if len(s1) != len(s2): return False
        for i in range(len(s1)):
            if s1[i] != s2[i]:
                return False

        return True




    def print_game(self):
        """Prints out board."""
        os.system('cls')
        print(" ", end="")
        for i in range(6):
            print("  " + str(1+i), end="")
        print("     PLAYER 1")
        print("  " + str(self.player1.holes[::-1]))
        print("" + str(self.player1.pit) + 5*"    " + str(self.player2.pit))
        print("  " + str(self.player2.holes), end="")
        print(" ")
        for i in range(6):
            print("  " + str(6-i), end="")
        print("     PLAYER 2")



    def get_state(self, player):
        "returns current state as seen by each player"
        s = self.players[player].holes[::-1] + self.players[player-1].holes[::-1]
        s = torch.Tensor(s).to(self.device)
        return s


    def is_terminal(self, state):
        if self.eq_states(state[0:6], torch.Tensor([0,0,0,0,0,0]).to(self.device)) or self.eq_states(state[6:12], torch.Tensor([0,0,0,0,0,0]).to(self.device)):
            return True
        return False

    def end_game(self):
        """Checks whether state is terminal (if on one of the sides' there are no more stones).
         If so, ends the game by moving all remaining stones to the player's pit."""
        for p in self.players:
            p.pit += sum(p.holes)
            p.holes = np.array([0,0,0,0,0,0])

    def play(self):
        while(not self.is_terminal(self.get_state(0))):
            if self.verbose: time.sleep(1)
            """print the board"""

            if self.verbose: self.print_game()

            player = self.players[self.turn]
            if self.verbose: print("PLAYER " + str(self.turn +1) + " TURN")
            #choice between 0 and 5
            choice = player.do_turn()
            if self.verbose: print("Chose " + str(choice + 1))
            while (True):
                """until input is valid"""
                if(choice <0 or choice > 5):
                    #check legality of input
                    if self.verbose: print("Illegal hole number!")
                    choice = player.do_turn()
                elif(player.holes[5-choice] == 0):
                    if self.verbose: print("No stones in hole number " + str(choice+1) + ", please choose another hole.")
                    choice = player.do_turn()
                else:
                    break
            """setting up holes and stones vars which are vital for turn execution. Because of cyclic property of game, some properties are flipped around."""
            #print(player.holes)
            #holes = np.concatenate((player.holes, np.array([player.pit]), self.players[self.turn-1].holes))
            holes = player.holes + [player.pit] + self.players[self.turn - 1].holes
            #print(holes[5-choice])
            #print(type(holes[5-choice]))
            stones_in_hole = range(holes[5 - choice])


            j = 6-choice
            """execution of turn: until stones are over, one each is distributed to consequent holes."""
            for i in stones_in_hole:
                holes[5-choice] -= 1
                holes[j%13]+=1
                j+=1
            """the j-1 is because j is increased after last iteration unnecesarly"""
            if (j-1)%13 < 6 and holes[(j-1)%13] == 1:
                holes[6] +=holes[(j-1)%13] + holes[13-(j%13)]
                if self.verbose: print("PLAYER " + str(self.turn+1) + " captured " + str(holes[13-(j%13)])+ " stone(s)!")
                holes[(j - 1) % 13] = 0
                holes[13 - (j % 13)] = 0


            #if last stone ends up in pit player gets another turn
            player.holes = holes[:6]
            player.pit = holes[6]
            self.players[self.turn - 1].holes = holes[7:]
            if(j%13 != 7):
                self.turn = abs(self.turn - 1)
            else:
                if self.verbose: print("Another turn for PLAYER " + str(self.turn+1) + "!")


        self.end_game()

        #learning last action
        try:
            self.player1.learn(self.player1.state, self.player1.last_action, self.player1.get_reward())
        except:
            pass

        try:
            self.player2.learn(self.player2.state, self.player2.last_action, self.player2.get_reward())
        except:
            pass

        if self.players[0].pit > self.players[1].pit:
            if self.verbose:print("Player 1 won!")
            return  1
        elif self.players[0].pit < self.players[1].pit:
            if self.verbose:print("Player 2 won!")
            return 2
        else:
            if self.verbose:print("Tie!")
            return 0











