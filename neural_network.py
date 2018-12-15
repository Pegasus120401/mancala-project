import torch
import torch.nn as nn


class NN(nn.Module):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    def __init__(self, layers, alpha=0.01, final_alpha = 1e-5):
        super(NN, self).__init__()
        self.alpha = alpha
        self.final_alpha = final_alpha
        #list of layers from input to output
        self.layers = layers
        #create model and append linear layers with biases + activation functions
        self.model = nn.Sequential()
        for i in range(len(self.layers)-1):
            self.model.add_module("linear_layer_" + str(i), nn.Linear(self.layers[i], self.layers[i + 1]))
            self.model.add_module("activation_" + str(i), nn.Tanh())

        #set optimizer, loss function, device
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.alpha)
        self.loss = torch.nn.MSELoss()
        self.to(NN.device)


    def forward(self, input):
        #forward propagate by passing input from function to function
        out = input
        for func in self.model:
            out = func(out)

        return out


