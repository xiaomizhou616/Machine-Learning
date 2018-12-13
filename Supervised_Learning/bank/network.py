import torch
import torch.nn as nn
import math
        
class classifier(nn.Module):
    def __init__(self, InputDim=20, nLayers=1, nNeurons=20):
        super(classifier, self).__init__()
        self.main = nn.Sequential()
        self.main.add_module("Input", nn.Linear(InputDim, nNeurons, bias=False))
        self.main.add_module("Input_BN", nn.BatchNorm1d(nNeurons))
        self.main.add_module("Input_ReLU", nn.ReLU(inplace=True))
        
        for ii in range(nLayers):
            self.main.add_module("Linear_{}".format(str(ii)), nn.Linear(nNeurons, nNeurons, bias=False))
            self.main.add_module("BN_{}".format(str(ii)), nn.BatchNorm1d(nNeurons))
            self.main.add_module("ReLU_{}".format(str(ii)), nn.ReLU(inplace=True))
            
        self.main.add_module("output", nn.Linear(nNeurons, 1))
        self.main.add_module("Sigmoid", nn.Sigmoid())
        
        for m in self.modules():
            if isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1.)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                if m.bias is not None:
                    m.bias.data.fill_(0.0)
                    
    def forward(self, x):
        return torch.squeeze(self.main(x), dim=1)