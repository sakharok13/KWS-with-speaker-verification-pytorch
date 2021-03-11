from torch import nn

class Nnet(nn.Module):
    def __init__(self, embedding_dim=256, lstm_conf=None):
        super(Nnet, self).__init__()
        self.encoder = nn.LSTM(input_size=40, hidden_size=512) #input_size should be equal to n_mels 
        self.linear = nn.Linear(512, 128)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x, _ = self.Nnet(x)
        if x.dim() == 3:
            x = self.linear(x[:, -1, :])
        else:
            x = self.linear(x[-1, :])
        return x / torch.norm(x, dim=-1, keepdim=True)
        
