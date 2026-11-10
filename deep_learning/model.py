import torch.nn as nn


class NeuralAmp(nn.Module):
    def __init__(self, hidden_size=64, num_layers=2, input_size=1):
        # input_size = 1 (solo audio) o 1 + nº de amplificadores (modelo condicionado)
        super(NeuralAmp, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )
        self.linear = nn.Linear(hidden_size, 1)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        predictions = self.linear(lstm_out)
        return predictions
