import os
datadir = "speech_commands"

os.system('wget http://download.tensorflow.org/data/speech_commands_v0.01.tar.gz -O speech_commands_v0.01.tar.gz')
os.system('mkdir {datadir} && tar -C {datadir} -xvzf speech_commands_v0.01.tar.gz 1> log')

samples_by_target = {
    cls: [os.path.join(datadir, cls, name) for name in os.listdir("./speech_commands/{}".format(cls))]
    for cls in os.listdir(datadir)
    if os.path.isdir(os.path.join(datadir, cls))
}

import pandas as pd
import numpy as np
import torch
import torchaudio
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
from torch import distributions
import librosa
from IPython import display as display_
import wandb
from torch import nn

torch.backends.cudnn.deterministic = True
torch.manual_seed(19)
torch.cuda.manual_seed(19)
np.random.seed(19)

train_data = []
val_data = []
train_size = 0.85
for key in samples_by_target.keys():
  cur = samples_by_target[key]
  length = len(cur)
  train_ind = np.random.choice(length, int(train_size*length), replace=False)
  val_ind = np.setdiff1d(range(length), train_ind)
  for ind in train_ind:
    if key == 'marvin':
      for i in range(36):
        train_data.append((cur[ind], 1))
    elif key != '_background_noise_':
      train_data.append((cur[ind], 0))
  for ind in val_ind:
    if key == 'marvin':
      for i in range(36):
        val_data.append((cur[ind], 1))
    elif key != '_background_noise_':
      val_data.append((cur[ind], 0))
train_data = pd.DataFrame(data=train_data, columns = ['wav', 'class'])
val_data = pd.DataFrame(data=val_data, columns = ['wav', 'class'])

mel_spectrogramer = torchaudio.transforms.MelSpectrogram(
    sample_rate=16000,
    n_fft=1024,
    win_length=1024,
    hop_length=256,
    f_min=0,
    f_max=8000,
    n_mels=40,
)

def trans(wav):
    wav, _ = torchaudio.load(wav)
    wav = nn.functional.pad(wav, (0, 16000 - wav.shape[1]))
    wav = wav.squeeze_()
    freq_masker = torchaudio.transforms.FrequencyMasking(1)
    time_masker = torchaudio.transforms.TimeMasking(1, True)
    noiser = distributions.Normal(0, 0.1)
    noisy_wav = wav
    mel_spec = mel_spectrogramer(noisy_wav)
    return time_masker(freq_masker(torch.log(mel_spec + 1e-8)))

X_train = torch.stack(list(train_data['wav'].progress_apply(lambda x : trans(x))))
y_train = torch.tensor(train_data['class'])
X_val = torch.stack(list(val_data['wav'].progress_apply(lambda x : trans(x))))
y_val = torch.tensor(val_data['class'])

train_dataset = TensorDataset(X_train, y_train)
val_dataset = TensorDataset(X_val, y_val)

train_dataloader = DataLoader(
            train_dataset,
            batch_size=512,
            num_workers=8,
            shuffle=True
        )

val_dataloader = DataLoader(
            val_dataset,
            batch_size=512,
            num_workers=8,
            shuffle=True
        )

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class Transpose(nn.Module):
    def __init__(self, dim0, dim1):
        super().__init__()
        self.dim0 = dim0
        self.dim1 = dim1
    
    def forward(self, x):
        #print(x)
        return torch.transpose(x, self.dim0, self.dim1)

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.enc = nn.Sequential(
            nn.Conv1d(40, 64, 20, 1, 10),
            Transpose(0, 2),
            Transpose(1, 2)
        )
        self.gru = nn.GRU(64, 128)

    def forward(self, x, h=None):
        
        if h == None:
          return self.gru(self.enc(x))
        else:
          return self.gru(self.enc(x), h)

class Attention(nn.Module):
    def __init__(self):
        super(Attention, self).__init__()
        self.attention_mechanism = nn.Sequential(
            nn.Linear(128, 64),
            nn.Softmax(dim=-1),
            nn.Tanh(),
            nn.Linear(64, 1),
            nn.Softmax(dim=1)
        )

    def forward(self, h):
        return self.attention_mechanism(h)

class MyKWS(nn.Module):
    def __init__(self):
        super(MyKWS, self).__init__()
        self.U = nn.Linear(128, 2)
        self.p = nn.Softmax(dim=-1)
        self.enc = Encoder()
        self.att = Attention()

    def forward(self, x, hid=None):
        h, hid = self.enc(x,hid)
        h = torch.transpose(h, 0, 1)
        att = self.att(h)
        c = torch.bmm(torch.transpose(h, 1, 2), att).squeeze()
        return self.p(self.U(c)), hid

model=MyKWS().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
criterion = nn.CrossEntropyLoss()

def FA_FR_rate(p, y_true):
    total_acc = (y_true == 1).sum()
    total_rej = (y_true == 0).sum()
    if total_acc == 0 or total_rej == 0:
        return np.array([0.0, 1.0]), np.array([1.0, 0.0])
    y_true = y_true[np.argsort(p)]
    fr = np.cumsum(y_true == 1) / total_acc
    fr = np.concatenate([[0.0], fr])
    fa = 1.0 - np.cumsum(y_true == 0) / total_rej
    fa = np.concatenate([[1.0], fa])
    return fr, fa

from sklearn.metrics import auc

def train(model, train_dataloader, criterion, optimizer, epoch, total):
    model.train()
    train_loss = 0
    correct = 0
    False_Acceptance = 0
    False_Rejection = 0
    avg_auc = 0
    FRR = 0
    for batch_idx, (specs, y_true) in enumerate(train_dataloader):
        specs = specs.to(device)
        y_true = y_true.to(device)
        optimizer.zero_grad()
        output, _ = model(specs)
        y_pred = torch.argmax(output, dim=1)
        correct += (y_pred == y_true).sum().item()
        for i in range(y_true.shape[0]):
          if (y_pred[i] == 1 and y_true[i] == 0):
            False_Acceptance += 1
          elif (y_pred[i] == 0 and y_true[i] == 1):
            False_Rejection += 1
        loss = criterion(output, y_true)
        train_loss += loss.item()
        loss.backward()
        optimizer.step()
        FRr, FAr = FA_FR_rate(output[:, 1].detach().cpu().numpy(), y_true.cpu().numpy())
        avg_auc += auc(FRr, FAr)
        FRR += FRr[np.sum(FAr > 1/3600)]
    return train_loss / len(train_dataloader), correct / total, False_Acceptance / total, False_Rejection / total + 1e-9, avg_auc / len(train_dataloader), FRR / len(train_dataloader)

for epoch in range(1, 50):
  train_total = len(train_dataloader.dataset)
  train_loss, train_acc, train_FA, train_FR, train_FRFA_AUC, train_FRR = train(model, train_dataloader, criterion, optimizer, epoch, train_total)
checkpoint = {'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()}
torch.save(checkpoint, 'kws.pth')