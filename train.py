#training
import wandb
from torchaudio.transforms import MelSpectrogram
from torch import optim
import os 
import umap
import torch
#

reducer = umap.UMAP(random_state=42, n_neighbors=7, min_dist=0.1)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

manifold_approx = []
val_loss = []
num_utt = 27629
num_epoch = 30
model = Nnet()
optimizer = optim.Adam(model.parameters())
criterion = GE2ELoss(init_w=10.0, init_b=-5.0, loss_method='softmax')
melspec = MelSpectrogram(n_mels=128, n_fft=400).to(device)
for i in range(num_epoch):
  run = wandb.init(project="ge2e", reinit=True)
  running_loss =0.0
  for k in range(num_utt // 80):
    model.to(device)
    model.train()
    X = get_batch('/content/LibriSpeech/train-clean-100', 8, 10).sampler()
    X = padding_batch(X).to(device)

    X = melspec(X)

    optimizer.zero_grad()
    outputs = model(X).view(8, 10, -1)
    loss = criterion(outputs)
    loss.backward()

    optimizer.step()
    
    running_loss += loss.item()
    if (k + 1) % 20 == 0:
      print(f'Train epoch:{i}, mean loss = {running_loss / 20}')

      wandb.log({"Loss": running_loss / 20})

      running_loss = 0.0
    #there should be 3 different piles of points as val_data has 3 different speakers
    if (k + 1) % 40 == 0:
      model.eval()
      model.to('cpu')
      with torch.no_grad():
        test = model(val_audio)
        reducer.fit(test)
        manifold_approx.append(reducer.fit_transform(test))
        print('Val loss is: ', criterion(model(val_audio).view(3,9,-1)))
        val_loss.append(criterion(model(val_audio).view(3,9,-1)))
  run.finish()
        
torch.save(model.state_dict(), os.path.join(wandb.run.dir, 'model.pt'))
