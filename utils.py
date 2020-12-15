import torch 
import torchaudio
import random
def pad_tensor(tensor, max_length):
  # input tensor (1, n) --> (1, max_length)
  n = tensor.size(1)
  zeros = torch.zeros(1, max_length)
  zeros[:, :n] = tensor
  return zeros

def padding_batch(sequences):
    """
    sequences is a list of tensors
    """
    num = len(sequences)
    max_len = max([s.size(1) for s in sequences])
    
    padded_batch = torch.cat([pad_tensor(i, max_len) for i in sequences])
    return padded_batch


"""
getting batch for lstm input randomly choosing from different spaekers directories 
"""

class get_batch(object):
  def __init__(self, path, N, M):
    self.N = N 
    self.path = path
    self.M = M

    self.speakers = [i[len(path):] for i in glob.glob(path + '/*')]

  def sampler(self):
    batch_speakers = [] 
    for k in range(self.N):

      speaker = random.choice(self.speakers)
      newpath = self.path + speaker
      readings = [r for r in glob.glob(newpath + '/*')] #utterances of a certain speaker in librispeech

      for sp in range(self.M):

        reading = random.choice(readings)
        utterance = random.choice([ut for ut in glob.glob(reading + '/*.flac')])
        audio = torchaudio.load(utterance)[0]
        batch_speakers.append(audio)

    return batch_speakers




        
