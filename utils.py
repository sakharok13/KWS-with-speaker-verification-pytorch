#utils like get_centroids etc
def get_centroids(embeddings):
  return embeddings.mean(dim=1)

def get_centroids_prior(embeddings):
  centroids = []
  for speaker in embeddings:
    centroid = 0
    for utterance in speaker:
      centroid = centroid + utterance 
    centroid = centroid / len(speaker)
    centroids.append(centroid)
  centroids = torch.stack(centroids)
  return centroids

def get_centroid(embeddings, speaker_num, utterance_num):
  centroid = 0 
  
  for utterance_id, utterance in enumerate(embeddings[speaker_num]):
    if utterance_id == utterance_num:
      continue
    centroid = centroid + utterance 
  centroid = centroid/(len(embeddings[speaker_num])-1)

  return centroid
  
  .......... to be continued 
