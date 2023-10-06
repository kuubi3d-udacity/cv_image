import torch
import torch.nn as nn
import torchvision.models as models

class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        return features

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super(DecoderRNN, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)
    
    def forward(self, features, captions):
        embeddings = self.embed(captions[:, :-1])
        embeddings = torch.cat((features.unsqueeze(1), embeddings), 1)
        hiddens, _ = self.lstm(embeddings)
        outputs = self.linear(hiddens)
        return outputs


    def beam_search(self, features, start_token, end_token, lstm_states, k, max_len):
      
        #linear_layer = torch.nn.Linear(256, 512).to(device)
        #features = linear_layer(encoder(image).unsqueeze(0))
        batch_size=features.size(2)

        #'''
        # Initialize LSTM states with image features
        lstm_states = (
            features.squeeze(0),  # Initial hidden state (hx)
            torch.zeros(1, 1, batch_size).to(features.device).squeeze(0) # Initial cell state (cx)
            )
        #'''

        #lstm_states = None
    
        inputs = features # Add a time step dimension
        beams = [(torch.tensor([start_token]).to(inputs.device), inputs, lstm_states, 0)] * batch_size
  
        for _ in range(max_len):
            new_beams = []

            for (tokens, lstm_states, beam_scores), _ in zip(beams, range(batch_size)):

                if tokens[-1] == end_token:
                    new_beams.append((tokens, lstm_states, beam_scores))
                    continue
          
                #embeddings = self.embed(torch.tensor([tokens[-1]]).unsqueeze(0).to(features.device))
                embed_token = self.embed(torch.tensor([tokens[-1]]).to(features.device))
                hiddens, lstm_states = self.lstm(embed_token, lstm_states)
                scores = self.linear(hiddens.squeeze(1))
                top_scores, top_indices = scores.topk(k)

                for i in range(k):
                    next_token = top_indices[0][i].item()
                    next_score = top_scores[0][i].item()
                    new_score = beam_scores + next_score

                    new_tokens = tokens + [next_token]
                    new_beams.append((new_tokens, lstm_states, new_score))

            # Sort beams based on new scores and keep the top-k beams
            beams = sorted(new_beams, key=lambda x: x[0], reverse=True)[:k]

        #'''
        # Extract the best captions for each batch element
        caption_list = [beam[2] for beam in beams]

        best_caption = [sublist for sublist in caption_list[0]]
        return best_caption
        #'''

        '''
        caption = [beam[2] for beam in beams]
        best_caption = [token for tokens in caption for token in tokens]

        return best_caption
        #'''
        
        
        '''
        best_caption = max(beams, key=lambda x: x[0])[1]
        return best_caption
        #'''

        '''
        # Extract best captions for each batch element
        caption_list = [max(beams[i * k: (i + 1) * k], key=lambda x: x[0])[2] for i in range(batch_size)]
        print('list',caption_list)
        best_captions = [captions for sublist in caption_list for captions in sublist]

        return best_captions
        #'''

        


    def sample(self, features, states=None, max_len=20):
        # Original pseudo-code line 3: Walk over each step-in sequence

        #inputs = features.unsqueeze(1)
        inputs = features
        sampled_ids = []
        #print("inputs", inputs)
        
        for _ in range(max_len):
            hiddens, states = self.lstm(inputs, states)
            outputs = self.linear(hiddens.squeeze(1))
            predicted = outputs.argmax(1)           
            sampled_ids.append(predicted.tolist()[0])
            inputs = self.embed(predicted)
            inputs = inputs.unsqueeze(1)

            #print("inputs", inputs)
            #print("outputs", outputs)
            #print("predicted", predicted)
            print("sample_ids", sampled_ids)
 
        return sampled_ids
    

"""
Input:
features: input features
k: maximum beam size
max_len: maximum hypothesis length
states: optional states for LSTM

1: B0 ← { (0.0, [<sos>]) }
2: for t ∈ {1, . . . , max_len}:
3:    B ← ∅
4:    for (score, token) in Bt-1:
5:        if token.last().item() = end_token:
6:            B.add((score, token))
7:            continue
8:        hiddens, states ← lstm(inputs, states)
9:        caption_scores ← linear(hiddens.squeeze(1))
10:       top_scores, top_indices ← caption_scores.topk(k)
11:       predicted ← caption_scores.argmax(1)
12:       inputs ← embed(predicted)
13:       inputs ← inputs.unsqueeze(1)
14:       for i ∈ {1, . . . , k}:
15:           new_score ← score + top_scores[0][i]
16:           new_caption ← concatenate(token, [top_indices[0][i]])
17:           B.add((new_score, new_caption))
18:   Bt ← B.top(k)
19: return B.max()
"""

"""
Algorithm 1 Standard beam search4
Input: x: source sentence
k : maximum beam size
nmax: maximum hypothesis length
score(·, ·): scoring function
1: B0 ← {0, BOS }
2: for t ∈ {1, . . . , nmax −1} :
3: B ← ∅
4: for s, y ∈ Bt−1 :
5: if y.last() = EOS :
6: B.add(s, y)
7: continue
8: for y ∈ V :
9: s ← score(x, y ◦ y)
10: B.add(s, y ◦ y)
11: Bt ← B.top(k)
12: return B.max()
"""