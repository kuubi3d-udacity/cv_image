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

    def beam_search(self, features, start_token, end_token, k, max_len, states=None):
        inputs = features.unsqueeze(1)  # Add a time step dimension
        candidates = []

        score = torch.tensor([0.0]).to(inputs.device)
        token = torch.tensor([start_token]).to(inputs.device)
        weights = torch.tensor([0.0]).to(inputs.device)

        beams = [(score, token, inputs, states)]  # Removed unnecessary unpacking
        beam, path, node, next_node, new_beam = [],[],[],[],[]

        for _ in range(max_len):
            #beam, path, next_node, new_beam = []
            embedded_token = self.embed(torch.tensor([start_token]).to(inputs.device))

            for weights, tier, inputs, states in beams:
                if tier[-1].item() == end_token:
                    candidates.append((weights, tier.tolist()))
                    continue

                embedded_token = self.embed(tier[-1].unsqueeze(0).unsqueeze(0)) 
                hiddens, states = self.lstm(embedded_token, states)
                scores = self.linear(hiddens.squeeze(1))
                top_scores, top_indices = scores.topk(k)
                #print("embedded_token", embedded_token)

                for i in range(k):
                    k_node = top_indices[0][i].unsqueeze(0)
                    weights = top_scores[0][i].unsqueeze(0)
                    next_node = torch.cat((weights, k_node))
                    #node.append(next_node)

                    node = node + next_node.tolist()


                    #path = path + (next_node[0].tolist()), (next_node[1].tolist())
                    #new_inputs = torch.cat((inputs, embedded_token.unsqueeze(0)), dim=1)
                    #new_beam.append((weights, new_beam, new_inputs, states))
            beam.append(node)
            new_beam = beam

            print('node', node)
            print('path', path)
            print('beam', beam)
            print('new_beam', new_beam)
            #new_beam.sort(key=lambda x: x[0], reverse=True)
            break
            beams = new_beam[:k]
            print('beams', beams)
        
        for score, token, _, _ in beams:
            if token[-1].item() != end_token:
                candidates.append((score, token.tolist()))
        
        candidates.sort(key=lambda x: x[0], reverse=True)
        top_score, top_caption = candidates[0]
        return top_caption




    def sample(self, features, k, states=None, max_len=20):
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
            print("outputs", outputs)
            print("predicted", predicted)
            #print("sample_ids", sampled_ids)
 
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
