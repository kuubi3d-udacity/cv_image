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
        input = features.unsqueeze(1)  # Add a time step dimension
        candidates = []

        score = torch.tensor([0.0]).to(input.device)
        token = torch.tensor([start_token]).to(input.device)

        beams = [(score, token, input, states)]
        next_beam = []
        for _ in range(max_len):
            
            for scores, tokens, _, _ in beams:
            #for score, tokens, input, states in beams:
                if token[-1].item() == end_token:
                    candidates.append((score, tokens.tolist()))
                    continue

                hiddens, states = self.lstm(input.squeeze(0).squeeze(0), states)
                scores = self.linear(hiddens.squeeze(1))
                print("scores", scores)
                top_scores, top_indices = scores.topk(k)
                #next_beam.append((top_scores))
                #embedded_token = self.embed(tokens[-1].unsqueeze(0).unsqueeze(0).unsqueeze(0))

                print("next beam", next_beam)
                #print("top scores", top_scores, top_indices)

                for i in range(k):
                    next_score = top_scores[0][i].unsqueeze(0)
                    next_node = top_indices[0][i].unsqueeze(0)
                    print("next_score", next_score)
                    next_word = torch.cat((next_score, next_node))
                    #next_inputs = torch.cat((input, embedded_token), dim=1)
                    #next_beam.append((next_score, next_word, input, states))
                    next_beam.append((next_score, next_node))
                    
                    #print("next beam", next_beam[0][1])


            next_beam.sort(key=lambda x: x[0], reverse=True)
            input = self.embed(next_beam)
            beams = next_beam
        
        for score, token, _, _ in beams:
            if token[-1].item() != end_token:
                candidates.append((score, token.tolist()))
        
        top_score, top_caption = candidates[0]
        top_candidate = top_caption
        print("output", top_candidate)
        
        return top_candidate
        


    def sample(self, features, k, states=None, max_len=20):
        # Original pseudo-code line 3: Walk over each step-in sequence

        #inputs = features.unsqueeze(1)
        inputs = features
        sampled_ids = []
        #print("inputs", inputs)
        
        for _ in range(max_len):
            hiddens, states = self.lstm(inputs, states)
            outputs = self.linear(hiddens.squeeze(1))
            print("outputs", outputs)
            predicted = outputs.argmax(1)           
            sampled_ids.append(predicted.tolist()[0])
            inputs = self.embed(predicted)
            inputs = inputs.unsqueeze(1)

            print("inputs", inputs)
            
            #print("predicted", predicted)
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