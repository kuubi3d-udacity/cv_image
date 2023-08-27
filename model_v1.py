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
        embeddings = self.embed(captions[:,:-1])
        embeddings = torch.cat((features.unsqueeze(1), embeddings), 1)
        #packed = pack_padded_sequence(embeddings, lengths, batch_first=True) 
        hiddens, _ = self.lstm(embeddings)
        outputs = self.linear(hiddens)
        return outputs

    def beam_search(self, features, k, max_len, states=None):
        inputs = features
        candidates = []
        
        start_token = torch.tensor([0]).to(inputs.device) 
        end_token = torch.tensor([1]).to(inputs.device)
        
        score = torch.tensor([0.0]).to(inputs.device)
        caption = torch.tensor([start_token]).to(inputs.device)
        beams = [(score, caption)]  
        
        for _ in range(max_len):
            new_beams = []
            for beam in beams:
                score, partial_caption = beam
                
                if partial_caption[-1].item() == end_token[0].item():
                #if partial_caption[-1] == end_token:
                    candidates.append((score, partial_caption.tolist()))
                    continue

                hiddens, states = self.lstm(inputs, states)
                caption_scores = self.linear(hiddens.squeeze(1))              
                top_scores, top_indices = caption_scores.topk(k)
                
                predicted = caption_scores.argmax(1)
                inputs = self.embed(predicted)
                inputs = inputs.unsqueeze(1)

                for i in range(k):
                    new_score = score + top_scores[0][i]
                    new_caption = torch.cat((partial_caption, top_indices[0][i].unsqueeze(0)))
                    new_beams.append((new_score, new_caption))
                    #candidates.append((new_caption.tolist()))
                    #candidates.append((score, partial_caption.tolist()))
                
                print("predicted", predicted)
                print('cadidates', candidates)
            
            beams = sorted(new_beams, key=lambda x: x[0], reverse=True)[:k]
            print("beam", beam)

        # Add any remaining beams that reach the maximum length
        for beam in beams:
            score, partial_caption = beam
            if partial_caption[-1] != end_token:
                candidates.append((score, partial_caption.tolist()))
        
            #top_candidate = beam[1]
        top_candidate = candidates[0][1]
        #top_candidate = beam[1].tolist()
        print("candidates",candidates)
        #print("top_candidate", candidates[0][1])    
        return top_candidate


    def sample(self, features, k, states=None, max_len=20):
        # Original pseudo-code line 3: Walk over each step-in sequence

        #inputs = features.unsqueeze(1)
        inputs = features
        sampled_ids = []
        for _ in range(max_len):
            hiddens, states = self.lstm(inputs, states)
            outputs = self.linear(hiddens.squeeze(1))
            predicted = outputs.argmax(1)
            print("predicted", predicted)
            sampled_ids.append(predicted.tolist()[0])
            inputs = self.embed(predicted)
            inputs = inputs.unsqueeze(1)
            print("sample_ids", sampled_ids)

            return sampled_ids


"""Input:
features: input features
k: maximum beam size
max_len: maximum hypothesis length
states: optional states for LSTM

1: B0 ← { (0.0, [<sos>]) }
2: for t ∈ {1, . . . , max_len}:
3:    B ← ∅
4:    for (score, partial_caption) in Bt-1:
5:        if partial_caption.last().item() = end_token:
6:            B.add((score, partial_caption))
7:            continue
8:        hiddens, states ← lstm(inputs, states)
9:        caption_scores ← linear(hiddens.squeeze(1))
10:       top_scores, top_indices ← caption_scores.topk(k)
11:       predicted ← caption_scores.argmax(1)
12:       inputs ← embed(predicted)
13:       inputs ← inputs.unsqueeze(1)
14:       for i ∈ {1, . . . , k}:
15:           new_score ← score + top_scores[0][i]
16:           new_caption ← concatenate(partial_caption, [top_indices[0][i]])
17:           B.add((new_score, new_caption))
18:   Bt ← B.top(k)
19: return B.max()
"""