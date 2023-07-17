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
        start_token = 0
        end_token = 1




        score = torch.tensor([0.0]).to(inputs.device)
        #caption = torch.tensor([start_token]).to(inputs.device)

        hiddens, states = self.lstm(inputs, states)
        caption = self.linear(hiddens.squeeze(1))
        beams = [(score, caption)]  # (score, caption)

        for _ in range(max_len):
            new_beams = []
            for beam in beams:
                score, partial_caption = beam
                if partial_caption[-1].item() == end_token[0].item():
                    candidates.append((score, partial_caption.tolist()))
                    continue

                hiddens, states = self.lstm(inputs, states)
                caption = self.linear(hiddens.squeeze(1))
                top_scores, top_indices = caption.topk(k)
                #inputs = self.embed(torch.Tensor(candidates))
                #inputs = inputs.unsqueeze(1)


                for i in range(k):
                    new_score = score + top_scores[0][i]
                    new_caption = torch.cat((partial_caption, top_indices[0][i].unsqueeze(0).unsqueeze(0)), dim=1)
                    new_beams.append((new_score, new_caption))

                #print('inputs', inputs)
                #print('beams', beam)
                #print('beams', beam)
                print('score', score.tolist())
                print('caption', caption.tolist())
                #print('predicted', predicted)
                print('cadidates', candidates)


            beams = sorted(new_beams, key=lambda x: x[0], reverse=True)[:k]

        # Add any remaining beams that reach the maximum length
        for beam in beams:
            score, partial_caption = beam
            if partial_caption[-1] != end_token[0].item():
                candidates.append((score, partial_caption.tolist()))

        top_candidate = candidates[0][1]

        return candidates

ValueError: only one element tensors can be converted to Python scalars
    '''
    def beam_search(self, features, k, max_len, states=None):

        inputs = features
        #inputs = features.unsqueeze(1)
        candidates = []
        start_token = 0
        end_token = 1

        score = torch.tensor([0.0]).to(inputs.device)
        caption = torch.tensor([start_token]).to(inputs.device)
        beams = [score, caption]  # (score, caption)

        for _ in range(max_len):

            hiddens, states = self.lstm(inputs, states)
            caption = self.linear(hiddens.squeeze(1))
            predicted = caption.argmax(1)
            candidates.append(predicted.tolist()[0])
            inputs = self.embed(predicted)
            inputs = inputs.unsqueeze(1)

        return candidates
    
        
            #predicted = outputs.argmax(1)
            #candidates.append(predicted.tolist()[0])
            #inputs = self.embed(predicted)
            #inputs = inputs.unsqueeze(1)
            
            print('score', score.tolist())
            print('caption', caption.tolist())
            print('predicted', predicted.tolist())
            print('candidates', candidates)
            #print('beams', beams.tolist())
        
            for score, caption in beams:
                if caption[-1] == end_token:
                    # If the caption ends with the end token, add it as a candidate
                    candidates.append((score, caption))
                    continue

                embeddings = self.embed(caption)
                #inputs = torch.cat((features.unsqueeze(1), embeddings(1)), 1)
                #hiddens, _ = self.lstm(inputs)
                #outputs = self.linear(hiddens)
                
                top_scores, top_indices = torch.topk(outputs.squeeze(1), k)
                for s, idx in zip(top_scores, top_indices):
                    candidate_score = score + s.item()
                    candidate_caption = torch.cat((caption, idx.unsqueeze(0)))
                    candidates.append((candidate_score, candidate_caption))

            candidates.sort(reverse=True, key=lambda x: x[0])
            beams = candidates[:k]

        top_predictions = [caption for score, caption in beams]
        return top_predictions
        '''
        
    
    '''
    def sample(self, features, k, states=None, max_len=20):
        # Original pseudo-code line 3: Walk over each step-in sequence

        #inputs = features.unsqueeze(1)
        inputs = features
        sampled_ids = []
        for _ in range(max_len):
            hiddens, states = self.lstm(inputs, states)
            outputs = self.linear(hiddens.squeeze(1))
            predicted = outputs.argmax(1)
            sampled_ids.append(predicted.tolist()[0])
            inputs = self.embed(predicted)
            inputs = inputs.unsqueeze(1)

        return sampled_ids
    '''


    '''
    def sample(self, inputs, k, states=None, max_len=20):
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
        sampled_ids = []
        for i in range(20):                                    # maximum sampling length
            hiddens, states = self.lstm(inputs, states)        # (batch_size, 1, hidden_size), 
            outputs = self.linear(hiddens.squeeze(1))
            predicted = outputs.max(1)[1]
            sampled_ids.append(predicted.tolist()[0])
            inputs = self.embed(predicted)
            inputs = inputs.unsqueeze(1)                       # (batch_size, 1, embed_size)
       
        #sampled_ids = torch.cat(sampled_ids, 1)                # (batch_size, 20)
        return sampled_ids#.squeeze()
    '''
