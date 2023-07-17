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
        predicted = []
        start_token = 0
        end_token = 1


        hiddens, states = self.lstm(inputs, states)
        caption = self.linear(hiddens.squeeze(1))
        score = torch.tensor([0.0]).to(inputs.device)
        beams = [(score, caption)]  # (score, caption)
        print('beams', beams)

        for _ in range(max_len):
            new_beams = []
            for beam in beams:
                score, partial_caption = beam
                last_caption = partial_caption[-1]
                print('score', score.tolist())
                print('last_caption', last_caption)
                
                if partial_caption[-1] == end_token:
                    predicted.append((score, partial_caption.tolist()))
                    print('score', score.tolist())
                    print('p_caption', partial_caption.tolist())
                    continue
                
                hiddens, states = self.lstm(inputs, states)
                caption = self.linear(hiddens.squeeze(1))
                top_scores, top_indices = torch.topk(caption, k)

                print('top', top_scores, top_indices[0][2])#, top_indices)

                for i in range(k):
                    new_score = score + top_scores[0][i]
                    new_caption = torch.cat((partial_caption, top_indices[0][i].unsqueeze(0).unsqueeze(0)), dim=1)
                    new_beams.append((new_score, new_caption))

            beams = sorted(new_beams, key=lambda x: x[0], reverse=True)[:k]


            #print('inputs', inputs)
            #print('beams', beam)
            #print('beams', beam)
            print('score', score.tolist())
            print('caption', caption.tolist())
            print('predicted', predicted)
            

        #predicted = sorted(predicted, key=lambda x: x[0], reverse=True)
        top_candidate = sorted(predicted, key=lambda x: x[0], reverse=True)
        #top_candidate = predicted[0][1]

        print('top candidate', top_candidate)
        print('predicted', predicted)
        #print('beams', beams.tolist())

        return top_candidate

    

    def sample(self, features, k, max_len, states=None):

        inputs = features
        #inputs = features.unsqueeze(1)
        candidates = []
    
        for _ in range(max_len):

            hiddens, states = self.lstm(inputs, states)
            caption = self.linear(hiddens.squeeze(1))
            predicted = caption.argmax(1)
            candidates.append(predicted.tolist()[0])
            inputs = self.embed(predicted)
            inputs = inputs.unsqueeze(1)
            
            print('caption', caption.tolist())
            print('predicted', predicted.tolist())
            print('candidates', candidates)
           

        return candidates
    
        
            #predicted = outputs.argmax(1)
            #candidates.append(predicted.tolist()[0])
            #inputs = self.embed(predicted)
            #inputs = inputs.unsqueeze(1)
                  
    
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
