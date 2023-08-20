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
        hiddens, _ = self.lstm(embeddings)
        outputs = self.linear(hiddens)
        return outputs

    def beam_search(self, features, k, max_len, states=None):
        #inputs = features.unsqueeze(1)  # Adjust inputs for the first step
        inputs = features
        candidates = []
        start_token = 0
        end_token = 1

        score = torch.tensor([0.0]).to(inputs.device)
        #caption = torch.tensor([[start_token]]).to(inputs.device)  # Initialize with 2D tensor
        caption = torch.tensor([start_token]).to(inputs.device)


        hiddens, states = self.lstm(inputs, states)
        beams = [(score, caption)]

        for _ in range(max_len):
            new_beams = []
            for beam in beams:
                score, partial_caption = beam

                #if partial_caption[0][-1].item() == end_token:
                if partial_caption[-1] == end_token:
                    candidates.append((score, partial_caption))
                    continue

                hiddens, states = self.lstm(inputs, states)  # Use the partial caption for LSTM input
                caption = self.linear(hiddens.squeeze(1))
                top_scores, top_indices = caption.topk(k)


                predicted = caption.argmax(1)
                print("predicted", predicted)
                print("candidates", candidates)
                candidates.append(predicted.tolist()[0])
                inputs = self.embed(predicted)
                inputs = inputs.unsqueeze(1)

                for i in range(k):
                    new_score = score + top_scores[0][i]
                    #new_caption = torch.cat((partial_caption, top_indices[0][i].unsqueeze(0).unsqueeze(0)), dim=1)
                    new_caption = torch.cat((partial_caption, top_indices[0][i].unsqueeze(0)))
                    new_beams.append((new_score, new_caption))

            beams = sorted(new_beams, key=lambda x: x[0], reverse=True)[:k]

        # Extract the best caption from beams and convert it to a list of integers
        best_caption = beams[0][1][0].squeeze().tolist()
        #return best_caption
        return candidates
