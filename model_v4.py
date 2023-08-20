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
        inputs = features
        candidates = []
        start_token = 0
        end_token = 1

        # Initialize the initial beam with a dummy caption and score
        initial_caption = torch.tensor([[start_token]]).to(inputs.device)
        initial_score = torch.tensor([0.0]).to(inputs.device)
        initial_beam = (initial_score, initial_caption)

        beams = [initial_beam]

        for _ in range(max_len):
            new_beams = []
            for score, partial_caption in beams:
                if partial_caption[0][-1] == end_token:
                    # If the current partial caption ends, add it to candidates
                    candidates.append((score, partial_caption.tolist()))
                    continue

                embeddings = self.embed(partial_caption)
                hiddens, states = self.lstm(embeddings, states)
                caption_scores = self.linear(hiddens.squeeze(1))
                top_scores, top_indices = caption_scores.topk(k)
                
                for i in range(k):
                    new_score = score + top_scores[0][i]
                    new_caption = torch.cat((partial_caption, top_indices[0][i].unsqueeze(0).unsqueeze(0)), dim=1)
                    #new_caption = torch.cat((partial_caption, top_indices[0][i].unsqueeze(0)))
                    new_beams.append((new_score, new_caption))

            # Sort and keep the top-k beams
            beams = sorted(new_beams, key=lambda x: x[0], reverse=True)[:k]

        # Extract the best caption from the beams
        best_caption = beams[0][1]

        # Remove the start token and return the predicted caption
        predicted_caption = best_caption[0][1:]

        return predicted_caption.squeeze().tolist()