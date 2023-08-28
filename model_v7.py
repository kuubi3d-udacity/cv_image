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
        caption = torch.tensor([start_token]).to(inputs.device)

        beams = [(score, caption, inputs, states)]

        for _ in range(max_len):
            new_beams = []
            for score, partial_caption, inputs, states in beams:
                if partial_caption[-1].item() == end_token:
                    candidates.append((score, partial_caption.tolist()))
                    continue

                embedded_token = self.embed(partial_caption[-1].unsqueeze(0))
                hiddens, states = self.lstm(embedded_token, states)
                caption_scores = self.linear(hiddens.squeeze(1))
                top_scores, top_indices = caption_scores.topk(k)

                for i in range(k):
                    predicted = top_indices[0][i].unsqueeze(0)
                    new_score = score + top_scores[0][i]
                    new_caption = torch.cat((partial_caption, predicted))
                    new_inputs = torch.cat((inputs, embedded_token), dim=1)
                    new_beams.append((new_score, new_caption, new_inputs, states))

            new_beams.sort(key=lambda x: x[0], reverse=True)
            beams = new_beams[:k]

        for score, partial_caption, _, _ in beams:
            if partial_caption[-1].item() != end_token:
                candidates.append((score, partial_caption.tolist()))

        top_score, top_caption = candidates[0]
        top_candidate = top_caption

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
            predicted = outputs.argmax(1)           
            sampled_ids.append(predicted.tolist()[0])
            inputs = self.embed(predicted)
            inputs = inputs.unsqueeze(1)

            #print("inputs", inputs)
            #print("outputs", outputs)
            #print("predicted", predicted)
            #print("sample_ids", sampled_ids)
 
        return sampled_ids

