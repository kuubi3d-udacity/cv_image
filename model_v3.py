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

        """
        Used caption_scores instead of overwriting the caption tensor in the loop.
        Fixed the comparison of the last token with end_token.
        Simplified the extraction of the top_candidate using the max function with a custom key function.
        
        """
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

                if partial_caption[-1].item() == end_token.item():
                #if partial_caption[-1] == end_token:
                    candidates.append((score, partial_caption.tolist()))
                    continue

                hiddens, states = self.lstm(inputs, states)
                caption_scores = self.linear(hiddens.squeeze(1))
                top_scores, top_indices = caption_scores.topk(k)
                
                for i in range(k):
                    new_score = score + top_scores[0][i]
                    new_caption = torch.cat((partial_caption, top_indices[0][i].unsqueeze(0)))
                    new_beams.append((new_score, new_caption))
                
                print("partial_caption[-1].item()", partial_caption[-1].item())
                print("end_token.item()", end_token.item())
                print("top_indices[0]", top_indices[0])

            beams = sorted(new_beams, key=lambda x: x[0], reverse=True)[:k]

        for beam in beams:
            score, partial_caption = beam
            if partial_caption[-1].item() != end_token.item():
                candidates.append((score, partial_caption.tolist()))

        top_candidate = max(candidates, key=lambda x: x[0])[1]
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
