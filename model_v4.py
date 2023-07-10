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
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.embed_size = embed_size

        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)
    
    def forward(self, features, captions):
        embeddings = self.embed(captions)
        inputs = torch.cat((features.unsqueeze(1), embeddings), 1)
        hiddens, _ = self.lstm(inputs)
        outputs = self.linear(hiddens)
        return outputs

    def beam_search(self, features, start_token, end_token, score, caption, k, max_length):
        # Define sequence of the words in caption
        #beams = [(score, caption)]  # (score, caption)
        
        beams = [(torch.tensor(0.0).to(score.device), [torch.tensor.to(caption.device)])]  # (score, caption)
        # Original pseudo-code line 1: Start
        # Initialize the beam with the start token
        candidates = []
        for _ in range(max_length):
            
            for score, caption in beams:
                if caption[-1] == end_token:
                    # Original pseudo-code line 5: if y.last() = EOS
                    # If the caption ends with the end token, add it as a candidate
                    candidates.append((score, caption))
                    continue
                
                embeddings = self.embed(caption)
                inputs = torch.cat((features.unsqueeze(1), embeddings), 1)
                hiddens, _ = self.lstm(inputs)
                outputs = self.linear(hiddens)

                top_scores, top_indices = torch.topk(outputs.squeeze(1), k)
                for s, idx in zip(top_scores, top_indices):
                    candidate_score = score + s.item()
                    candidate_caption = caption + [idx.item()]
                    candidates.append((candidate_score, candidate_caption))

            candidates.sort(reverse=True, key=lambda x: x[0])
            # Original pseudo-code line 3: Order all the candidates by score
            
            beams = candidates[:k]
            # Original pseudo-code line 4: Select the best k predictions
        
        top_predictions = [caption[1:] for score, caption in beams]
        # Original pseudo-code line 4: Return top k predictions
        return top_predictions


    def sample(self, features, states=None, max_len=20):
        # Original pseudo-code line 3: Walk over each step-in sequence

        inputs = features.unsqueeze(1)
        sampled_ids = []
        for _ in range(max_len):
            hiddens, states = self.lstm(inputs, states)
            outputs = self.linear(hiddens.squeeze(1))
            predicted = outputs.argmax(1)
            sampled_ids.append(predicted.tolist()[0])
            inputs = self.embed(predicted)
            inputs = inputs.unsqueeze(1)

        return sampled_ids



