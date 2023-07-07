import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn.utils.rnn import pack_padded_sequence
from heapq import heappush, heappop

class BeamNode:
    def __init__(self, hidden, caption, log_prob, length):
        self.hidden = hidden
        self.caption = caption
        self.log_prob = log_prob
        self.length = length

    def __lt__(self, other):
        # Define the comparison between nodes for the heap
        return self.log_prob > other.log_prob


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
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1, beam_width=4):
        super(DecoderRNN, self).__init__()
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        self.beam_width = beam_width

        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)

    def forward(self, features, captions, beam_search=False, beam_width=None):
        if beam_search:
            beam_width = self.beam_width if beam_width is None else beam_width
            return self._beam_search(features, beam_width)
        else:
            return self._forward(features, captions)

    def _forward(self, features, captions):
        embeddings = self.embed(captions[:, :-1])
        embeddings = torch.cat((features.unsqueeze(1), embeddings), 1)
        hiddens, _ = self.lstm(embeddings)
        outputs = self.linear(hiddens)
        return outputs

    def _beam_search(self, features, beam_width):
        batch_size = features.size(0)
        hidden = self.init_hidden(batch_size)

        start_token = torch.tensor([[1]] * batch_size, dtype=torch.long, device=features.device)  # Start token is 1
        end_token = 2  # End token is 2
        
          # Check the number of dimensions in features tensor
        if features.dim() < 3:
            # If features tensor has fewer than 4 dimensions, add additional dimensions
            features = features.unsqueeze(1).unsqueeze(2)  # Shape: (batch_size, 1, 1, ...)
        else:
            # Convert features to a tensor if it is not already
            features = torch.tensor(features)

        # Expand features to match beam width
        features = features.expand(batch_size, 1, *features.shape[2:])

        beams = [[] for _ in range(batch_size)]
        for i in range(batch_size):
            node = BeamNode(hidden[:, i, :].unsqueeze(1), start_token[i], 0.0, 0)
            beams[i].append(node)

        for _ in range(captions.size(1) - 1):  # -1 because we don't need to predict the last token
            new_beams = [[] for _ in range(batch_size)]
            for i in range(batch_size):
                for node in beams[i]:
                    if node.caption.item() == end_token or node.length >= captions.size(1):
                        new_beams[i].append(node)
                        continue

                    embeddings = self.embed(node.caption)
                    embeddings = torch.cat((features[i].unsqueeze(0).unsqueeze(1), embeddings.unsqueeze(1)), dim=1)
                    hiddens, hidden = self.lstm(embeddings, node.hidden)
                    outputs = self.linear(hiddens.squeeze(1))
                    log_probs = torch.log_softmax(outputs, dim=1)
                    top_log_probs, top_indices = log_probs.topk(beam_width, dim=1)

                    for j in range(beam_width):
                        next_log_prob = top_log_probs[0, j].item()
                        next_word = top_indices[0, j].unsqueeze(0)

                        new_node = BeamNode(hidden, torch.cat((node.caption, next_word), dim=1), node.log_prob + next_log_prob, node.length + 1)
                        new_beams[i].append(new_node)

            beams = new_beams

        # Select the best caption from each beam
        captions = torch.zeros(batch_size, captions.size(1), dtype=torch.long, device=features.device)
        for i in range(batch_size):
            beams[i].sort()
            captions[i] = beams[i][0].caption.squeeze(0)

        return captions

    def init_hidden(self, batch_size):
        hidden = torch.zeros(self.num_layers, batch_size, self.hidden_size)
        return hidden, hidden


# Example usage
embed_size = 256
hidden_size = 512
vocab_size = 10000

encoder = EncoderCNN(embed_size)
decoder = DecoderRNN(embed_size, hidden_size, vocab_size, num_layers=1, beam_width=4)

# Use beam search
features = torch.randn(8, embed_size)  # Example input features
captions = torch.randint(0, vocab_size, (8, 10))  # Example input captions
output = decoder(features, captions, beam_search=True)

print(output)
