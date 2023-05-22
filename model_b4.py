import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F

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
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1, beam_width=3):
        super(DecoderRNN, self).__init__()
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        self.beam_width = beam_width

        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)

    def forward(self, features, captions):
        embeddings = self.embed(captions[:, :-1])
        embeddings = torch.cat((features.unsqueeze(1), embeddings), 1)
        hiddens, _ = self.lstm(embeddings)
        outputs = self.linear(hiddens)
        return outputs

    def beam_search(self, inputs, states=None, max_len=20):
        inputs = inputs.unsqueeze(1)
        batch_size = inputs.size(0)

        # Initialize the beam search
        beam = [(inputs, states, [], 0)]  # (input, hidden state, output sequence, cumulative log probability)

        for _ in range(max_len):
            candidates = []
            for input_, state, sequence, log_prob in beam:
                lstm_output, lstm_state = self.lstm(input_, state)
                output = self.linear(lstm_output.squeeze(1))
                log_probs = F.log_softmax(output, dim=1)

                top_log_probs, top_indices = log_probs.topk(self.beam_width, dim=1)
                top_log_probs = top_log_probs.squeeze() + log_prob

                for i in range(self.beam_width):
                    candidate_log_prob = top_log_probs[i].item()
                    candidate_word = top_indices[:, i]
                    candidate_sequence = sequence + [candidate_word]
                    candidates.append((lstm_output, lstm_state, candidate_sequence, candidate_log_prob))

            candidates.sort(key=lambda x: x[3], reverse=True)
            beam = candidates[:self.beam_width]

        return beam[0][2]
        #return sampled_ids.tolist()
