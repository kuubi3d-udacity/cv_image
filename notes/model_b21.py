import torch
import torch.nn as nn
from torch.nn import functional as F
import torchvision.models as models
from torch.nn.utils.rnn import pack_padded_sequence
from heapq import heappush, heappop
import numpy as np

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
    
    def beam_search(self, inputs, beam_width=3, max_len=20):
        start = torch.tensor([[self.vocab_size-2]])  # Index of "<start>"

        print('start', start, 'vocab size', self.vocab_size)
        start_word = [(start, 0.0)]

        print(start_word)

        for _ in range(max_len):
            temp = []
            for s in start_word:
                seq, score = s
                embeddings = self.embed(seq)
                hiddens, _ = self.lstm(embeddings)
                outputs = self.linear(hiddens[:, -1, :])
                probabilities = F.log_softmax(outputs.squeeze(1), dim=1)  # Adjusted dimension
                top_scores, top_words = torch.topk(probabilities, beam_width, dim=1)  # Updated dimension
                top_scores = top_scores.squeeze(0)
                top_words = top_words.squeeze(0)

                for i in range(beam_width):
                    word = top_words[0][i].unsqueeze(0)
                    score = top_scores[0][i] + score
                    temp.append((torch.cat((seq, word), dim=1), score))

            start_word = sorted(temp, key=lambda x: x[1], reverse=True)[:beam_width]

        end_word = start_word[0][0].squeeze(0).tolist()
        caption = [idx2word[word_idx] for word_idx in end_word]
        caption = ' '.join(caption[1:])  # Exclude "<start>"
        return caption
    '''   
    def sample(self, inputs, states=None, max_len=20):
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
        sampled_ids = []
        #inputs = inputs.unsqueeze(1)
        for i in range(max_len):                                    # maximum sampling length
            hiddens, states = self.lstm(inputs, states)        # (batch_size, 1, hidden_size), 
            outputs = self.linear(hiddens.squeeze(1))
            #print(outputs)# (batch_size, vocab_size)
            predicted = outputs.max(1)[1]
            #print('predicted',predicted)
            #print(predicted.argmax())
            sampled_ids.append(predicted.tolist()[0])
            inputs = self.embed(predicted)
            inputs = inputs.unsqueeze(1)                       # (batch_size, 1, embed_size)
        #print('sampled_ids',sampled_ids)
        #print(sampled_ids.squeeze())
        #sampled_ids = torch.cat(sampled_ids, 1)                # (batch_size, 20)
        return sampled_ids#.squeeze()
    '''