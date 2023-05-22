import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn.utils.rnn import pack_padded_sequence
from heapq import heappush, heappop


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
    
    def forward(self, features, captions):
        embeddings = self.embed(captions[:, :-1])
        embeddings = torch.cat((features.unsqueeze(1), embeddings), 1)
        hiddens, _ = self.lstm(embeddings)
        outputs = self.linear(hiddens)
        return outputs

    def beam_search(self, inputs, states=None, max_len=20):
        "Accepts pre-processed image tensor (inputs) and returns predicted sentence using beam search."
        batch_size = inputs.size(0)
        device = inputs.device
        
        # Expand inputs to match beam width
        inputs = inputs.unsqueeze(1).expand(batch_size, self.beam_width, self.embed_size)
        
        # Initialize beams
        beam_scores = torch.zeros(batch_size, self.beam_width).to(device)
        beam_seqs = torch.zeros(batch_size, self.beam_width, max_len).long().to(device)
        beam_seqs[:, :, 0] = inputs.squeeze(1)
        beam_hiddens = None
        
        for t in range(1, max_len):
            if t > 1:
                inputs = beam_seqs[:, :, t-1].unsqueeze(2)
            
            # Perform one step of LSTM
            hiddens, states = self.lstm(inputs, states)  # (batch_size, beam_width, hidden_size)
            outputs = self.linear(hiddens.squeeze(1))  # (batch_size, beam_width, vocab_size)
            
            # Calculate scores for each beam
            scores = outputs.log_softmax(dim=2)  # (batch_size, beam_width, vocab_size)
            scores = beam_scores.unsqueeze(2) + scores  # (batch_size, beam_width, vocab_size)
            
            # Reshape scores for topk calculation
            reshaped_scores = scores.view(batch_size, -1)  # (batch_size, beam_width * vocab_size)
            
            # Perform beam search
            topk_scores, topk_indices = reshaped_scores.topk(self.beam_width, dim=1)  # (batch_size, beam_width)
            
            # Calculate beam indices and token indices
            beam_indices = topk_indices // self.vocab_size  # (batch_size, beam_width)
            token_indices = topk_indices % self.vocab_size  # (batch_size, beam_width)
            
            # Update beam scores
            beam_scores = topk_scores
            
            # Update beam sequences
            beam_seqs[:, :, :t] = beam_seqs[beam_indices, :, :t]
            beam_seqs[:, :, t] = token_indices
            
            # Update hidden states for next step
            beam_hiddens = (hiddens[beam_indices, :, :], states)
        
        # Return the sequences with the highest scores
        best_seqs = beam_seqs[:, 0, :].tolist()  # (batch_size, max_len)
        return best_seqs
