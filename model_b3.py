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

    def beam_search(self, features, beam_width=5, max_len=20):
        # Prepare the initial inputs for beam search
        batch_size = features.size(0)
        inputs = features.unsqueeze(1)
        states = None

        # Initialize the beam search
        top_beam = [(inputs, states, [], 0)]  # (inputs, states, current_tokens, log_prob)
        completed_beams = []

        # Start the beam search
        for _ in range(max_len):
            all_candidates = []
            for curr_inputs, curr_states, curr_tokens, curr_log_prob in top_beam:
                if len(curr_tokens) > 0 and curr_tokens[-1] == 1:  # Check if the last token is the end token
                    completed_beams.append((curr_tokens, curr_log_prob))
                    continue

                lstm_output, new_states = self.lstm(curr_inputs, curr_states)
                output = self.linear(lstm_output.squeeze(1))

                # Apply log softmax to calculate the log probabilities
                log_probs = torch.log_softmax(output, dim=1)

                # Get the top beam_width candidates
                top_log_probs, top_tokens = log_probs.topk(beam_width, dim=1)

                for i in range(beam_width):
                    token = top_tokens[:, i].unsqueeze(1)
                    log_prob = top_log_probs[:, i].item()

                    new_inputs = self.embed(token)
                    new_inputs = new_inputs.unsqueeze(1)

                    all_candidates.append((new_inputs, new_states, curr_tokens + [token], curr_log_prob + log_prob))

            # Select the top beam_width candidates for the next iteration
            sorted_candidates = sorted(all_candidates, key=lambda x: x[3], reverse=True)
            top_beam = sorted_candidates[:beam_width]

        if len(completed_beams) > 0:
            completed_beams.sort(key=lambda x: x[1], reverse=True)
            sampled_ids = completed_beams[0][0]
        else:
            sampled_ids = top_beam[0][2]

        return sampled_ids
