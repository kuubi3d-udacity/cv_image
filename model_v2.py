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

    def beam_search(self, inputs, k, max_length):
        device = inputs.device
        batch_size = inputs.size(0)

        # Expand inputs to match beam width
        inputs = inputs.unsqueeze(1).expand(batch_size, k, *inputs.shape[1:])

        sequences = [(inputs, torch.tensor([[1]], device=device))]  # Start token
        sequence_scores = torch.zeros(batch_size, 1, device=device)

        for step in range(max_length):
            all_candidates = []

            for sequence in sequences:
                image_features = sequence[0]  # Extract image features from the sequence
                captions = sequence[1]  # Extract partial captions from the sequence
                embeddings = self.embed(captions[:, :-1])

                print('image_features.size() = ', image_features.size())
                print('embeddings.size() = ', embeddings.size())

                # Assuming image_features has shape [batch_size, channels, height]
                image_features = image_features.view(batch_size * k, *image_features.shape[2:])
                image_features = image_features.unsqueeze(1)  # Add an extra dimension

                # Assuming embeddings has shape [batch_size, sequence_length, embedding_size]
                embeddings = embeddings.view(batch_size * k, *embeddings.shape[1:])

                print('image_features.size() = ', image_features.size())
                print('embeddings.size() = ', embeddings.size())

                embeddings = torch.cat((image_features.repeat(1, embeddings.shape[1], 1), embeddings), dim=2)

                hiddens, _ = self.lstm(embeddings)
                outputs = self.linear(hiddens)

                predictions = torch.log_softmax(outputs[:, -1, :], dim=1)  # Apply log-softmax to get scores

                top_scores, top_indices = torch.topk(predictions, k=k, dim=1)
                top_scores = sequence_scores.view(batch_size, k, 1).expand_as(top_scores) + top_scores

                candidates = (sequence[0].view(batch_size, k, *sequence[0].shape[2:]),
                            sequence[1].view(batch_size, k, -1))
                candidates = (torch.cat([candidates[0], image_features], dim=1),
                            torch.cat([candidates[1], top_indices.view(batch_size, k, 1)], dim=2))
                all_candidates.append((candidates, top_scores))

            all_candidates = sorted(all_candidates, key=lambda x: x[1].view(-1), reverse=True)
            sequences = [cand[0] for cand in all_candidates[:k]]
            sequence_scores = torch.cat([cand[1] for cand in all_candidates[:k]], dim=2)

        top_predictions = sequences[0]
        return top_predictions

'''
# Example usage
encoder = EncoderCNN(embed_size=256)
decoder = DecoderRNN(embed_size=256, hidden_size=512, vocab_size=1000)

# Obtain the embedded image features.
image = torch.randn(1, 3, 224, 224)  # Example image tensor
features = encoder(image).unsqueeze(1)

# Pass the embedded image features through the model to get a predicted caption using beam search.
beam_width = 3
max_length = 20

top_predictions = decoder.beam_search(features, beam_width, max_length)

# Print the top predictions
print(top_predictions)
'''