import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.functional import softmax

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
        packed = pack_padded_sequence(embeddings, lengths, batch_first=True) 
        hiddens, _ = self.lstm(packed)
        outputs = self.linear(hiddens)
        return outputs

def beam_search(features, beam_size, max_length, device):
    start_token = torch.tensor([[1]], device=device)  # Assuming the start token has index 1
    end_token = torch.tensor([[2]], device=device)  # Assuming the end token has index 2
    
    features = features.repeat(beam_size, 1)  # Repeat the image features beam_size times
    captions = start_token.repeat(beam_size, 1)  # Repeat the start token beam_size times
    
    beam_scores = torch.zeros(beam_size, 1, device=device)  # Initialize beam scores
    finished_captions = []
    
    for _ in range(max_length):
        embeddings = decoder.embed(captions[:, :-1])
        embeddings = torch.cat((features.unsqueeze(1), embeddings), 1)
        hiddens, _ = decoder.lstm(embeddings)
        outputs = decoder.linear(hiddens[:, -1])
        log_probs = torch.log(softmax(outputs, dim=2))
        
        scores = beam_scores.repeat(1, vocab_size) + log_probs.squeeze(1)
        scores = scores.view(-1, beam_size * vocab_size)
        
        top_scores, top_indices = scores.topk(beam_size, dim=1)
        captions = captions.repeat(1, vocab_size).view(beam_size, -1)
        captions = torch.cat((captions[:, :captions.shape[1] - 1], top_indices.unsqueeze(2)), dim=2)
        
        end_indices = (captions[:, -1] == end_token.squeeze()).nonzero()
        if end_indices.size(0) > 0:
            for idx in end_indices.squeeze():
                caption = captions[idx]
                score = beam_scores[idx]
                finished_captions.append((caption, score))
                beam_scores[idx] = -float('inf')
        
        beam_scores = top_scores.view(-1, 1)
        captions = captions.view(beam_size, -1)
    
    finished_captions.sort(key=lambda x: x[1], reverse=True)
    best_caption = finished_captions[0][0]
    
    return best_caption.squeeze()

# Example usage
embed_size = 256
hidden_size = 512
vocab_size = 10000
num_layers = 1
beam_size = 5
max_length = 20
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

encoder = EncoderCNN(embed_size).to(device)
decoder = DecoderRNN(embed_size, hidden_size, vocab_size, num_layers).to(device)

# Assume you have image features in the 'features' tensor
features = torch.randn(1, embed_size).to(device)

# Get the best caption using beam search
best_caption = beam_search(decoder, features, beam_size, max_length, device)

print(best_caption)


'''
## Beam search implementation (Attempt)
def beam_search_sample(self, inputs, beam=3):
    output = []
    batch_size = inputs.shape[0] # batch_size is 1 at inference, inputs shape : (1, 1, embed_size)
    hidden = self.lstm(batch_size) # Get initial hidden state of the LSTM
    
    # sequences[0][0] : index of start word
    # sequences[0][1] : probability of the word predicted
    # sequences[0][2] : hidden state related of the last word
    sequences = [[[torch.Tensor([0])], 1.0, hidden]]
    max_len = 20

    ## Step 1
    # Predict the first word <start>
    outputs, hidden = DecoderRNN.get_outputs(self, inputs, hidden)
    _, max_indice = torch.max(outputs, dim=1) # predict the most likely next word, max_indice shape : (1)
    output.append(max_indice.cpu().numpy()[0].item()) # storing the word predicted 
    # inputs = DecoderRNN.get_next_word_input(self, max_indice)
    
    
    l = 0
    while len(sequences[0][0]) < max_len: 
        print("l:", l)
        l+= 1
        temp = []
        for seq in sequences:
#                 print("seq[0]: ", seq[0])
            inputs = seq[0][-1] # last word index in seq
            inputs = inputs.type(torch.cuda.LongTensor)
            print("inputs : ", inputs)
            # Embed the input word
            inputs = self.embed(inputs) # inputs shape : (1, embed_size)
            inputs = inputs.unsqueeze(1) # inputs shape : (1, 1, embed_size) 
            
            # retrieve the hidden state
            hidden = seq[2]
            
            preds, hidden = DecoderRNN.get_outputs(self, inputs, hidden)

            # Getting the top <beam_index>(n) predictions
            softmax_score = F.log_softmax(outputs, dim=1) # Define a function to sort the cumulative score
            sorted_score, indices = torch.sort(-softmax_score, dim=1)
            word_preds = indices[0][:beam]
            best_scores = sorted_score[0][:beam]

            # Creating a new list so as to put them via the model again
            for i, w in enumerate(word_preds):
#                     print("seq[0]: ", seq[0][0][:].cpu().numpy().item())
                next_cap, prob = seq[0][0].cpu().numpy().tolist(), seq[1]
                
                next_cap.append(w)
                print("next_cap : ", next_cap)
                prob *best_scores[i].cpu().item()
                temp.append([next_cap, prob])

        sequences = temp
        # Order according to proba
        ordered = sorted(sequences, key=lambda tup: tup[1])

        # Getting the top words
        sequences = ordered[:beam]
        print("sequences: ", sequences)

def get_outputs(self, inputs, hidden):
    lstm_out, hidden = self.lstm(inputs, hidden) # lstm_out shape : (1, 1, hidden_size)
    outputs = self.linear(lstm_out)  # outputs shape : (1, 1, vocab_size)
    outputs = outputs.squeeze(1) # outputs shape : (1, vocab_size)

    return outputs, hidden

def get_next_word_input(self, max_indice):
    ## Prepare to embed the last predicted word to be the new input of the lstm
    inputs = self.embed(max_indice) # inputs shape : (1, embed_size)
    inputs = inputs.unsqueeze(1) # inputs shape : (1, 1, embed_size)

    return inputs


def sample(self, inputs, states=None, max_len=20):
    " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
    sampled_ids = []
    #inputs = inputs.unsqueeze(1)
    for i in range(20):                                    # maximum sampling length
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
