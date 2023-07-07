
import torch
import torch.nn as nn
import torchvision.models as models
#from torch.nn.utils.rnn import pack_padded_sequence
#from heapq import heappush, heappop


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


class BeamSearchNode:
    def __init__(self, hidden_state, previous_word_idx, sequence, log_prob, length):
        self.hidden_state = hidden_state
        self.previous_word_idx = previous_word_idx
        self.sequence = sequence
        self.log_prob = log_prob
        self.length = length

    def __lt__(self, other):
        return self.log_prob < other.log_prob


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
    '''
    def beam_search(image, max_len,  beam_index = 3):
        start = [word2idx["<start>"]]
        
        # start_word[0][0] = index of the starting word
        # start_word[0][1] = probability of the word predicted
        start_word = [[start, 0.0]]
        
        while len(start_word[0][0]) < max_len:
            temp = []
            for s in start_word:
                par_caps = sequence.pad_sequences([s[0]], maxlen=max_len, padding='post')
                e = encoding_test[image[len(images):]]
                preds = final_model.predict([np.array([e]), np.array(par_caps)])
                
                # Getting the top <beam_index>(n) predictions
                word_preds = np.argsort(preds[0])[-beam_index:]
                
                # creating a new list so as to put them via the model again
                for w in word_preds:
                    next_cap, prob = s[0][:], s[1]
                    next_cap.append(w)
                    prob += preds[0][w]
                    temp.append([next_cap, prob])
                        
            start_word = temp
            # Sorting according to the probabilities
            start_word = sorted(start_word, reverse=False, key=lambda l: l[1])
            # Getting the top words
            start_word = start_word[-beam_index:]
        
        start_word = start_word[-1][0]
        intermediate_caption = [idx2word[i] for i in start_word]

        final_caption = []
        
        for i in intermediate_caption:
            if i != '<end>':
                final_caption.append(i)
            else:
                break
        
        final_caption = ' '.join(final_caption[1:])
        return final_caption
    '''