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
    
    import torch.nn.functional as F

    def beam_search(self, inputs, beam=3):
        output = []
        batch_size = inputs.shape[0]  # batch_size is 1 at inference, inputs shape: (1, 1, embed_size)
        hidden = self.init_hidden(batch_size)  # Get initial hidden state of the LSTM

        # sequences[0][0]: index of start word
        # sequences[0][1]: probability of the word predicted
        # sequences[0][2]: hidden state related to the last word
        sequences = [[[torch.Tensor([0])], 1.0, hidden]]
        max_len = 20

        # Step 1
        # Predict the first word <start>
        outputs, hidden = self.get_outputs(inputs, hidden)
        _, max_indices = torch.max(outputs, dim=1)  # predict the most likely next word, max_indices shape: (1,)
        output.append(max_indices.cpu().numpy()[0].item())  # storing the word predicted

        l = 0
        while len(sequences[0][0]) < max_len:
            print("l:", l)
            l += 1
            temp = []
            for seq in sequences:
                inputs = seq[0][-1]  # last word index in seq
                inputs = inputs.type(torch.cuda.LongTensor)
                print("inputs: ", inputs)
                # Embed the input word
                inputs = self.word_embeddings(inputs)  # inputs shape: (1, embed_size)
                inputs = inputs.unsqueeze(1)  # inputs shape: (1, 1, embed_size)

                # retrieve the hidden state
                hidden = seq[2]

                preds, hidden = self.get_outputs(inputs, hidden)

                # Getting the top <beam_index>(n) predictions
                softmax_score = F.log_softmax(preds, dim=1)
                sorted_score, indices = torch.sort(-softmax_score, dim=1)
                word_preds = indices[0][:beam]
                best_scores = sorted_score[0][:beam]

                # Creating a new list to put them via the model again
                for i, w in enumerate(word_preds):
                    next_cap, prob = seq[0][:], seq[1]
                    next_cap.append(w)
                    print("next_cap: ", next_cap)
                    prob *= best_scores[i].cpu().item()
                    temp.append([next_cap, prob])

            sequences = temp
            # Order according to probability
            ordered = sorted(sequences, key=lambda tup: tup[1], reverse=True)

            # Getting the top words
            sequences = ordered[:beam]
            print("sequences: ", sequences)

    '''
    ## Beam search implementation (Attempt)
    def beam_search_sample(self, inputs, beam=3):
        output = []
        batch_size = inputs.shape[0] # batch_size is 1 at inference, inputs shape : (1, 1, embed_size)
        hidden = self.init_hidden(batch_size) # Get initial hidden state of the LSTM
        
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
                inputs = self.word_embeddings(inputs) # inputs shape : (1, embed_size)
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
    
    def beam_search(self, inputs, states=None, max_len=20, beam_width=5):
        """Accepts pre-processed image tensor (inputs) and performs beam search
        to generate the predicted sentence (list of tensor ids of length max_len)."""
        batch_size = inputs.size(0)

        # Expand inputs and states to beam width
        inputs = inputs.expand(batch_size, beam_width, -1, -1)
        if states is not None:
            states = (states[0].expand(-1, beam_width, -1),
                    states[1].expand(-1, beam_width, -1))

        # Initialize beam search
        start_token = torch.tensor([0])  # Assuming the start token has an ID of 0
        beams = [[(0.0, start_token, states)] for _ in range(batch_size)]

        # Perform beam search
        for _ in range(max_len):
            all_candidates = []
            for beam in beams:
                score, tokens, states = beam[-1]
                tokens = tokens.unsqueeze(0)  # Reshape tokens to (batch_size, sequence_length)
                if tokens[:, -1].item() == 1:  # Assuming the end token has an ID of 1
                    all_candidates.append(beam)
                    continue
                hiddens, states = self.lstm(inputs, states)
                outputs = self.linear(hiddens.squeeze(1))
                scores, preds = outputs.topk(beam_width, dim=1)
                for i in range(beam_width):
                    candidate = (score + scores[:, i], torch.cat([tokens, preds[:, i].unsqueeze(1)], dim=1), states)
                    all_candidates.append(beam + [candidate])

                # Select top beam_width candidates
                top_candidates = sorted(all_candidates, key=lambda x: x[-1][0], reverse=True)[:beam_width]

                # Check for completed sequences
                beams = []
                for candidate in top_candidates:
                    score, tokens, states = candidate[-1]
                    if tokens[:, -1].item() == 1:
                        beams.append(candidate)
                    else:
                        beams.append(candidate[:-1])

                    # Get the best sequence from each beam
                    sampled_ids = [[token.item() for token in beam[-1][1][0]] for beam in beams]

                    return sampled_ids

    '''
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

    '''
    def beam_search(self, features, beam_width, max_len):
        # Prepare the initial inputs for beam search
        batch_size = features.size(0)
        inputs = features.squeeze(1)
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
   
    '''
    '''