import time
import torch
import torch.nn as nn
import torch.nn.functional as F

SOS_token = 1


class EncoderRNN(nn.Module):
    def __init__(self, hidden_size, vocab_size, input_lengths, n_layers=1, embedding=None):
        super(EncoderRNN, self).__init__()
        self.input_lengths = input_lengths
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.embedding = embedding

        self.gru = nn.GRU(input_size=self.hidden_size, hidden_size=self.hidden_size, num_layers=self.n_layers,
                          dropout=0,
                          bidirectional=True)

    def forward(self, x, hidden=None):
        embedded = self.embedding(x)
        packed = nn.utils.rnn.pack_padded_sequence(embedded, lengths=torch.tensor([43,43,43,34]),enforce_sorted=False)
        outputs, hidden = self.gru(packed, hidden)
        outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs)
        outputs = outputs[:, :, :self.hidden_size] + outputs[:, :, self.hidden_size:]
        return outputs, hidden


class Attention(nn.Module):
    def __init__(self, attention_dim):
        super(Attention, self).__init__()
        self.W = nn.Linear(attention_dim, attention_dim, bias=False)

    def score(self, decoder_hidden, encoder_out):
        # linear transform encoder out (seq, batch, dim)
        encoder_out = self.W(encoder_out)
        # (batch, seq, dim) | (2, 15, 50)
        encoder_out = encoder_out.permute(1, 0, 2)
        # (2, 15, 50) @ (2, 50, 1)
        return encoder_out @ decoder_hidden.permute(1, 2, 0)

    def forward(self, decoder_hidden, encoder_out):
        energies = self.score(decoder_hidden, encoder_out)
        mask = F.softmax(energies, dim=1)  # batch, seq, 1
        context = encoder_out.permute(
            1, 2, 0) @ mask  # (2, 50, 15) @ (2, 15, 1)
        context = context.permute(2, 0, 1)  # (seq, batch, dim)
        mask = mask.permute(2, 0, 1)  # (target, batch, source)
        return context, mask


class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, vocab_size, output_size, n_layers=1, embedding=None):
        super(DecoderRNN, self).__init__()
        self.output_size = output_size
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.embedding = embedding
        self.gru = nn.GRU(input_size=self.hidden_size, hidden_size=self.hidden_size, num_layers=self.n_layers,
                          dropout=0)

        self.attention = Attention(attention_dim=hidden_size)
        self.concat = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input_step, prev_hidden, encoder_outputs):
        if len(input_step.shape) < 3:
            input_step = self.embedding(input_step)
        output, hidden = self.gru(input_step, prev_hidden)

        context, _ = self.attention(prev_hidden, encoder_outputs)
        output = output.squeeze(0)
        context = context.squeeze(0)

        concat_input = torch.cat((output, context), 1)
        concat_output = torch.tanh(self.concat(concat_input))

        output = self.out(concat_output)
        output = F.softmax(output, dim=1)

        return output, hidden


class ChatBot(nn.Module):
    def __init__(self, hidden_size, vocab_size, output_size, n_layers=1, dropout=0):
        super(ChatBot, self).__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.encoder = EncoderRNN(hidden_size=hidden_size, vocab_size=vocab_size, input_lengths=output_size, n_layers=1,
                                  embedding=self.embedding)
        self.decoder = DecoderRNN(hidden_size=hidden_size, vocab_size=vocab_size, output_size=output_size, n_layers=1,
                                  embedding=self.embedding)

    def forward(self):
        pass
