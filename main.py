import json

from torch import optim
import torch
import torch.nn as nn
from chatbot import *
from dataloader import get_training_data

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

max_target_len = 10
batch_size = 4

model = ChatBot(hidden_size=256, vocab_size=68215, output_size=68215, n_layers=1, dropout=0).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)
encoder_hidden = torch.zeros(2, 4,256).to(device)
decoder_input  = torch.zeros(1, 4, 256).to(device)
decoder_hidden = torch.zeros(1, 4, 256).to(device)
X_train, Y_train, vocab_words = get_training_data(batch_size=4)
for epoch in range(0, 600000):
    for iteration in range(X_train.shape[0]):
        input_variable = torch.tensor(X_train[iteration]).transpose(1,0).long().to(device)
        global_target = torch.tensor(Y_train[iteration]).long().to(device)
        encoder_outputs, encoder_hidden = model.encoder(input_variable, encoder_hidden)
        optimizer.zero_grad()
        loss_sum = 0
        loss = 0
        full_answer = None
        for t in range(global_target.shape[1]):
            decoder_output, decoder_hidden = model.decoder(decoder_input, decoder_hidden, encoder_outputs)
            _, topi = decoder_output.topk(1)
            decoder_input = torch.LongTensor([[topi[i][0] for i in range(batch_size)]]).to(device)
            decoder_hidden = decoder_hidden.detach()
            encoder_hidden = encoder_hidden.detach()
            target = global_target[:, t]
            if full_answer is None:
                full_answer = topi
            else:
                full_answer = torch.cat((full_answer,topi), dim=1)

            loss += criterion(decoder_output, target)
        qna = []

        if epoch % 1000:
            for i in range(batch_size):
                print('Answer was ' + ' '.join([vocab_words[word.item()] for word in full_answer[i]]))
                print('Question was ' + ' '.join([vocab_words[word.item()] for word in input_variable[i]]))
                print("Loss this epoch is {}".format(loss.item()/4))
        loss.backward()
        optimizer.step()