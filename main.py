from torch import optim

from chatbot import *
max_target_len = 10
batch_size = 4

model = ChatBot(hidden_size=128, vocab_size=100, output_size=100, n_layers=1, dropout=0)

criterion = nn.CrossEntropyLoss()
optimizer_encoder = optim.SGD(model.parameters(), lr=0.01)
decoder_input  = torch.zeros(1,4,128)
decoder_hidden = torch.zeros(1,4,128)
input_variable = torch.zeros((100, 4)).long()

for epoch in range(0,300):
    encoder_outputs, encoder_hidden = model.encoder(input_variable, torch.zeros((2, 4,128)))
    for t in range(max_target_len):
        decoder_output, decoder_hidden = model.decoder(decoder_input, decoder_hidden, encoder_outputs)
        _, topi = decoder_output.topk(1)
        decoder_input = torch.LongTensor([[topi[i][0] for i in range(batch_size)]])
        print("It worked")
