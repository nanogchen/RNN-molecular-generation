#!/usr/bin/env python
# author: GC @ 01/07/2020 RNN model for SMILES generation

import torch as tc
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

tokens = ['<', '>', '#', '%', '(', ')', '*', '+', '-', '/', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '=', 'B', 'C', 'F', 'G', 'H', 'I', 'K', 'L', 'N', 'O', 'P', 'S', 'T', 'Z', '[', '\\', ']', 'a', 'b', 'c', 'd', 'e', 'i', 'l', 'n', 'o', 'r', 's', '<PAD>']
# char2int = dict((c, i) for i,c in enumerate(tokens))
# int2char = dict((i, c) for i,c in enumerate(tokens))

class RNN(nn.Module):
	def __init__(self, n_neurons, n_layers, embedding_dim, lr=1e-3):
		super().__init__()
		
		self.n_neurons = n_neurons
		self.n_layers = n_layers
		self.embedding_dim = embedding_dim
		self.nb_labels = len(tokens)-1
		self.char2int = dict((c, i) for i,c in enumerate(tokens))	
		self.int2char = dict((i, c) for i,c in enumerate(tokens))
		self.device = tc.device('cuda' if tc.cuda.is_available() else 'cpu')
		
		# embedding layer: pad all sequence to have the same length
		padding_idx = self.char2int['<PAD>']
		self.embedding = nn.Embedding(num_embeddings=len(tokens), embedding_dim=self.embedding_dim) #, padding_idx=padding_idx)
		
		# lstm layer [features, n_hidden, n_layers]
		self.lstm = nn.LSTM(self.embedding_dim, self.n_neurons, self.n_layers, batch_first=True) # for batch training: batch_first = True
		
		# final layer: predict the final char
		self.fc = nn.Linear(self.n_neurons, self.nb_labels)

		# loss func
		self.loss_fn = nn.CrossEntropyLoss()		

		# optimizer
		# self.optimizer = tc.optim.Adam(self.parameters(), lr=lr) # has to be placed below the layers
		self.optimizer = tc.optim.AdamW(self.parameters(),lr=0.001,betas=(0.9, 0.999),eps=1e-08,weight_decay=0.001,amsgrad=False)
	
	def forward(self, x, x_len, hidden):
		"""Forward pass: char to char level prediction model
		x: [batch_size, padded_seq_len, features];
		x_len: length of each unpadded sample;
		"""

		batch_size, seq_len = x.shape
		
		# embedding layer
		x1 = self.embedding(x) # [batch_size, seq_len, embedded_dim]
		# print(x1[0])

		# LSTM layer: pad first -> lstm -> unpad
		x1 = tc.nn.utils.rnn.pack_padded_sequence(x1, x_len, batch_first=True, enforce_sorted=False) # 
		# print(x1[0])
		x1, next_hidden = self.lstm(x1, hidden)
		# print(x1[0])
		x1, out_len = tc.nn.utils.rnn.pad_packed_sequence(x1, batch_first=True) # total_length=max_seq_len
		# print(x1[0]) # [batch_size, max_len, hidden_neurons]
		# sys.exit()

		# forward connected layer
		out = self.fc(x1)
		
		return out, next_hidden

	def init_hidden(self, batch_size, device='gpu'):
		""" initialize hidden state"""
		# if self.use_cuda:

		if device == 'gpu':
			return (tc.zeros(self.n_layers, batch_size, self.n_neurons).to(self.device),\
				tc.zeros(self.n_layers, batch_size, self.n_neurons).to(self.device))

		else:
			return (tc.zeros(self.n_layers, batch_size, self.n_neurons),\
				tc.zeros(self.n_layers, batch_size, self.n_neurons))

	def generation(self, tokens, max_seq_len=150):
		"""generation step on CPU"""

		# start with '<'
		start = tokens[0]
		newSMILES = ''.join(start)

		# prepare the inp for generation
		inp = tc.tensor(self.char2int[start]).view(1,-1)

		# hidden
		hidden = self.init_hidden(1, device='cpu')

		for timestep in range(max_seq_len):
			# in generation use the following rather than output, hidden = self(inp, hidden) # output in [1,1,50]
			x1 = self.embedding(inp)
			x1, hidden = self.lstm(x1, hidden)
			output = self.fc(x1)

			# use softmax to get the most probable idx
			max_prob = tc.softmax(output, dim=2).view(-1) # [1,1]
			top_idx = tc.multinomial(max_prob, 1)[0].cpu().numpy() # has to be set for random generation
			# print(max_prob)

			# get the most chars
			pred_char = self.int2char[int(top_idx)]
			# print(pred_char)

			# update SMILES
			newSMILES += pred_char

			# update inp
			inp = tc.tensor(self.char2int[pred_char]).view(1,-1)

			# determine the length
			if pred_char == tokens[1]: # end token '>'
				break

		return newSMILES

def saveData(SMI_list, idx):

	filename = "generated_SMILES" + str(idx) + ".txt"
	Nsamples = len(SMI_list)

	# write
	with open(filename, 'w') as fo:
		for i in range(Nsamples):
			fo.write(SMI_list[i])
			fo.write("\n")


##########################################################		

if __name__ == '__main__':
	import sys

	# define the parameters
	n_neurons = 128
	n_layers = 2
	vocab_size = 50
	embedding_dim = 20
	lr = 1e-3
	epochs = 50
	print_freq = 1
	Train_samples = 1000
	batch_size = 100 

	# load model and evaluate
	bestModel = RNN(n_neurons, n_layers, embedding_dim)
	bestModel.load_state_dict(tc.load("RNNmodel.pt"))
	print(bestModel)

	# total parameters
	print(sum([param.nelement() for param in bestModel.parameters()])) # 216217 parameters

	# parameters of generation
	Ntargets = 1000000
	Nwrite = 10000 # output data every Nwrite frequency
	NgeneratedSMILES = []

	# generate
	for i in range(Ntargets):
		smi = bestModel.generation(tokens, max_seq_len=150)
		NgeneratedSMILES.append(smi[1:-1])

		if (i+1) % Nwrite == 0:
			saveData(NgeneratedSMILES, int((i+1)/Nwrite))
			NgeneratedSMILES = []