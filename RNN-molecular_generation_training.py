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

		# LSTM layer: pad first -> lstm -> unpad
		x1 = tc.nn.utils.rnn.pack_padded_sequence(x1, x_len, batch_first=True, enforce_sorted=False) # 
		x1, next_hidden = self.lstm(x1, hidden)
		x1, out_len = tc.nn.utils.rnn.pad_packed_sequence(x1, batch_first=True) # total_length=max_seq_len

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

	def train_step(self, inp, target):
		"""train step for a x-y sequence pair"""

		hidden = self.init_hidden(1) # batch_size = 1
 
		self.optimizer.zero_grad()
		loss = 0
		
		output, hidden = self.forward(inp, hidden)
		loss = self.loss_fn(output, target)

		loss.backward()
		self.optimizer.step()

		return loss.item()

	def fit(self, dataloader, epochs, print_freq):
		"""model training function"""

		# start = time.time()
		loss_avg = 0.0
		losses = []

		for epoch in range(epochs):
			# loop over batches
			for step, (x_padded, y_padded, x_lens, y_lens) in enumerate(dataloader):

				inp = x_padded.to(self.device)
				target = y_padded.to(self.device)

				# hidden
				hidden = self.init_hidden(x_padded.shape[0])
				pred, hidden = self(inp, x_lens, hidden) # pred in [batch_size, seq_len, nb_tags]
				# print(pred.shape)
				# print(target.shape) # target in [batch_size, tag idx]
			
				# sum up loss
				batch_loss = 0.0
				for i in range(pred.size(0)):
					ce_loss = F.cross_entropy(pred[i], target[i], ignore_index=49)
					batch_loss += ce_loss
				# loss = self.loss_fn(pred.view(pred.shape[0]*pred.shape[1], pred.shape[2]), target.view(-1))
				loss_avg += batch_loss.item() # total loss of all samples
				
				self.optimizer.zero_grad()
				batch_loss.backward()
				self.optimizer.step()

			losses.append(loss_avg/len(dataX)) # average loss of a sample			
			if epoch % print_freq == 0:
				print("Epoch [{}/{}], Loss {:.4e}".format(epoch+1, epochs, loss_avg/len(dataX)))

			loss_avg = 0.0

		return losses

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

def pad_collate(batch):
	"""pad list of tc.tensor with variable length into equal length batch"""

	xx, yy = zip(*batch)

	x_lens = [len(x) for x in xx]
	y_lens = [len(y) for y in yy]

	xx_pad = pad_sequence(xx, batch_first=True, padding_value=49)
	yy_pad = pad_sequence(yy, batch_first=True, padding_value=49)

	return xx_pad, yy_pad, x_lens, y_lens

##########################################################		

if __name__ == '__main__':
	import sys

	# define the parameters
	n_neurons = 128
	n_layers = 2
	vocab_size = 50
	embedding_dim = 20
	lr = 1e-3
	epochs = 500
	print_freq = 1
	Train_samples = 1000
	batch_size = 100 

#################################################################################
	# data preprocess: load the data
	dataX = np.load("dataX_unpadded.npy", allow_pickle=True)
	dataY = np.load("dataY_unpadded.npy", allow_pickle=True)

	# form train data: create dataloader
	dataset = [[tc.Tensor(dataX[i]).long(), tc.Tensor(dataY[i]).long()] for i in range(dataX.shape[0])]	
	dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=pad_collate)

################################################################################	

	# create a model
	RNNmodel = RNN(n_neurons, n_layers, embedding_dim)
	RNNmodel.to(RNNmodel.device)

	# train
	losses = RNNmodel.fit(dataloader, epochs, print_freq)
	
	# save loss
	np.savetxt("loss.txt", np.asarray(losses))

	# save model
	tc.save(RNNmodel.state_dict(), "RNNmodel.pt")

	# load model and evaluate
	bestModel = RNN(n_neurons, n_layers, embedding_dim)
	bestModel.load_state_dict(tc.load("RNNmodel.pt"))
	print(bestModel)
	# total parameters
	print(sum([param.nelement() for param in bestModel.parameters()]))

	# # generation
	# Ntargets = 1000
	# NgeneratedSMILES = []
	# for i in range(Ntargets):
	# 	smi = bestModel.generation(tokens, max_seq_len=150)
	# 	NgeneratedSMILES.append(smi[1:-1])

	# # write
	# with open('GneratedSMILES.txt', 'w') as fo:
	# 	for i in range(Ntargets):
	# 		fo.write(NgeneratedSMILES[i])
	# 		fo.write("\n")

#################################################################################	
	# # test
	# batch_size = 4
	# x = dataX[0:batch_size]
	# x = tc.Tensor(x).long() #.unsqueeze(-1) # [batch_size, seq_len, features]

	# y_true = dataY[0:batch_size]
	# y_true = tc.Tensor(y_true).long() #.unsqueeze(-1)
	# # print(y_true.shape) #[4,307]
	
	# hidden = RNNmodel.init_hidden(batch_size)
	# y_pred, newhidden = RNNmodel(x, X_len[0:batch_size], hidden)
	# # print(y_pred.shape) #[4, 10, 49]
	# y_pred = F.log_softmax(y_pred, dim=2)
	# # print(y_pred) #[4, 10, 49]
	# # sys.exit()

	# # flatten y_pred and y_true
	# y_pred = y_pred.view(-1, RNNmodel.nb_labels)
	# y_true = y_true.view(-1)
	# # print(y_true[:100])

	# tag_pad_token = RNNmodel.char2int['<PAD>'] # 49
	# # print(tag_pad_token)
	# mask = (y_true < tag_pad_token).float()
	# # print(mask) # 1228

	# # count how many tokens
	# nb_tokens = int(tc.sum(mask).data) # 31
	# # print(nb_tokens)

	# # select the values that are not PAD and zeros out the rest
	# y_pred = y_pred[range(y_pred.shape[0]), y_true] * mask
	# print(y_pred)

	# sys.exit()

	# # loss
	# loss_fn = nn.CrossEntropyLoss() # = log_softmax() + NLLLoss()
	# # The input is expected to contain raw, unnormalized scores for each class.
	# # pred in [batch_size, Classes]
	# # target in [classes[batch_size]]
	# loss = loss_fn(y_pred.view(y_pred.shape[1], y_pred.shape[2]), y_true.view(-1))
	# print(loss)

	# sys.exit()
#################################################################################	

	