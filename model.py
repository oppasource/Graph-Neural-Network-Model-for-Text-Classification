import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import pdb


class vanila_lstm_classifier(nn.Module):
	def __init__(self, embed_size, nHidden, nClasses):
		super(vanila_lstm_classifier, self).__init__()
		self.embed_size = embed_size
		self.nHidden = nHidden

		self.lstm = nn.LSTM(embed_size, nHidden, bidirectional = True)
		self.out_linear = nn.Linear(2 * nHidden, nClasses)


	def forward(self, in_seq):
		in_seq = in_seq.view(-1, 1, self.embed_size)
		recurrent, (hidden, c) = self.lstm(in_seq)
		hidden = hidden.view(-1, 2*self.nHidden)

		out = self.out_linear(hidden)
		out = out.view(1,-1)

		return out


class ggnn(nn.Module):
	def __init__(self, embed_size, nClasses):
		super(ggnn, self).__init__()
		self.embed_size = embed_size
		# self.nHidden = nHidden

		# self.lstm = nn.LSTM(embed_size, nHidden, bidirectional = True)
		self.Wa = nn.Linear(embed_size, embed_size)
		self.gru = nn.GRU(embed_size, embed_size)

		self.f1 = nn.Linear(embed_size, embed_size)
		self.f2 = nn.Linear(embed_size, embed_size)

		self.sigmoid = nn.Sigmoid()
		self.tanh = nn.Tanh()

		self.out_layer = nn.Linear(embed_size, nClasses)


	def forward(self, in_seq, adjmat):
		# GGNN Layer 1
		at = self.Wa(torch.matmul(adjmat, in_seq))
		at = at.view(1, -1, self.embed_size)
		in_seq = in_seq.view(1, -1, self.embed_size)
		out_seq, hn = self.gru(in_seq, at)

		# # GGNN Layer 2
		# at = self.Wa(torch.matmul(adjmat, out_seq))
		# at = at.view(1, -1, self.embed_size)
		# in_seq = out_seq.view(1, -1, self.embed_size)
		# out_seq, hn = self.gru(in_seq, at)

		# # GGNN Layer 3
		# at = self.Wa(torch.matmul(adjmat, out_seq))
		# at = at.view(1, -1, self.embed_size)
		# in_seq = out_seq.view(1, -1, self.embed_size)
		# out_seq, hn = self.gru(in_seq, at)


		# Readout layer
		out_seq = out_seq.view(-1, self.embed_size)
		hv = self.sigmoid(self.f1(out_seq)) * self.tanh(self.f2(out_seq))
		hG = torch.mean(hv, dim = 0) + torch.max(hv, dim = 0)[0]

		out = self.out_layer(hG)
		out = out.view(1, -1)
		return out
