import pandas as pd 
import pdb
import numpy as np

from nltk.tokenize import word_tokenize

from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import spacy

import model as m


############ Parameters
train_size = 0.8
lr = 0.001
embed_size = 50
hidden_dim = 20
num_class = 2
batch_size = 32
epochs = 100

trained_models_path = './trained_model/'

path2embeddings = '/path/to/glove/glove.6B/glove.6B.' + str(embed_size) + 'd.txt'

np.random.seed(0)

log_file = open('./dep_log.txt', 'w')
log_file.close()

########## helper functions
def loadGloveModel(File):
    f = open(File,'r')
    gloveModel = {}
    for line in f:
        splitLines = line.split()
        word = splitLines[0]
        wordEmbedding = np.array([float(value) for value in splitLines[1:]])
        gloveModel[word] = wordEmbedding
    
    unkembed = np.mean(np.stack(gloveModel.values()), axis=0)
    gloveModel['[unk]'] = unkembed
    return gloveModel


def get_embed(word, glovemodel):
	try:
		embed = glovemodel[word.lower()]
	except:
		embed = glovemodel['[unk]']
	return embed


def get_sliding_adj_mat(tok_sent, window_size = 3):
	num_nodes = len(tok_sent)
	adjmat = np.zeros((num_nodes, num_nodes))

	for row in range(num_nodes):
		for i in range(window_size):
			try:
				adjmat[row][row + i] = 1
			except:
				pass
			try:
				adjmat[row][row - i] = 1
			except:
				pass
	return adjmat


def get_dep_adj_mat(text):
	mat = []
	doc = nlp(text)
	N = 0
	for tok in doc:
		N = N+1

	for word in doc:
		word_mat = [0]*N

		edge_indexes = [c.i for c in word.children] 
		if word.dep_ != 'ROOT':
			edge_indexes = edge_indexes + [word.head.i]

		for e in edge_indexes:
			word_mat[e] = 1

		mat.append(np.array(word_mat))

	return np.array(mat)
	


########### load glove embeddings
glovemodel = loadGloveModel(path2embeddings)


############ Model related
# Using gpu if available else cpu
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')
model = m.ggnn(embed_size, num_class)
model = model.to(device)

# Defining loss function
criterion = nn.CrossEntropyLoss()
# Defining optimizer with all parameters
optimizer = torch.optim.Adam(model.parameters(), lr = lr)

# Initializing spacy
nlp = spacy.load('en')

# Loading Data
df = pd.read_csv('../qacc_data.csv')
questions = df['Question'].tolist()
acceptability = df['Q_Acc'].tolist()

train_data = questions[: int(len(questions) * train_size)]
train_labels = acceptability[: int(len(acceptability) * train_size)]

val_size = int(len(acceptability[int(len(acceptability) * train_size) : ]) / 2)

val_data = questions[int(len(questions) * train_size) : int(len(questions) * train_size) + val_size]
val_labels = acceptability[int(len(acceptability) * train_size) : int(len(questions) * train_size) + val_size]

test_data = questions[int(len(questions) * train_size) + val_size : ]
test_labels = acceptability[int(len(acceptability) * train_size) + val_size : ]

print('Train size', len(train_data))
print('Val size', len(val_data))
print('Test size', len(test_data))


def single_epoch(data, label, train_flag):
	optimizer.zero_grad()
	pred_labels = []
	golden_labels = []
	total_loss = 0
	count = 0

	for i in range(len(data)):
		count += 1
		question = data[i]
		acc = int(label[i])

		# tok_ques = word_tokenize(question)
		tok_ques = [t.text.lower() for t in nlp(question)]

		# adjmat = get_sliding_adj_mat(tok_ques)
		adjmat = get_dep_adj_mat(question)
		adjmat = torch.tensor(adjmat).float().to(device)

		embed_ques = [get_embed(w, glovemodel) for w in tok_ques]
		embed_ques = torch.tensor(embed_ques).float().to(device)

		# pdb.set_trace()

		prediction_class = model(embed_ques, adjmat)

		# Getting true label
		true_class =  torch.tensor([acc]).to(device)

		# Loss
		loss = criterion(prediction_class, true_class)
		# Backpropagation (gradients are accumulated in parameters)
		loss.backward()

		# Accumulating the total loss
		total_loss += loss.item()

		if train_flag and (count % batch_size == 0):
			# Gradient Descent after all the batch gradients are accumulated
			optimizer.step()
			optimizer.zero_grad()

		# For getting accuracy 
		golden_labels.append(int(acc))
		pred_labels.append(torch.argmax(prediction_class, dim = 1).item())

	# Final update for remaining datapoints not included in any batch
	if train_flag:
		optimizer.step()
		optimizer.zero_grad()

	avg_loss = total_loss/len(data)
	return avg_loss, golden_labels, pred_labels


prev_val_loss = 100000000

for e in range(epochs):
	train_loss, train_golden, train_preds = single_epoch(train_data, train_labels, train_flag = True)
	val_loss, val_golden, val_preds = single_epoch(val_data, val_labels, train_flag = False)

	out_line = 'For epoch ' + str(e) + ' Train loss: ' + str(train_loss) + ' Train acc: ' + \
				str(accuracy_score(train_golden, train_preds)) + ' | Val loss: ' + str(val_loss) + \
				' Val acc: ' + str(accuracy_score(val_golden, val_preds)) + '\n\n'
	print(out_line)
	log_file = open('./dep_log.txt', 'a')
	log_file.write(out_line)
	log_file.close()

	if val_loss < prev_val_loss:
		prev_val_loss = val_loss
		test_loss, test_golden, test_preds = single_epoch(test_data, test_labels, train_flag = False)
		
		out_line = 'Test loss: ' + str(test_loss) + ' Test acc: ' + str(accuracy_score(test_golden, test_preds)) + '\n\n'
		log_file = open('./dep_log.txt', 'a')
		log_file.write(out_line)
		log_file.close()

		best_log = open('./dep_best_log.txt', 'w')
		best_log.write('For Epoch: ' + str(e) + '\n\n')
		best_log.write(classification_report(train_golden, train_preds))
		best_log.write('\n\n')
		best_log.write(classification_report(val_golden, val_preds))
		best_log.write('\n\n')
		best_log.write(classification_report(test_golden, test_preds))
		best_log.write('\n\n')
		best_log.close()

		torch.save(model.state_dict(), trained_models_path + 'ggnn_dep_best_model.pt')