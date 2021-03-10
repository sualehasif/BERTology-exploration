import os
from file_parser import parse_reviews

from transformers import BertTokenizer, BertModel
from file_parser import parse_reviews
import torch
import numpy as np
import sklearn

from matplotlib import pyplot as plt
from matplotlib import cm
from sklearn import manifold

# Pretty colors
BLACK = "k"
GREEN = "#59d98e"
SEA = "#159d82"
BLUE = "#3498db"
PURPLE = "#9b59b6"
GREY = "#95a5a6"
RED = "#e74c3c"
ORANGE = "#f39c12"


def save_attention_list(attention_list, filename="attention_reviews.pt"):
	""" Save attention list  saves the output of attention_list as a file
	"""
	torch.save(attention_list, filename)


def get_attentions(model, tokenizer, sentence_a, sentence_b=None, layer=None, heads=None):
	""" Provides the attentions and tokens from the model for sentence_a (and paired with sentence_b if provided.)
	"""
	inputs = tokenizer.encode_plus(sentence_a, sentence_b, return_tensors='pt', add_special_tokens=True)
	input_ids = inputs['input_ids']
	if sentence_b:
		token_type_ids = inputs['token_type_ids']
		attention = model(input_ids, token_type_ids=token_type_ids)[-1]
		sentence_b_start = token_type_ids[0].tolist().index(1)
	else:
		attention = model(input_ids)[-1]
		sentence_b_start = None
	input_id_list = input_ids[0].tolist() # Batch index 0
	tokens = tokenizer.convert_ids_to_tokens(input_id_list)
	
	return attention, tokens


def compute_avg_attention(attention_list):
	""" Computes the avg attention and other statistics for the attnetion_list """
	n_docs = len(attention_list)
	def data_iterator():
		for i, doc in enumerate(attention_list):
			if i % 100 == 0 or i == len(attention_list) - 1:
				print("{:.1f}% done".format(100.0 * (i + 1) / len(attention_list)))
			yield doc["tokens"], doc["attn"].detach().numpy()

	avg_attns = {
		k: np.zeros((12, 12)) for k in [
		"self", "right", "left", "sep", "sep_sep", "rest_sep",
		"cls", "punct"]
	}

	print("Computing token stats")
	for tokens, attns in data_iterator():
		n_tokens = attns.shape[-1]

	# create masks indicating where particular tokens are
	seps, clss, puncts = (np.zeros(n_tokens) for _ in range(3))
	for position, token in enumerate(tokens):
		if token == "[SEP]":
			seps[position] = 1
		if token == "[CLS]":
			clss[position] = 1
		if token == "." or token == ",":
			puncts[position] = 1

	# create masks indicating which positions are relevant for each key
	sep_seps = np.ones((n_tokens, n_tokens))
	sep_seps *= seps[np.newaxis]
	sep_seps *= seps[:, np.newaxis]

	rest_seps = np.ones((n_tokens, n_tokens))
	rest_seps *= (np.ones(n_tokens) - seps)[:, np.newaxis]
	rest_seps *= seps[np.newaxis]

	selectors = {
		"self": np.eye(n_tokens, n_tokens),
		"right": np.eye(n_tokens, n_tokens, 1),
		"left": np.eye(n_tokens, n_tokens, -1),
		"sep": np.tile(seps[np.newaxis], [n_tokens, 1]),
		"sep_sep": sep_seps,
		"rest_sep": rest_seps,
		"cls": np.tile(clss[np.newaxis], [n_tokens, 1]),
		"punct": np.tile(puncts[np.newaxis], [n_tokens, 1]),
	}

	# get the average attention for each token type
	for key, selector in selectors.items():
		if key == "sep_sep":
			denom = 2
		elif key == "rest_sep":
			denom = n_tokens - 2
		else:
			denom = n_tokens
		avg_attns[key] += (
			(attns * selector[np.newaxis, np.newaxis]).sum(-1).sum(-1) /
			(n_docs * denom))
	
	return avg_attns


def generate_attention_for_reviews(reviews):
	""" 
	Generates teh attention matrices from the model and puts them in a list of dictionaries. 
	"""
	model_v = 'bert-base-cased'

	model = BertModel.from_pretrained(model_v, output_attentions=True)
	tokenizer = BertTokenizer.from_pretrained(model_v, do_lower_case=False)
	
	attention_list = []
	
	for sentence_id, sentence in enumerate(reviews):
		if sentence_id % 10 == 0:
			print(f"getting attention for sentence: {sentence_id}")
		attn, sentence_tokens = get_attentions(model, tokenizer, sentence)
		attention_matrix = torch.stack(tuple(attn[i][0] for i in range(len(attn))))
		
		attention_list.append({'sentence': sentence, 'attn': attention_matrix, 'tokens': sentence_tokens})
	
	return attention_list

def generate_entropy_for_reviews(attention_list):
	""" 
	Generates entropies from the layers of matrices and puts them in a list indexed by layer. 
	"""
	n_docs = len(attention_list)
	def data_iterator():
		for i, doc in enumerate(attention_list):
			if i % 100 == 0 or i == len(attention_list) - 1:
				print("{:.1f}% done".format(100.0 * (i + 1) / len(attention_list)))
			yield doc["tokens"], doc["attn"].detach().numpy()
	
	uniform_attn_entropy = 0  # entropy of uniform attention
	entropies = np.zeros((12, 12))  # entropy of attention heads
	entropies_cls = np.zeros((12, 12))  # entropy of attention from [CLS]

	print("Computing entropy stats")
	for tokens, attns in data_iterator():
		attns = 0.9999 * attns + (0.0001 / attns.shape[-1])  # smooth to avoid NaNs
		uniform_attn_entropy -= np.log(1.0 / attns.shape[-1])
		entropies -= (attns * np.log(attns)).sum(-1).mean(-1)
		entropies_cls -= (attns * np.log(attns))[:, :, 0].sum(-1)

	uniform_attn_entropy /= n_docs
	entropies /= n_docs
	entropies_cls /= n_docs

	return entropies, entropies_cls

def plot_attn(example, heads, disable_sep=False, disable_cls=True):
  """Plots attention maps for the given example and attention heads."""
  width = 3
  example_sep = 3
  word_height = 1
  pad = 0.1

  for ei, (layer, head) in enumerate(heads):
	yoffset = 1
	xoffset = ei * width * example_sep

	attn = example["attn"][layer][head][-15:, -15:]
	attn = attn.detach().numpy()
	attn /= attn.sum(axis=-1, keepdims=True)
	words = example["tokens"][-15:]
#     words[0] = "..."
	n_words = len(words)

	for position, word in enumerate(words):
	  plt.text(xoffset + 0, yoffset - position * word_height, word,
			   ha="right", va="center")
	  plt.text(xoffset + width, yoffset - position * word_height, word,
			   ha="left", va="center")
	if (disable_cls and disable_sep):
		for i in range(1, n_words):
		  for j in range(1, n_words - 1):
			plt.plot([xoffset + pad, xoffset + width - pad],
					 [yoffset - word_height * i, yoffset - word_height * j],
					 color="blue", linewidth=1, alpha=attn[i, j])
	elif (disable_sep):
		for i in range(0, n_words -1):
		  for j in range(0, n_words -1):
			plt.plot([xoffset + pad, xoffset + width - pad],
					 [yoffset - word_height * i, yoffset - word_height * j],
					 color="blue", linewidth=1, alpha=attn[i, j])
	elif (disable_cls):
		for i in range(1, n_words):
		  for j in range(1, n_words):
			plt.plot([xoffset + pad, xoffset + width - pad],
					 [yoffset - word_height * i, yoffset - word_height * j],
					 color="blue", linewidth=1, alpha=attn[i, j])
	else:
		for i in range(0, n_words):
		  for j in range(0, n_words):
			plt.plot([xoffset + pad, xoffset + width - pad],
					 [yoffset - word_height * i, yoffset - word_height * j],
					 color="blue", linewidth=1, alpha=attn[i, j])

def embed_js_attentions(avg_attns, entropies):
	""" 
	Embeds the js_distances which have been precomputed into a 2d space and then plots them" 
	"""
	js_divergences = torch.load("head_distances.pt")

	ENTROPY_THRESHOLD = 2.2  # When to say a head "attends broadly"
	POSITION_THRESHOLD = 0.5  # When to say a head "attends to next/prev"
	SPECIAL_TOKEN_THRESHOLD = 0.6  # When to say a heads attends to [CLS]/[SEP]"
	# Heads that were found to have linguistic behaviors
	LINGUISTIC_HEADS = {
		(4, 3): "Coreference",
		(8, 4): "Adjectival modifier",
		(9, 7): "Nominal Subject",
		(3, 9): "Passive auxiliary",
		(6, 5): "Possesive",
	}

	# Use multi-dimensional scaling to compute 2-dimensional embeddings that
	# reflect Jenson-Shannon divergences between attention heads.
	mds = sklearn.manifold.MDS(metric=True, n_init=5, n_jobs=4, eps=1e-10,
							max_iter=1000, dissimilarity="precomputed")
	pts = mds.fit_transform(js_divergences)
	pts = pts.reshape((12, 12, 2))
	pts_flat = pts.reshape([144, 2])

	colormap = cm.seismic(np.linspace(0, 1.0, 12))
	plt.figure(figsize=(4.8, 9.6))
	plt.title("BERT Attention Heads")

	for color_by_layer in [False, True]:
		ax = plt.subplot(2, 1, int(color_by_layer) + 1)
		seen_labels = set()
		for layer in range(12):
			for head in range(12):
				label = ""
				color = GREY
				marker = "o"
				markersize = 4
				x, y = pts[layer, head]

				if avg_attns["right"][layer, head] > POSITION_THRESHOLD:
					color = RED
					marker = ">"
					label = "attend to next"
					
				if avg_attns["left"][layer, head] > POSITION_THRESHOLD:
					color = BLUE
					label = "attend to prev."
					marker = "<"

				if avg_attns["cls"][layer, head] > SPECIAL_TOKEN_THRESHOLD:
					color = PURPLE
					label = "attend to [CLS]"
					marker = "$C$"
					markersize = 5

				if avg_attns["sep"][layer, head] > SPECIAL_TOKEN_THRESHOLD:
					color = GREEN
					marker = "$S$"
					markersize = 5
					label = "attend to [SEP]"

				if entropies[layer, head] > ENTROPY_THRESHOLD:
					color = ORANGE
					label = "attend broadly"
					marker = "^"

				if avg_attns["punct"][layer, head] > SPECIAL_TOKEN_THRESHOLD:
					color = SEA
					marker = "s"
					markersize = 3.2
					label = "attend to . and ,"

				if color_by_layer:
					label = str(layer + 1)
					color = colormap[layer]
					marker = "o"
					markersize = 3.8

				if not color_by_layer:
					if (layer, head) in LINGUISTIC_HEADS:
						label = ""
						color = BLACK
						marker = "x"
						ax.text(x, y, LINGUISTIC_HEADS[(layer, head)], color=color)

				if label not in seen_labels:
					seen_labels.add(label)
				else:
					label = ""

				ax.plot([x-0.5], [y+0.5], marker=marker, markersize=markersize,
						color=color, label=label, linestyle="")

	ax.set_xticks([])
	ax.set_yticks([])
	ax.spines["top"].set_visible(False)
	ax.spines["right"].set_visible(False)
	ax.spines["bottom"].set_visible(False)
	ax.spines["left"].set_visible(False)
	ax.set_facecolor((0.96, 0.96, 0.96))
	plt.title(("Colored by Layer" if color_by_layer else "Behaviors"))
	handles, labels = ax.get_legend_handles_labels()
	ax.legend(handles, labels, loc="best")

	plt.suptitle("Embedded BERT attention heads", fontsize=14, y=1.05)
	plt.subplots_adjust(top=1, bottom=0, right=1, left=0,
						hspace=0.1, wspace=0)
	plt.show()


if __name__ == "__main__": 
	print("Starting the task")

	review_file_path = str(os.getcwd()) + "/restaurant_reviews.txt"
	review_sentences = parse_reviews(review_file_path)

	print(review_sentences[:15])
	