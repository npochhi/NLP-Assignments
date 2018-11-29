
# coding: utf-8

# In[1]:


import nltk
from nltk.corpus import brown
from nltk import bigrams, ngrams, trigrams
import tqdm
import numpy as np
import math
import re
import pdb
from collections import Counter


# In[2]:


nltk.download('brown')


# In[3]:


sent_list = [' '.join(sent) for sent in brown.sents()[0:40000]]
sent_list = [re.sub('[^A-Za-z ]', '', sent).lower() for sent in sent_list]
sent_list


# In[28]:


file = open('test_sentences.txt', 'r')
test_sentences = []
for line in file:
    test_sentences += [line[:-1]]
test_sentences


# # Part 1 - No Smoothing

# In[29]:


# UniGram
unigrams = []

def unigram_model():
    for elem in sent_list:
        unigrams.extend(elem.split())
    model = Counter(unigrams)
    total_unigrams = len(unigrams) + 2 * 40000
    for uni in model:
        model[uni] /= total_unigrams
    return model

uni_model = unigram_model()
total_unigrams = len(unigrams) + 2 * 40000
uni_model['<s>'] = 40000 / total_unigrams
uni_model['</s>'] = 40000 / total_unigrams
print(uni_model['<s>'])
print(uni_model['</s>'])


# In[6]:


# BiGram

def bigram_model():
    model = {}
    for sent in sent_list:
        for w1, w2 in ngrams(sent.split(), 2, pad_left=True, pad_right=True, left_pad_symbol='<s>', right_pad_symbol='</s>'):
            model[w1] = model[w1] if w1 in model else {}
            model[w1][w2] = (model[w1][w2] + 1) if w2 in model[w1] else 1
    for w1 in model:
        total_bigrams = float(sum(model[w1].values()))
        for w2 in model[w1]:
            model[w1][w2] /= total_bigrams
    return model


bi_model = bigram_model()
bi_model


# In[7]:


# TriGram

def trigram_model():
    model = {}
    for sent in sent_list:
         for w1, w2, w3 in ngrams(sent.split(), 3, pad_left=True, pad_right=True, left_pad_symbol='<s>', right_pad_symbol='</s>'):
            if (w1, w2) not in model:
                model[(w1, w2)] = {}
            if w3 not in model[(w1, w2)]:
                model[(w1, w2)][w3] = 0
            model[(w1, w2)][w3] += 1
    for (w1, w2) in model:
        total_trigrams = float(sum(model[(w1, w2)].values()))
        for w3 in model[(w1, w2)]:
            model[(w1, w2)][w3] /= total_trigrams
    return model

tri_model = trigram_model()
tri_model


# In[8]:


# Top 10 values

def unigram_count():
    unigrams = []
    for elem in sent_list:
        unigrams.extend(elem.split())
    model = Counter(unigrams)
    return model

def bigram_count():
    model = {}
    for sent in sent_list:
        for w1, w2 in ngrams(sent.split(), 2, pad_left=True, pad_right=True):
            if w1 != None and w2 != None:
                model[(w1, w2)] = model.get((w1, w2), 0) + 1
    return model

def trigram_count():
    model = {}
    for sent in sent_list:
         for w1, w2, w3 in ngrams(sent.split(), 3, pad_left=True, pad_right=True):
            if w1 != None and w2 != None and w3 != None:
                model[(w1, w2, w3)] = model.get((w1, w2, w3), 0) + 1
    return model

all_unigram_counts = [[w, n] for w, n in unigram_count().items()]
all_bigram_counts = [[w, n] for w, n in bigram_count().items()]
all_trigram_counts = [[w, n] for w, n in trigram_count().items()]

sorted_uni = sorted(all_unigram_counts, key=lambda val: val[1])
sorted_bi = sorted(all_bigram_counts, key=lambda val: val[1])
sorted_tri = sorted(all_trigram_counts, key=lambda val: val[1])

print("Top 10 unigrams:")
print(sorted_uni[-10:])
print("Top 10 bigrams:")
print(sorted_bi[-10:])
print("Top 10 trigrams:")
print(sorted_tri[-10:])


# In[10]:


import matplotlib.pyplot as plt

model = bigram_count()
uniq_bi_counts = list(set(list(bigram_count().values())))
uniq_bi_counts.sort(reverse=True)
rank_freq = dict((val, i) for i, val in enumerate(uniq_bi_counts))
x_vals = uniq_bi_counts
y_vals = [rank_freq[w] for w in uniq_bi_counts]

plt.plot(x_vals, y_vals)
plt.xlabel('BiGrams')
plt.show()


# In[11]:


model = unigram_count()
uniq_uni_counts = list(set(unigram_count().values()))
uniq_uni_counts.sort(reverse=True)
rank_freq = dict((val, i) for i, val in enumerate(uniq_uni_counts))
x_vals = uniq_uni_counts
y_vals = [rank_freq[w] for w in uniq_uni_counts]


plt.plot(x_vals, y_vals)
plt.xlabel('Unigrams')
plt.show()


# In[12]:


model = trigram_count()
uniq_tri_counts = list(set(list(trigram_count().values())))
uniq_tri_counts.sort(reverse=True)
rank_freq = dict((val, i) for i, val in enumerate(uniq_tri_counts))
x_vals = uniq_tri_counts
y_vals = [rank_freq[w] for w in uniq_tri_counts]

plt.plot(x_vals, y_vals)
plt.xlabel('TriGrams')
plt.show()


# In[30]:


def compute_log_prob_no_smoothing():
    print("Unigram:")
    for sent in test_sentences:
        val, per = 0, 1
        for w in sent.split():
            val += np.log(uni_model.get(w, 0))
            per *= uni_model.get(w, 0)
        print(sent, val, (1.0 / per) ** (1.0 / len(sent.split())))
    print("BiGram:")
    for sent in test_sentences:
        val, per = 0, 1
        for w1, w2 in ngrams(sent.split(), 2, pad_left=True, pad_right=True, left_pad_symbol='<s>', right_pad_symbol='</s>'):
            if w1 in bi_model and w2 in bi_model[w1]:
                val += np.log(bi_model[w1][w2])
                per *= bi_model[w1][w2]
            else:
                val += np.log(0)
                per = 0
        print(sent, val, (1.0 / per if per != 0 else math.inf) ** (1.0 / len(sent.split())))
    print("TriGram:")
    for sent in test_sentences:
        val, per = 0, 1
        for w1, w2, w3 in ngrams(sent.split(), 3, pad_left=True, pad_right=True, left_pad_symbol='<s>', right_pad_symbol='</s>'):
            if (w1, w2) in tri_model and w3 in tri_model[(w1, w2)]:
                val += np.log(tri_model[(w1, w2)][w3])
                per *= tri_model[(w1, w2)][w3]
            else:
                val += np.log(0)
                per = 0
        print(sent, val, (1.0 / per if per != 0 else math.inf) ** (1.0 / len(sent.split())))
compute_log_prob_no_smoothing()


# # Part 2(i): Laplacian Smoothing

# In[14]:


# UniGram Laplacian Smoothing

def count(k, num, denom):
    n = len(uni_model)
    return (k + num) / (k * n + denom)

def unigram_lap_smoothing(k):
    unigrams = []
    for elem in sent_list:
        unigrams.extend(elem.split())
    model = Counter(unigrams)
    total_unigrams = len(unigrams)
    for uni in model:
        model[uni] = count(k, model[uni], total_unigrams)
    return model

uni_lap_model = unigram_lap_smoothing(1)
uni_lap_model


# In[16]:


# BiGram Laplacian Smoothing

def bigram_lap_smoothing(k):
    model = {}
    for sent in sent_list:
        for w1, w2 in ngrams(sent.split(), 2, pad_left=True, pad_right=True):
            model[w1] = model[w1] if w1 in model else {}
            model[w1][w2] = model[w1][w2] + 1 if w2 in model[w1] else 1
    for w1 in model:
        total_bigrams = float(sum(model[w1].values()))
        for w2 in model[w1]:
            model[w1][w2] = count(k, model[w1][w2], total_bigrams)
    return model

bi_lap_model = bigram_lap_smoothing(1)
bi_lap_model


# In[17]:


# TriGram Laplacian Smoothing

def trigram_lap_smoothing(k):
    model = {}
    for sent in sent_list:
         for w1, w2, w3 in ngrams(sent.split(), 3, pad_left=True, pad_right=True):
            if (w1, w2) not in model:
                model[(w1, w2)] = {}
            if w3 not in model[(w1, w2)]:
                model[(w1, w2)][w3] = 0
            model[(w1, w2)][w3] += 1
    for (w1, w2) in model:
        total_trigrams = float(sum(model[(w1, w2)].values()))
        for w3 in model[(w1, w2)]:
            model[(w1, w2)][w3] = count(k, model[(w1, w2)][w3], total_trigrams)
    return model

tri_lap_model = trigram_lap_smoothing(1)
tri_lap_model


# In[18]:


def unigram_count():
    unigrams = []
    for elem in sent_list:
        unigrams.extend(elem.split())
    model = Counter(unigrams)
    model['<s>'] = 40000
    model['</s>'] = 40000
    return model

def bigram_count():
    model = {}
    for sent in sent_list:
        for w1, w2 in ngrams(sent.split(), 2, pad_left=True, pad_right=True, left_pad_symbol='<s>', right_pad_symbol='</s>'):
            model[w1] = model.get(w1, {})
            model[w1][w2] = model[w1].get(w2, 0) + 1
    return model

def trigram_count():
    model = {}
    for sent in sent_list:
        for w1, w2, w3 in ngrams(sent.split(), 3, pad_left=True, pad_right=True, left_pad_symbol='<s>', right_pad_symbol='</s>'):
            model[(w1, w2)] = model.get((w1, w2), {})
            model[(w1, w2)][w3] = model[(w1, w2)].get(w3, 0) + 1
    return model


def compute_log_prob_lap_smoothing(k):
    model_uni = unigram_count()
    model_bi = bigram_count()
    model_tri = trigram_count()
    print("k =", k)
    print("Unigram:")
    for sent in test_sentences:
        val, per = 0, 1
        model = unigram_lap_smoothing(k)
        for w in sent.split():
            val += np.log(model.get(w, 1.0 / len(model_uni)))
            per *= model.get(w, 1.0 / len(model_uni))
        print(sent, val, (1.0 / per if per != 0 else math.inf) ** (1.0 / len(sent.split())))
    print("BiGram:")
    for sent in test_sentences:
        val, per = 0, 1
        for w1, w2 in ngrams(sent.split(), 2, pad_left=True, pad_right=True, left_pad_symbol='<s>', right_pad_symbol='</s>'):
            w1_count = float(sum(model_bi.get(w1, {}).values()))
            val += np.log(count(k, model_bi.get(w1, {}).get(w2, 0), w1_count))
            per *= count(k, model_bi.get(w1, {}).get(w2, 0), w1_count)
        print(sent, val, (1.0 / per if per != 0 else math.inf) ** (1.0 / len(sent.split())))
    print("TriGram:")
    for sent in test_sentences:
        val, per = 0, 1
        for w1, w2, w3 in ngrams(sent.split(), 3, pad_left=True, pad_right=True, left_pad_symbol='<s>', right_pad_symbol='</s>'):
            w1w2_count = float(sum(model_tri.get((w1, w2), {}).values()))
            val += np.log(count(k, model_tri.get((w1, w2), {}).get(w3, 0), w1w2_count))
            per *= count(k, model_tri.get((w1, w2), {}).get(w3, 0), w1w2_count)
        print(sent, val, (1.0 / per if per != 0 else math.inf) ** (1.0 / len(sent.split())))
compute_log_prob_lap_smoothing(1)
compute_log_prob_lap_smoothing(0.1)
compute_log_prob_lap_smoothing(0.01)
compute_log_prob_lap_smoothing(0.001)
compute_log_prob_lap_smoothing(0.0001)


# # Part 2(ii): Good Turing Smoothing

# In[19]:


bi_list_count = {}
b_count = bigram_count()
for w1 in b_count:
    for w2 in b_count[w1]:
        bi_list_count[(w1, w2)] = b_count[w1][w2]

bi_freq_count = Counter(bi_list_count.values())
bi_freq_rank = sorted(list(bi_freq_count.keys()))
bi_freq_rank_dict = dict((k, i + 1) for i, k in enumerate(bi_freq_rank))
bi_r_bar = {}

for w1, w2 in bi_list_count:
    rank = bi_freq_rank_dict[bi_list_count[(w1, w2)]]
    if rank == len(bi_freq_rank):
        bi_r_bar[(w1, w2)] = 0
        continue
    bi_r_bar[(w1, w2)] = bi_freq_rank[rank] * bi_freq_count[bi_freq_rank[rank]] / bi_freq_count[bi_freq_rank[rank - 1]]

zero_bigram_count = bi_freq_count[bi_freq_rank[0]] / (len(unigram_count()) * len(unigram_count()) - len(bigram_count()))
denom = float(sum(bi_list_count.values()))
smooth_turing_bigram = {}
for sent in test_sentences:
    val, per = 0, 1
    for w1, w2 in ngrams(sent.split(), 2, pad_left=True, pad_right=True, left_pad_symbol='<s>', right_pad_symbol='</s>'):
        val += np.log(bi_r_bar.get((w1, w2), zero_bigram_count) / denom)
        per *= bi_r_bar.get((w1, w2), zero_bigram_count) / denom
    print(sent, val, (1.0 / per if per != 0 else math.inf) ** (1.0 / len(sent.split())))


# In[20]:


tri_list_count = {}
t_count = trigram_count()
for w1, w2 in t_count:
    for w3 in t_count[(w1, w2)]:
        tri_list_count[(w1, w2, w3)] = t_count[(w1, w2)][w3]

tri_freq_count = Counter(tri_list_count.values())
tri_freq_rank = sorted(list(tri_freq_count.keys()))
tri_freq_rank_dict = dict((k, i + 1) for i, k in enumerate(tri_freq_rank))
tri_r_bar = {}
for w1, w2, w3 in tri_list_count:
    rank = tri_freq_rank_dict[tri_list_count[(w1, w2, w3)]]
    if rank == len(tri_freq_rank):
        tri_r_bar[(w1, w2, w3)] = 0
        continue
    tri_r_bar[(w1, w2, w3)] = tri_freq_rank[rank] * tri_freq_count[tri_freq_rank[rank]] / tri_freq_count[tri_freq_rank[rank - 1]]

zero_trigram_count = tri_freq_count[tri_freq_rank[0]] / (len(unigram_count()) * len(unigram_count()) * len(unigram_count()) - len(trigram_count()))
denom = float(sum(tri_list_count.values()))
smooth_turing_trigram = {}
for sent in test_sentences:
    val, per = 0, 1
    for w1, w2, w3 in ngrams(sent.split(), 3, pad_left=True, pad_right=True, left_pad_symbol='<s>', right_pad_symbol='</s>'):
        val += np.log(tri_r_bar.get((w1, w2, w3), zero_trigram_count) / denom)
        per *= tri_r_bar.get((w1, w2, w3), zero_trigram_count) / denom
    print(sent, val, (1.0 / per if per != 0 else math.inf) ** (1.0 / len(sent.split())))


# # Part 3: Interpolation Smoothing

# In[21]:


def bigram_interpolation(l):
    model = {}
    for w1 in bi_model:
        for w2 in bi_model[w1]:
            if w1 not in model:
                model[w1] = {}
                model[w1][w2] = l * bi_model[w1][w2] + (1 - l) * uni_model[w2]
            else:
                model[w1][w2] = l * bi_model[w1][w2] + (1 - l) * uni_model[w2]
    return model

bi_interpolation_model = bigram_interpolation(0.2)
bi_interpolation_model


# In[23]:


def test_interpolation_smoothing(l):
    model_interpolation = bigram_interpolation(l)
    print("Lambda =", l)
    for sent in test_sentences:
        val, per = 0, 1
        for w1, w2 in ngrams(sent.split(), 2, pad_left=True, pad_right=True, left_pad_symbol='<s>', right_pad_symbol='</s>'):
            val += np.log(l * bi_model.get(w1, {}).get(w2, 0) + (1 - l) * uni_model.get(w2, 0))
            per *= l * bi_model.get(w1, {}).get(w2, 0) + (1 - l) * uni_model.get(w2, 0)
#             pdb.set_trace()
        print(sent, val, (1.0 / per if per != 0 else math.inf) ** (1.0 / len(sent.split())))

test_interpolation_smoothing(0.2)
test_interpolation_smoothing(0.5)
test_interpolation_smoothing(0.8)

