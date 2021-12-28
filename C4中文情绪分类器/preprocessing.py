import re
import jieba
from collections import Counter
import numpy as np

good_file = 'good.txt'
bad_file = 'bad.txt'


def filter_punc(sentence):
    sentence = re.sub("[\s+\.\!\/_,$%^*(+\"\'“”《》？“]+|[+——！，。？、~@￥%.&*（）：]+", "", sentence)
    return sentence


def Prepare_data(good_file, bad_file, is_filter=True):
    all_words = []
    pos_sentence = []
    neg_sentence = []
    with open(good_file, 'r', encoding='utf8') as fr:
        for idx, line in enumerate(fr):
            if is_filter:
                line = filter_punc(line)
            words = jieba.lcut(line)
            if len(words) > 0:
                all_words += words
                pos_sentence.append(words)
    print(good_file, idx + 1, len(all_words))
    count = len(all_words)
    with open(bad_file, 'r', encoding='utf-8') as fr:
        for idx, line in enumerate(fr):
            if is_filter:
                line = filter_punc(line)
            words = jieba.lcut(line)
            if len(words) > 0:
                all_words += words
                neg_sentence.append(words)
    print(bad_file, idx + 1, len(all_words) - count)
    dictionary = {}
    cnt = Counter(all_words)
    for word, freq in cnt.items():
        dictionary[word] = [len(dictionary), freq]
    print('Dictionary length:{}'.format(len(dictionary)))
    return pos_sentence, neg_sentence, dictionary


pos_sentences, neg_sentences, diction = Prepare_data(good_file, bad_file, True)
st = sorted(([(v[1], w) for w, v in diction.items()]))


def word2vec(word, diction):
    if word in diction:
        value = diction[word][0]
    else:
        value = -1
    return value


def index2word(index, diction):
    for w, v in diction.item():
        if v[0] == index:
            return w
    return None


def sentence2vec(sentence, dictionaty):
    vector = np.zeros(len(dictionaty))
    for l in sentence:
        vector[l] += 1
    return 1.0 * vector / len(sentence)


dataset = []
labels = []
sentences = []

for sentence in pos_sentences:
    new_sentence = []
    for l in sentence:
        if l in diction:
            new_sentence.append(word2vec(l, diction))
    dataset.append(sentence2vec(new_sentence, diction))
    labels.append(0)
    sentences.append(sentence)

for sentence in neg_sentences:
    new_sentence = []
    for l in sentence:
        if l in diction:
            new_sentence.append(word2vec(l, diction))
    dataset.append(sentence2vec(new_sentence, diction))
    labels.append(1)
    sentences.append(sentence)

indices = np.random.permutation(len(dataset))
dataset = [dataset[i] for i in indices]
labels = [labels[i] for i in indices]
sentences = [sentences[i] for i in indices]

test_size = len(dataset) // 10
print(test_size)
train_data = dataset[2 * test_size:]
train_label = labels[2 * test_size:]

valid_data = dataset[:test_size]
valida_label = labels[:test_size]

test_data = dataset[test_size:2 * test_size]
test_label = labels[test_size:2 * test_size]


