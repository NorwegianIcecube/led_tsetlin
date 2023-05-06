import numpy as np
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from tmu.models.autoencoder.autoencoder import TMAutoEncoder
from time import time

# Load data
#categories = ['alt.atheism', 'soc.religion.christian', 'talk.religion.misc']
categories = ['comp.graphics', 'comp.os.ms-windows.misc', 'comp.sys.ibm.pc.hardware', 'comp.sys.mac.hardware',
              'comp.windows.x', 'sci.crypt', 'sci.electronics', 'sci.med', 'sci.space', 'misc.forsale',
              'talk.politics.misc', 'talk.politics.guns', 'talk.politics.mideast', 'talk.religion.misc',
              'rec.autos', 'rec.motorcycles', 'rec.sport.baseball', 'rec.sport.hockey', 'alt.atheism', 'soc.religion.christian']
print("fetching training data \n")
data_train = fetch_20newsgroups(
    subset='train', categories=categories, shuffle=True, random_state=42)
print("fetching test data \n")
data_test = fetch_20newsgroups(
    subset='test', categories=categories, shuffle=True, random_state=42)

#target_words = ['god', 'jesus']
#target_words = ['graphics', 'windows', 'ibm', 'mac', 'x', 'crypt', 'electronics', 'medicine', 'space', 'sale', 'politics', 'guns', 'mideast', 'religion', 'autos', 'motorcycles', 'baseball', 'hockey', 'atheism', 'christian']
target_words = ['the', 'an', 'to']

def index_sentence(sentence, keyword):
    words = sentence.split()
    index = ""
    found_index = 0
    for i in range(len(words)):
        if words[i] == keyword:
            found_index = i
            break
    if found_index == 0:
        return sentence
    else:
        for i in range(len(words)):
            index += f"{words[i]}:{i - found_index} "
    return index[:-1]

def drop_post_index(sentence):
    words = sentence.split()
    sentence = ""
    for word in words:
        sentence += f"{word} "
        if ":0" in word:
            break
    return sentence[:-1]
    
def weighted_average_precision_recall(model, num_target_words, num_examples):
    #profile = np.empty((len(target_words), clauses))
    precision = []
    recall = []
    f1 = []
    all_precision = 0
    num_weights = 0
    for i in range(num_target_words):
        clause_precision = model.clause_precision(i, True, X_train_counts, number_of_examples=num_examples)
        for index, clause in enumerate(clause_precision):
            if np.isnan(clause):
                continue
            weighted_precision = clause * model.get_weights(i)[index]
            all_precision += weighted_precision
            num_weights += model.get_weights(i)[index]
        precision.append(all_precision / num_weights)

    all_recall = 0
    num_weights = 0
    for i in range(num_target_words):
        clause_recall = model.clause_recall(i, True, X_train_counts, number_of_examples=num_examples)
        for index, clause in enumerate(clause_recall):
            if np.isnan(clause):
                continue
            weighted_recall = clause * model.get_weights(i)[index]
            all_recall += weighted_recall
            num_weights += model.get_weights(i)[index]
        recall.append(all_recall / num_weights)
        
    for i in range(num_target_words):
        f1.append(2 * (precision[i] * recall[i]) / (precision[i] + recall[i]))

    return precision, recall, f1

            

# Pre-process data
for data_set in [data_train, data_test]:
    temp = []
    for i in range(len(data_set.data)):
        # Remove all data before and including the line which includes an email address
        a = data_set.data[i].splitlines()
        for j in range(len(a)):
            if 'Lines:' in a[j]:
                a = a[j+1:]
                break
        data_set.data[i] = ' '.join(a)
        data_set.data[i] = data_set.data[i].replace(",", "")
        data_set.data[i] = data_set.data[i].replace('!', ' !')
        data_set.data[i] = data_set.data[i].replace('?', ' ?')
        data_set.data[i] = data_set.data[i].split('.')

    for doc in data_set.data:
        for sentence in doc:
            if len(sentence) < 5:
                continue
            if len(sentence.split()) < 5:
                continue
            sentence = sentence.strip()
            sentence = sentence.lower()
            for i in range(len(target_words)):
                indexed_sentence = index_sentence(sentence, target_words[i])
                #indexed_sentence = sentence
                if indexed_sentence != sentence:
                #if indexed_sentence == sentence:
                    indexed_sentence = drop_post_index(indexed_sentence)
                    temp.append(indexed_sentence)

    data_set.data = temp

# Create a count vectorizer
parsed_data_train = []
for i in range(len(data_train.data)):
    a = data_train.data[i].split()
    parsed_data_train.append(a)

def tokenizer(s):
    return s

count_vect = CountVectorizer(tokenizer=tokenizer, lowercase=False, binary=True)

X_train_counts = count_vect.fit_transform(parsed_data_train)
feature_names = count_vect.get_feature_names_out()
number_of_features = count_vect.get_feature_names_out().shape[0]
X_test_counts = count_vect.transform(data_test.data)

# Set Hyperparameters

clause_weight_threshold = 0
num_examples = 500
clauses = 100
# How many votes needed for action
margin = 150
# Forget value
specificity = 10.0
accumulation = 25
epochs = 20

# Create a Tsetlin Machine Autoencoder

output_active = np.empty(len(target_words), dtype=np.uint32)
for i in range(len(target_words)):
    target_word = f"{target_words[i]}:0"
    #target_word = target_words[i]

    target_id = count_vect.vocabulary_[target_word]
    output_active[i] = target_id

enc = TMAutoEncoder(number_of_clauses=clauses, T=margin,
                    s=specificity, output_active=output_active, accumulation=accumulation, feature_negation=False, platform='CPU', output_balancing=True)

# Train the Tsetlin Machine Autoencoder
print("Starting training \n")

for e in range(epochs):
    start_training = time()
    enc.fit(X_train_counts, number_of_examples=num_examples)
    stop_training = time()

    model_precision, model_recall, model_f1 = weighted_average_precision_recall(enc, len(target_words), num_examples)

    precision = []
    recall = []
    for i in range(len(target_words)):
        precision.append(enc.clause_precision(i, True, X_train_counts, number_of_examples=num_examples))
        recall.append(enc.clause_recall(i, True, X_train_counts, number_of_examples=num_examples))
    
    print("Epoch #%d" % (e+1))
    print("Precision: %s" % model_precision)
    print("Recall: %s" % model_recall)
    print(f"f1: {model_f1} \n")

print("Clauses\n")
for j in range(clauses):
    print("Clause #%d " % (j), end=' ')
    for i in range(len(target_words)):
        print("%s: W%d:P%.2f:R%.2f " % (target_words[i], enc.get_weight(
            i, j), precision[i][j], recall[i][j]), end=' ')

    l = []
    for k in range(enc.clause_bank.number_of_literals):
        if enc.get_ta_action(j, k) == 1:
            if k < enc.clause_bank.number_of_features:
                l.append("%s(%d)" % (
                    feature_names[k], enc.clause_bank.get_ta_state(j, k)))
            else:
                l.append("¬%s(%d)" % (
                    feature_names[k-enc.clause_bank.number_of_features], enc.clause_bank.get_ta_state(j, k)))
    print(" ∧ ".join(l))