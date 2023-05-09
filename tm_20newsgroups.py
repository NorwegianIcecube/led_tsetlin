import numpy as np
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from tmu.models.autoencoder.autoencoder import TMAutoEncoder
from time import time
import pandas as pd

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

all_data = pd.concat([pd.DataFrame(data_train.data), pd.DataFrame(data_test.data)])
all_data = all_data.sample(frac=1).reset_index(drop=True)
data_train.data = all_data.iloc[:int(len(all_data)/2)].to_numpy().flatten()
data_test.data = all_data.iloc[int(len(all_data)/2):].to_numpy().flatten()

#target_words = ['god', 'jesus']
target_words = ['graphics', 'windows', 'ibm', 'mac', 'x', 'crypt', 'electronics', 'medicine', 'space', 'sale', 'politics', 'guns', 'mideast', 'religion', 'autos', 'motorcycles', 'baseball', 'hockey', 'atheism', 'christian']


def index_sentence(sentence, keyword):
    words = sentence.split()
    index = ""
    found_index = 0
    for i, word in enumerate(words):
        if word == keyword:
            found_index = i
            break
    if found_index == 0:
        return sentence
    else:
        for i, word in enumerate(words):
            index += f"{word}:{i - found_index} "
    return index[:-1]

def drop_post_index(sentence, indexed=True, keyword=":0"):
    words = sentence.split()
    sentence = ""
    for word in words:
        sentence += f"{word} "
        if indexed and keyword in word:
            break
        elif keyword in word:
            break
    return sentence[:-1]
    
def weighted_average_precision_recall(model, num_target_words, num_examples, vectorized_data):
    precision = []
    recall = []
    f1 = []
    all_precision, all_recall = 0, 0
    num_p_weights,num_r_weights = 0, 0

    for i in range(num_target_words):
        clause_precision = model.clause_precision(i, True, vectorized_data, number_of_examples=num_examples)
        clause_recall = model.clause_recall(i, True, vectorized_data, number_of_examples=num_examples)

        for index, clause in enumerate(zip(clause_precision, clause_recall)):
            model_weights = model.get_weights(i)[index]
            if not np.isnan(clause[0]):
                weighted_precision = clause[0] * model_weights
                all_precision += weighted_precision
                num_p_weights += model_weights
            if not np.isnan(clause[1]):
                weighted_recall = clause[1] * model_weights
                all_recall += weighted_recall
                num_r_weights += model_weights
        
        try:
            precision.append(all_precision / num_p_weights)
        except ZeroDivisionError:
            precision.append(0)
        try:
            recall.append(all_recall / num_r_weights)
        except ZeroDivisionError:
            recall.append(0)

    for i in range(num_target_words):
        try:
            f1.append(2 * (precision[i] * recall[i]) / (precision[i] + recall[i]))
        except ZeroDivisionError:
            f1.append(0)

    average_precision = np.sum(precision)/len(precision)
    average_recall = np.sum(recall)/len(recall)
    average_f1 = np.sum(f1)/len(f1)

    return average_precision, average_recall, average_f1

def indexed_next_word_prediction_sentence(sentence, target_word):
    if target_word not in sentence:
        return None
    indexed_sentence = index_sentence(sentence, target_word)
    if indexed_sentence != sentence:
        indexed_next_word_prediction_sentence = drop_post_index(indexed_sentence)
        return indexed_next_word_prediction_sentence

def indexed_missing_word_prediction_sentence(sentence, target_word):
    if target_word not in sentence:
        return None
    indexed_sentence = index_sentence(sentence, target_word)
    return indexed_sentence

def standard_next_word_prediction_sentence(sentence, target_word):
    if target_word not in sentence:
        return None
    standard_next_word_prediction_sentence = drop_post_index(sentence, False, target_word)
    return standard_next_word_prediction_sentence


def standard_missing_word_prediction_sentence(sentence, target_word):
    if target_word not in sentence:
        return None
    return sentence

def pre_process(data):
    temp = []
    for doc in data.data:
        # Remove all data before and including the line which includes an email address
        a = doc.splitlines()
        for j in range(len(a)):
            if 'Lines:' in a[j]:
                a = a[j+1:]
                break
        doc = ' '.join(a)
        doc = doc.replace(",", "")
        doc = doc.replace('!', ' !')
        doc = doc.replace('?', ' ?')
        doc = doc.split('.')
        
        for sentence in doc:
            sentence = sentence.strip()
            sentence = sentence.lower()
            for word in target_words:
                if sentence is None:
                    continue
                # Choose 1 of the following 4 lines
                new_sentence = indexed_next_word_prediction_sentence(sentence, word)
                #new_sentence = indexed_missing_word_prediction_sentence(sentence, word)
                #new_sentence = standard_next_word_prediction_sentence(sentence, word)
                #new_sentence = standard_missing_word_prediction_sentence(sentence, word)
                if new_sentence is not None:
                    temp.append(new_sentence)

    data.data = temp
    
    return data

# Pre-process data
for data_set in [data_train, data_test]:
    data_set = pre_process(data_set)

# Create count vectorizers
parsed_data_train = []
for i in range(len(data_train.data)):
    a = data_train.data[i].split()
    parsed_data_train.append(a)

parsed_data_test = []
for i in range(len(data_test.data)):
    a = data_test.data[i].split()
    parsed_data_test.append(a)

def tokenizer(s):
    return s

count_vect = CountVectorizer(tokenizer=tokenizer, lowercase=False, binary=True)

X_train_counts = count_vect.fit_transform(parsed_data_train)
feature_names = count_vect.get_feature_names_out()
number_of_features = count_vect.get_feature_names_out().shape[0]
X_test_counts = count_vect.transform(parsed_data_test)

# Set Hyperparameters

clause_weight_threshold = 0
num_examples = 1000
clauses = 15
margin = 350 # How many votes needed for action
specificity = 5.0   # Forget value
accumulation = 50
epochs = 20

# Create a Tsetlin Machine Autoencoder

output_active = np.empty(len(target_words), dtype=np.uint32)
for i in range(len(target_words)):
    # Choose 1 of the following 2 lines based on whether you want to use the indexed or uninexed version
    #target_id = count_vect.vocabulary_[target_words[i]]
    target_id = count_vect.vocabulary_[f"{target_words[i]}:0"]
    output_active[i] = target_id

enc = TMAutoEncoder(number_of_clauses=clauses, T=margin, max_included_literals=3,
                    s=specificity, output_active=output_active, accumulation=accumulation, feature_negation=False, platform='CPU', output_balancing=True)

# Create dataframe to log results
df = pd.DataFrame(columns=['epoch', 'precision', 'recall', 'f1'])
df.to_csv('tm_20newsgroups.csv', index=False, mode='w')

# Train the Tsetlin Machine Autoencoder
print("Starting training \n")

for e in range(epochs):
    start_training = time()
    enc.fit(X_train_counts, number_of_examples=num_examples)
    stop_training = time()

    model_precision, model_recall, model_f1 = weighted_average_precision_recall(enc, len(target_words), num_examples, X_test_counts)

    precision = []
    recall = []
    for i in range(len(target_words)):
        precision.append(enc.clause_precision(i, True, X_train_counts, number_of_examples=num_examples))
        recall.append(enc.clause_recall(i, True, X_train_counts, number_of_examples=num_examples))

    
    print("Epoch #%d" % (e+1))
    print("Precision: %s" % model_precision)
    print("Recall: %s" % model_recall)
    print(f"f1: {model_f1} \n")

    e_df = pd.DataFrame(
        {
            'epoch': [e+1],
            'precision': model_precision,
            'recall': model_recall,
            'f1': model_f1
        }
    )
    e_df.to_csv('tm_20newsgroups.csv', index=False, mode='a', header=False)


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