import numpy as np
import pandas as pd
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from tmu.models.autoencoder.autoencoder import TMAutoEncoder

# Load data
categories = ['alt.atheism', 'soc.religion.christian', 'talk.religion.misc']
#categories = ['comp.graphics', 'comp.os.ms-windows.misc', 'comp.sys.ibm.pc.hardware',
              #'comp.sys.mac.hardware','comp.windows.x', 'sci.crypt', 'sci.electronics',
              #'sci.med', 'sci.space', 'misc.forsale','talk.politics.misc',
              #'talk.politics.guns', 'talk.politics.mideast', 'talk.religion.misc',
              #'rec.autos', 'rec.motorcycles', 'rec.sport.baseball', 'rec.sport.hockey',
              #'alt.atheism', 'soc.religion.christian']

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

target_words = ['jesus', 'christ', 'accept', 'trust', 'faith']
#target_words = ['graphics', 'windows', 'ibm', 'mac', 'gun', 'cryptography',
#                'electronics', 'medicine', 'space', 'sale', 'politics',
#                'guns', 'mideast', 'religion', 'autos', 'motorcycles',
#                'baseball', 'hockey', 'atheism', 'christian']


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
    
def weighted_average_precision_recall(model, num_target_words,
                                      num_examples, vectorized_data):
    precision = []
    recall = []
    f1 = []
    all_precision, all_recall = 0, 0
    num_p_weights,num_r_weights = 0, 0

    for i in range(num_target_words):
        clause_precision = model.clause_precision(i, True, vectorized_data,
                                                  number_of_examples=num_examples)
        clause_recall = model.clause_recall(i, True, vectorized_data,
                                            number_of_examples=num_examples)

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

def standard_missing_word_prediction_no_drop(sentence, target_word):
    return sentence

def pre_process(data, prediction_type):
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
                new_sentence = prediction_type(sentence, word)
                if new_sentence is not None:
                    temp.append(new_sentence)
    data.data = temp
    return data

def create_encoder(count_vect, target_words, clauses, margin,
                   specificity, accumulation, max_literals,
                   indexed=True):
    output_active = np.empty(len(target_words), dtype=np.uint32)
    for i, word in enumerate(target_words):
        if indexed:
            target_id = count_vect.vocabulary_[f"{word}:0"]
        else:
            target_id = count_vect.vocabulary_[word]
        output_active[i] = target_id

    enc = TMAutoEncoder(number_of_clauses=clauses, T=margin, max_included_literals=max_literals,
                        s=specificity, output_active=output_active, accumulation=accumulation,
                        feature_negation=False, platform='CPU', output_balancing=True)
    return enc

def tokenizer(s):
    return s

def create_count_vectorizer(data_train, data_test, tokenizer):
    parsed_data_train = []
    for i in range(len(data_train.data)):
        a = data_train.data[i].split()
        parsed_data_train.append(a)

    parsed_data_test = []
    for i in range(len(data_test.data)):
        a = data_test.data[i].split()
        parsed_data_test.append(a)

    count_vect = CountVectorizer(tokenizer=tokenizer, lowercase=False, binary=True)
    X_train_counts = count_vect.fit_transform(parsed_data_train)
    X_test_counts = count_vect.transform(parsed_data_test)

    return count_vect, X_train_counts, X_test_counts

def print_clauses(enc, target_words, precision, recall):
    clauses = enc.clause_bank.number_of_clauses
    count_vect = CountVectorizer(tokenizer=tokenizer, lowercase=False, binary=True)
    feature_names = count_vect.get_feature_names_out()
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
                        feature_names[k-enc.clause_bank.number_of_features],
                        enc.clause_bank.get_ta_state(j, k)
                        ))
        print(" ∧ ".join(l))
    
def train_encoder(enc, training_data, test_data, out_file, target_words,
                  num_examples, epochs, clause_print=False):
    metrics_per_epoch = np.empty((epochs, 7))
    print("Starting training \n")
    for e in range(epochs):
        enc.fit(training_data, number_of_examples=num_examples)

        model_precision, model_recall, model_f1 = weighted_average_precision_recall(enc, len(target_words),
                                                                                    num_examples, training_data)
        test_precision, test_recall, test_f1 = weighted_average_precision_recall(enc, len(target_words),
                                                                                num_examples, test_data)
        precision = []
        recall = []
        for i in range(len(target_words)):
            precision.append(enc.clause_precision(i, True, training_data, number_of_examples=num_examples))
            recall.append(enc.clause_recall(i, True, training_data, number_of_examples=num_examples))
        print("Epoch #%d" % (e+1))
        print("Precision: %s, %s" % (model_precision, test_precision))
        print("Recall: %s, %s" % (model_recall, test_recall))
        print("f1: %s, %s \n" % (model_f1, test_f1))

        e_df = pd.DataFrame(
            {
                'epoch': [e+1],
                'precision': model_precision,
                'recall': model_recall,
                'f1': model_f1,
                'test_precision': test_precision,
                'test_recall': test_recall,
                'test_f1': test_f1
            }
        )
        e_df.to_csv(out_file, index=False, mode='a', header=False)
        metrics_per_epoch[e] = [e+1, model_precision, model_recall, model_f1,
                                test_precision, test_recall, test_f1]
    
    if clause_print:
        print_clauses(enc, target_words, precision, recall)

    return metrics_per_epoch


if __name__ == "__main__":
    # Set Hyperparameters
    num_examples = 100
    clauses = 15
    margin = 500 # How many votes needed for action
    specificity = 2.5   # Forget value
    accumulation = 5
    max_literals = 20
    epochs = 15

    # Create dataframe to log results
    df = pd.DataFrame(columns=['epoch', 'precision', 'recall', 'f1',
                               'test_precision', 'test_recall', 'test_f1'])
    df.to_csv('tm_20newsgroups.csv', index=False, mode='w')

    for data_set in [data_train, data_test]:
        data_set = pre_process(data_set, indexed_next_word_prediction_sentence)

    count_vect, X_train_counts, X_test_counts = create_count_vectorizer(data_train, data_test, tokenizer)

    number_of_runs = 5
    results = np.empty((number_of_runs, epochs, 7))
    for i in range(number_of_runs):
        enc = create_encoder(count_vect, target_words, clauses, margin,
                            specificity, accumulation, max_literals,
                            indexed=True)
        results[i] = train_encoder(enc, X_train_counts, X_test_counts, 'tm_20newsgroups.csv',
                                    target_words, num_examples, epochs, clause_print=False)
    
    average_results = np.zeros((epochs, 7))
    divisor_array = np.full((epochs, 7), number_of_runs)
    for run in range(len(results)):
        for epoch in range(len(results[run])):
            for result in range(len(results[run][epoch])):
                if np.isnan(results[run][epoch][result]):
                    divisor_array[epoch][result] -= 1
                    continue
                average_results[epoch][result] += results[run][epoch][result]
    print(average_results)
    print(divisor_array)
    average_results = np.divide(average_results, divisor_array)
    

    print(average_results)
    df = pd.DataFrame(
        {
            'epoch': average_results[:, 0],
            'precision': average_results[:, 1],
            'recall': average_results[:, 2],
            'f1': average_results[:, 3],
            'test_precision': average_results[:, 4],
            'test_recall': average_results[:, 5],
            'test_f1': average_results[:, 6]
        }
    )
    df.to_csv('tm_20newsgroups_average.csv', index=False, mode='w', header=True)