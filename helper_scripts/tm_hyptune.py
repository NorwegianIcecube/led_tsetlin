from __future__ import annotations
import numpy as np
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from tmu.models.autoencoder.autoencoder import TMAutoEncoder
import pandas as pd
from time import time
from typing import Iterable, Any
from itertools import product


# Load data
# categories = ['alt.atheism', 'soc.religion.christian', 'talk.religion.misc']
categories = ["alt.atheism", "soc.religion.christian", "talk.religion.misc"]
print("fetching training data \n")
data_train = fetch_20newsgroups(
    subset='train', categories=categories, shuffle=True, random_state=42)

print("fetching test data \n")
data_test = fetch_20newsgroups(
    subset='test', categories=categories, shuffle=True, random_state=42)

all_data = pd.concat([pd.DataFrame(data_train.data),
                     pd.DataFrame(data_test.data)])
all_data = all_data.sample(frac=1).reset_index(drop=True)
data_train.data = all_data.iloc[:int(len(all_data)/2)].to_numpy().flatten()
data_test.data = all_data.iloc[int(len(all_data)/2):].to_numpy().flatten()

# target_words = ['god', 'jesus']
target_words = ["jesus",
                "christ",
                "accept",
                "trust",
                "faith"]


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
    num_p_weights, num_r_weights = 0, 0

    for i in range(num_target_words):
        clause_precision = model.clause_precision(
            i, True, vectorized_data, number_of_examples=num_examples)
        clause_recall = model.clause_recall(
            i, True, vectorized_data, number_of_examples=num_examples)

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
            f1.append(2 * (precision[i] * recall[i]) /
                      (precision[i] + recall[i]))
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
        indexed_next_word_prediction_sentence = drop_post_index(
            indexed_sentence)
        return indexed_next_word_prediction_sentence


def indexed_missing_word_prediction_sentence(sentence, target_word):
    if target_word not in sentence:
        return None
    indexed_sentence = index_sentence(sentence, target_word)
    return indexed_sentence


def standard_next_word_prediction_sentence(sentence, target_word):
    if target_word not in sentence:
        return None
    standard_next_word_prediction_sentence = drop_post_index(
        sentence, False, target_word)
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
                # new_sentence = indexed_next_word_prediction_sentence(sentence, word)
                # new_sentence = indexed_missing_word_prediction_sentence(sentence, word)
                # new_sentence = standard_next_word_prediction_sentence(sentence, word)
                #new_sentence = standard_missing_word_prediction_sentence(
                #    sentence, word)
                #if new_sentence is not None:
                temp.append(sentence)

    data.data = temp

    return data


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

# Create Hyperparameter vectors
examples_max = 1000
margin_max = 500
clause_max = 100
specificity_max = 20.0
accumulation_max = 50
max_num_literals = 9
steps = 2

examples_vector = [250]
margin_vector = [50]
clause_vector = [5]
# Setting specificity to 1 makes the epochs take 400+ seconds
specificity_vector = [2.5]
accumulation_vector = [5]
max_num_vector = [3]


def grid_parameters(parameters: dict[str, Iterable[Any]]) -> Iterable[dict[str, Any]]:
    for params in product(*parameters.values()):
        yield dict(zip(parameters.keys(), params))


for i in range(steps):
    examples_vector.append(
        examples_vector[i] + (examples_max - examples_vector[0]) // steps
    )
    margin_vector.append(
        margin_vector[i] + (margin_max - margin_vector[0]) // steps)
    clause_vector.append(
        clause_vector[i] + (clause_max - clause_vector[0]) // steps)
    specificity_vector.append(
        specificity_vector[i] +
        (specificity_max - specificity_vector[0]) / steps
    )
    accumulation_vector.append(
        accumulation_vector[i] +
        (accumulation_max - accumulation_vector[0]) // steps
    )
    max_num_vector.append(
        max_num_vector[i] + (max_num_literals - max_num_vector[0]) // steps)

settings = {
    "examples": examples_vector,
    "margin": margin_vector,
    "clauses": clause_vector,
    "specificity": specificity_vector,
    "accumulation": accumulation_vector,
    "max_num_literal": max_num_vector,
}
# Set Hyperparameters

clause_weight_threshold = 0
epochs = 15


output_active = np.empty(len(target_words), dtype=np.uint32)
for i in range(len(target_words)):
    target_word = target_words[i]

    target_id = count_vect.vocabulary_[target_word]
    output_active[i] = target_id

df = pd.DataFrame(
    columns=[
        "Precision",
        "Recall",
        "F1 Score",
        "Test Precision",
        "Test Recall",
        "Test F1",
        "Number of examples",
        "Margin",
        "Clauses",
        "Specificity",
        "Accumulation",
        "Max_included_literals",
    ]
)
df.to_csv("results.csv", index=False, mode="w")


def train(examples, margin, clauses, specificity, accumulation, max_num_literal):
    enc = TMAutoEncoder(
        number_of_clauses=clauses,
        T=margin,
        s=specificity,
        output_active=output_active,
        accumulation=accumulation,
        feature_negation=False,
        platform="CPU",
        output_balancing=True,
        max_included_literals=max_num_literal
    )

    # Train the Tsetlin Machine Autoencoder
    print("Starting training \n")
    for e in range(epochs):
        start_training = time()
        enc.fit(X_train_counts, number_of_examples=examples)
        stop_training = time()

        print("Epoch #%d" % (e + 1))
        print("Time %f" % (stop_training - start_training))
        if e == epochs - 1:
            train_precision, train_recall, train_f1 = weighted_average_precision_recall(
                enc, len(target_words), examples, X_train_counts)
            model_precision, model_recall, model_f1 = weighted_average_precision_recall(
                enc, len(target_words), examples, X_test_counts)

            temp_data = pd.DataFrame(
                {
                    "Precision": [train_precision],
                    "Recall": [train_recall],
                    "F1 Score": [train_f1],
                    "Test Precision": [model_precision],
                    "Test Recall": [model_recall],
                    "Test F1": [model_f1],
                    "Number of examples": [examples],
                    "Margin": [margin],
                    "Clauses": [clauses],
                    "Specificity": [specificity],
                    "Accumulation": [accumulation],
                    "Max_included_literals": [max_num_literal],
                }
            )

            temp_data.to_csv("results.csv", index=False,
                             header=False, mode="a")


for parameters in grid_parameters(settings):
    train(
        parameters["examples"],
        parameters["margin"],
        parameters["clauses"],
        parameters["specificity"],
        parameters["accumulation"],
        parameters["max_num_literal"]
    )
