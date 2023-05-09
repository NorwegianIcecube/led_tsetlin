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
        try:
            precision.append(all_precision / num_weights)
        except ZeroDivisionError:
            precision.append(0)
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
        try:
            recall.append(all_recall / num_weights)
        except ZeroDivisionError:
            recall.append(0)
        
    for i in range(num_target_words):
        try:
            f1.append(2 * (precision[i] * recall[i]) / (precision[i] + recall[i]))
        except ZeroDivisionError:
            f1.append(0)
    return precision, recall, f1


# Load data
categories = ['comp.graphics', 'comp.os.ms-windows.misc', 'comp.sys.ibm.pc.hardware', 'comp.sys.mac.hardware',
              'comp.windows.x', 'sci.crypt', 'sci.electronics', 'sci.med', 'sci.space', 'misc.forsale',
              'talk.politics.misc', 'talk.politics.guns', 'talk.politics.mideast', 'talk.religion.misc',
              'rec.autos', 'rec.motorcycles', 'rec.sport.baseball', 'rec.sport.hockey', 'alt.atheism', 'soc.religion.christian']
print("fetching training data \n")
data_train = fetch_20newsgroups(subset="train", categories=categories, shuffle=False)
print("fetching test data \n")
data_test = fetch_20newsgroups(subset="test", categories=categories, shuffle=False)

target_words = ['graphics', 'windows', 'ibm', 'mac', 'x', 'crypt', 'electronics', 'medicine', 'space', 'sale', 'politics', 'guns', 'mideast', 'religion', 'autos', 'motorcycles', 'baseball', 'hockey', 'atheism', 'christian']


# Pre-process data
for data_set in [data_train, data_test]:
    temp = []
    for i in range(len(data_set.data)):
        # Remove all data before and including the line which includes an email address
        a = data_set.data[i].splitlines()
        for j in range(len(a)):
            if "Lines:" in a[j]:
                a = a[j + 1 :]
                break
        data_set.data[i] = " ".join(a)
        data_set.data[i] = data_set.data[i].replace(",", "")
        data_set.data[i] = data_set.data[i].replace("!", " !")
        data_set.data[i] = data_set.data[i].replace("?", " ?")
        data_set.data[i] = data_set.data[i].split(".")

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
examples_max = 2000
margin_max = 500
clause_max = 250
specificity_max = 20.0
accumulation_max = 75
steps = 3

examples_vector = [100]
margin_vector = [50]
clause_vector = [15]
# Setting specificity to 1 makes the epochs take 400+ seconds
specificity_vector = [2.5]
accumulation_vector = [10]


def grid_parameters(parameters: dict[str, Iterable[Any]]) -> Iterable[dict[str, Any]]:
    for params in product(*parameters.values()):
        yield dict(zip(parameters.keys(), params))


for i in range(steps):
    examples_vector.append(
        examples_vector[i] + (examples_max - examples_vector[0]) // steps
    )
    margin_vector.append(margin_vector[i] + (margin_max - margin_vector[0]) // steps)
    clause_vector.append(clause_vector[i] + (clause_max - clause_vector[0]) // steps)
    specificity_vector.append(
        specificity_vector[i] + (specificity_max - specificity_vector[0]) / steps
    )
    accumulation_vector.append(
        accumulation_vector[i] + (accumulation_max - accumulation_vector[0]) // steps
    )

settings = {
    "examples": examples_vector,
    "margin": margin_vector,
    "clauses": clause_vector,
    "specificity": specificity_vector,
    "accumulation": accumulation_vector,
}
# Set Hyperparameters

clause_weight_threshold = 0
epochs = 15

# Create a Tsetlin Machine Autoencoder


output_active = np.empty(len(target_words), dtype=np.uint32)
for i in range(len(target_words)):
    target_word = f"{target_words[i]}:0"

    target_id = count_vect.vocabulary_[target_word]
    output_active[i] = target_id

df = pd.DataFrame(
    columns=[
        "Precision",
        "Recall",
        "F1 Score",
        "Number of examples",
        "Margin",
        "Clauses",
        "Specificity",
        "Accumulation",
        "Precision per target",
        "Recall per target",
        "F1 per target",
    ]
)
df.to_csv("results.csv", index=False, mode="w")



def train(examples, margin, clauses, specificity, accumulation):
    enc = TMAutoEncoder(
        number_of_clauses=clauses,
        T=margin,
        s=specificity,
        output_active=output_active,
        accumulation=accumulation,
        feature_negation=False,
        platform="CPU",
        output_balancing=True,
    )

    # Train the Tsetlin Machine Autoencoder
    print("Starting training \n")
    for e in range(epochs):
        start_training = time()
        enc.fit(X_train_counts, number_of_examples=examples)
        stop_training = time()

        profile = np.empty((len(target_words), clauses))
        precision = []
        recall = []

        print("Epoch #%d" % (e + 1))

        if e == epochs - 1:
            for i in range(len(target_words)):
                precision.append(
                    enc.clause_precision(
                        i, True, X_train_counts, number_of_examples=examples
                    )
                )
                recall.append(
                    enc.clause_recall(
                        i, True, X_train_counts, number_of_examples=examples
                    )
                )
                weights = enc.get_weights(i)
                profile[i, :] = np.where(weights >= clause_weight_threshold, weights, 0)

            print("Precision: %s" % precision)
            print("Recall: %s \n" % recall)
            print("Clauses\n")
            """
            clause_result = []
            for j in range(clause_vector[clause_index]):
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
                clause_result.append(" ∧ ".join(l))
        """

            similarity = cosine_similarity(profile)

            print("\nWord Similarity\n")
            word_result = ""
            for i in range(len(target_words)):
                print(target_words[i], end=": ")
                sorted_index = np.argsort(-1 * similarity[i, :])
                word_result += "\n" + target_words[i] + ": "
                for j in range(1, len(target_words)):
                    print(
                        "%s(%.2f) "
                        % (
                            target_words[sorted_index[j]],
                            similarity[i, sorted_index[j]],
                        ),
                        end=" ",
                    )
                    word_result += f"{target_words[sorted_index[j]]}: {similarity[i, sorted_index[j]]}, "

                print()
        print()
        print("\nTraining Time: %.2f" % (stop_training - start_training))
        if e == epochs - 1:
            model_precision, model_recall, model_f1 = weighted_average_precision_recall(enc, len(target_words), examples)
            temp_data = pd.DataFrame(
                {
                    "Precision": [np.sum(model_precision)/len(model_precision)],
                    "Recall": [np.sum(model_recall)/len(model_recall)],
                    "F1 Score": [np.sum(model_f1)/len(model_f1)],
                    "Number of examples": [examples],
                    "Margin": [margin],
                    "Clauses": [clauses],
                    "Specificity": [specificity],
                    "Accumulation": [accumulation],
                    "Precision per target": [model_precision],
                    "Recall per target": [model_recall],
                    "F1 per target": [model_f1],
                }
            )

            temp_data.to_csv("results.csv", index=False, header=False, mode="a")


for parameters in grid_parameters(settings):
    train(
        parameters["examples"],
        parameters["margin"],
        parameters["clauses"],
        parameters["specificity"],
        parameters["accumulation"],
    )
