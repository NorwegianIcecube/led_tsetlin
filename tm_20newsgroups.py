import numpy as np
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from tmu.models.autoencoder.autoencoder import TMAutoEncoder
from time import time
import pandas as pd
import matplotlib.pyplot as plt

# Load data
categories = ["alt.atheism", "soc.religion.christian", "talk.religion.misc"]
print("fetching training data \n")
data_train = fetch_20newsgroups(
    subset="train", categories=categories, shuffle=True, random_state=42
)
print("fetching test data \n")
data_test = fetch_20newsgroups(
    subset="test", categories=categories, shuffle=True, random_state=42
)


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
        data_set.data[i] = data_set.data[i].replace(",", " ,")
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
            temp.append(sentence)

    data_set.data = temp


# Create a count vectorizer
parsed_data_train = []
for i in range(len(data_train.data)):
    a = data_train.data[i].split()
    parsed_data_train.append(list(map(str.lower, a)))

parsed_data_test = []
for i in range(len(data_test.data)):
    a = data_test.data[i].split()
    parsed_data_test.append(list(map(str.lower, a)))


def tokenizer(s):
    return s


count_vect = CountVectorizer(tokenizer=tokenizer, lowercase=False, binary=True)

X_train_counts = count_vect.fit_transform(parsed_data_train)
feature_names = count_vect.get_feature_names_out()
number_of_features = count_vect.get_feature_names_out().shape[0]

# print(X_train_counts)


X_test_counts = count_vect.transform(parsed_data_test)


# Set Hyperparameters

clause_weight_threshold = 0
num_examples = 500
clauses = 100
# How many votes needed for action
margin = 150
# Forget value
specificity = 5.0
accumulation = 25
epochs = 100

# Create a Tsetlin Machine Autoencoder
target_words = [
    "jesus",
    "christ",
    "accept",
    "trust",
    "faith",
    "religious",
    "god",
    "atheists",
]
output_active = np.empty(len(target_words), dtype=np.uint32)
for i in range(len(target_words)):
    target_word = target_words[i]

    target_id = count_vect.vocabulary_[target_word]
    output_active[i] = target_id

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
df = pd.DataFrame(
    data=[
        f"clause_weight_threshold: {clause_weight_threshold}",
        f"num_examples: {num_examples}",
        f"clauses: {clauses}",
        f"margin: {margin}",
        f"specificity: {specificity}",
        f"accumulation: {accumulation}",
    ]
)
df.to_csv("meta.csv", index=False, header=False, mode="w")
data = pd.DataFrame(columns=["f1 score", "precision", "recall"])
data.to_csv("training.csv", mode="w")

print("Starting training \n")
f1_list, precision_list, recall_list = [], [], []

for e in range(epochs):
    start_training = time()
    enc.fit(X_train_counts, number_of_examples=num_examples)
    stop_training = time()

    profile = np.empty((len(target_words), clauses))
    precision = []
    recall = []
    for i in range(len(target_words)):
        precision.append(
            enc.clause_precision(
                i, True, X_train_counts, number_of_examples=num_examples
            )
        )
        recall.append(
            enc.clause_recall(i, True, X_train_counts, number_of_examples=num_examples)
        )
        weights = enc.get_weights(i)
        profile[i, :] = np.where(weights >= clause_weight_threshold, weights, 0)

    print("Epoch #%d" % (e + 1))

    if e == epochs - 1:
        print("Precision: %s" % precision)
        print("Recall: %s \n" % recall)
        print("Clauses\n")
        for j in range(clauses):
            print("Clause #%d " % (j), end=" ")
            for i in range(len(target_words)):
                print(
                    "%s: W%d:P%.2f:R%.2f "
                    % (
                        target_words[i],
                        enc.get_weight(i, j),
                        precision[i][j],
                        recall[i][j],
                    ),
                    end=" ",
                )

            l = []
            for k in range(enc.clause_bank.number_of_literals):
                if enc.get_ta_action(j, k) == 1:
                    if k < enc.clause_bank.number_of_features:
                        l.append(
                            "%s(%d)"
                            % (feature_names[k], enc.clause_bank.get_ta_state(j, k))
                        )
                    else:
                        l.append(
                            "¬%s(%d)"
                            % (
                                feature_names[k - enc.clause_bank.number_of_features],
                                enc.clause_bank.get_ta_state(j, k),
                            )
                        )
            print(" ∧ ".join(l))
    similarity = cosine_similarity(profile)

    np_precision = np.concatenate(precision, axis=0)
    np_precision = np_precision.flatten("C")
    np_precision = np_precision[~np.isnan(np_precision)]
    avg_precision = np.sum(np_precision) / np_precision.size

    np_recall = np.concatenate(recall, axis=0)
    np_recall = np_recall.flatten("C")
    np_recall = np_recall[~np.isnan(np_recall)]
    avg_recall = np.nansum(np_recall) / np_recall.size
    f1_score = 2 * ((avg_precision * avg_recall) / (avg_precision + avg_recall))

    f1_list.append(f1_score)
    precision_list.append(avg_precision)
    recall_list.append(avg_recall)

    temp = pd.DataFrame({"f1 score":[f1_score], 
                         "precision":[avg_precision], 
                        "recall":[avg_recall]})
    temp.to_csv("training.csv", index=False, header=False, mode="a")

    print("\nWord Similarity\n")

    for i in range(len(target_words)):
        print(target_words[i], end=": ")
        sorted_index = np.argsort(-1 * similarity[i, :])
        for j in range(1, len(target_words)):
            print(
                "%s(%.2f) "
                % (target_words[sorted_index[j]], similarity[i, sorted_index[j]]),
                end=" ",
            )
        print()

    print("\nTraining Time: %.2f" % (stop_training - start_training))
plt.plot(f1_list, label="f1_score")
plt.plot(precision_list, label="precision")
plt.plot(recall_list, label="recall")
plt.xlabel("Epochs")
plt.legend()
plt.show()
