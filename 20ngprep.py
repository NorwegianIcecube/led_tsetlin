import numpy as np
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer


categories = ['alt.atheism', 'soc.religion.christian', 'talk.religion.misc']
data_train = fetch_20newsgroups(subset='train', categories=categories, shuffle=True, random_state=42)
data_test = fetch_20newsgroups(subset='test', categories=categories, shuffle=True, random_state=42)

# Count number of word occurrences in training data
count_vect = CountVectorizer(lowercase=False, binary=True)
X_train_counts = count_vect.fit_transform(data_train.data)
feature_names = count_vect.get_feature_names_out()
number_of_features = count_vect.get_feature_names_out().shape[0]

# Print words with highest number of occurrences and number of occurrences
print("Words with highest number of occurrences")
print("========================================")
for i in range(50):
    max_id = np.argmax(X_train_counts.sum(axis=0))
    # Take next most frequent word if current word is a word such as "the", "and", "or", etc.
    while feature_names[max_id] in ["The", "com", "It", "If", "the", "and", "or", "to", "of", "in", "a", "is", "it", "that", "this", "for", "on", "as", "was", "with", "be", "by", "are", "i", "from", "at", "an", "not", "have", "has", "he", "his", "but", "they", "their", "you", "your", "which", "we", "were", "there", "if", "or", "so", "will", "would", "could", "should", "shall", "may", "might", "must", "can", "do", "does", "did", "about", "into", "than", "too", "very", "also", "such", "only", "even", "more", "most", "less", "least", "no", "nor", "either", "neither", "both", "all", "any", "some", "such", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "first", "second", "third", "fourth", "fifth", "sixth", "seventh", "eighth", "ninth", "tenth", "last", "next", "previous", "other", "another", "same", "different", "such", "what", "which", "who", "whom", "whose", "where", "when", "why", "how", "here", "there", "every", "each", "either", "neither", "any", "some", "none", "many", "much", "few", "little", "enough", "all", "both", "either", "neither", "each", "every", "any", "some", "none", "many", "much", "few", "little", "enough", "either", "neither", "every", "any", "some", "none", "many", "much", "few", "little", "enough", "all", "both", "each", "edu", "Re", "Organization"]:
        X_train_counts[:,max_id] = 0
        max_id = np.argmax(X_train_counts.sum(axis=0))
    print("%s: %d" % (feature_names[max_id], X_train_counts[:,max_id].sum()))
    X_train_counts[:,max_id] = 0
