import nltk
nltk.download('punkt')
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.util import ngrams
import string
import numpy as np
from sklearn.utils import shuffle
import collections
import re
import math
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

"""
Helper function to tokenize the data and split the data into ngrams
with a default of a trigram
"""
def tokenize(str_data):
    data = str_data.lower()
    data = re.sub(r'\n', '', data)
    data = re.sub(r'[^A-Za-z\s]', '', data)
    data = data.split(' ')
    data = list(filter(lambda item: item != '', data))
    return data
def extract_ngrams(tokens, n = 3):
    ngram_list = ngrams(tokens, n)
    return ngram_list

#Read in the books, and store each page in a list for each book. Each page is a set of trigrams
books = []
for i in range(1, 8):
    with open (f"harry_potter/HP{i}.txt", 'r') as f:
        pages = []
        data = f.read()
        data = data.split('\n')
        for page in data:
            pages.append(tokenize(page))
        books.append(pages)
        f.close()

# Label each page with the corresponding book number
data = []
labels = []
for i, book in enumerate(books):
    for page in book:
        data.append(page)
        labels.append(i + 1)
books = None

#Shuffle and split the data with seed 0
data, labels = shuffle(data, labels, random_state=0)
train_data = data[:int(len(data) * 0.8)]
train_labels = labels[:int(len(labels) * 0.8)]

test_data = data[int(len(data) * 0.8):int(len(data) * 0.9)]
test_labels = labels[int(len(labels) * 0.8):int(len(labels) * 0.9)]

val_data = data[int(len(data) * 0.9):]
val_labels = labels[int(len(labels) * 0.9):]

data = None
labels = None

#Using the training data, build the word frequency table
word_frequencies = collections.defaultdict(lambda: collections.defaultdict(int))
for label, page in zip(train_labels, train_data):
    ngrams_list = list(extract_ngrams(page))
    for ngram in ngrams_list:
        word_frequencies[label][ngram] += 1

"""
A classify function that takes in a page and a delta value, and returns the most likely book 
"""
def classify(page, delta = 0.01):
    class_counts = collections.Counter(label for label in train_labels)
    class_priors = {cls: count / len(train_data) for cls, count in class_counts.items()}

    page_ngrams = list(extract_ngrams(page))

    probabilities = {}
    for label in class_counts.keys():
        prob = math.log(class_priors[label])

        for ngram in page_ngrams:
            n_count = word_frequencies[label][tuple(ngram)] + delta
            n_total = sum(word_frequencies[label].values()) + delta * len(word_frequencies[label])
            prob += math.log(n_count / n_total)
        probabilities[label] = prob

    return max(probabilities, key=probabilities.get)

#Using the validation set, find the best delta (hyper-parameter) for the model
#Stores the best delta value to be used in the testing set
deltas = [0.01, 0.001, 0.0001, 0.00005, 0.00001, 5e-6, 1e-6, 5e-7, 1e-7, 5e-8]
accuracies = []
print("Testing delta values : ", deltas)
best_delta = 0
best_accuracy = 0
for delta in deltas:
    predicted = []
    for page in val_data:
        predicted.append(classify(page, delta))
    cm = confusion_matrix(val_labels, predicted)
    tot = np.sum(cm)
    correct = np.sum(np.diagonal(cm))
    accuracy = correct/tot * 100
    accuracies.append(accuracy)

    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_delta = delta
    
    print(f"Accuracy for delta = {delta} is {accuracy}")

print(f"Best delta value is {best_delta} with accuracy {best_accuracy}")


#Using the testing set, get a list of predicted pages
predicted = []
for page in test_data:
    predicted.append(classify(page, best_delta))


#Generate a confusion matrix and classification report for the testing set
#Uses the labels predicted above 
cm = confusion_matrix(test_labels, predicted)
print("Confusion Matrix for testing set : ")
print(cm)
print()
print("Classification Report : ")
report = classification_report(test_labels, predicted, target_names=['HP1', 'HP2', 'HP3', 'HP4', 'HP5', 'HP6', 'HP7'])
print(report)

#Plot and save the confusion matrix
class_labels = ['HP1', 'HP2', 'HP3', 'HP4', 'HP5', 'HP6', 'HP7']
plt.figure(figsize=(8, 6))
sns.set(font_scale=1.2)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, annot_kws={'size': 14}, square=True,
            xticklabels=class_labels, yticklabels=class_labels)
plt.xlabel('Predicted Labels', fontsize=14)
plt.ylabel('True Labels', fontsize=14)
plt.title('Confusion Matrix', fontsize=16)
plt.savefig('confusion_matrix.png', dpi=300)


#Create a plot for the classification report generated above
report = classification_report(test_labels, predicted, target_names=['HP1', 'HP2', 'HP3', 'HP4', 'HP5', 'HP6', 'HP7'], output_dict=True)
report_df = pd.DataFrame(report).transpose()

# Plot the precision, recall, and F1-score using a bar chart
plt.figure(figsize=(10, 6))
report_df[['precision', 'recall', 'f1-score']].plot(kind='bar', colormap='Set3', figsize=(10, 6))
plt.xticks(rotation=0)
plt.xlabel('Class', fontsize=14)
plt.ylabel('Score', fontsize=14)
plt.title('Classification Report', fontsize=16)
plt.legend(loc='lower right')
plt.tight_layout()
plt.savefig('classification_report.png', dpi=300)

#Plot the accuracy vs delta as an incremental plot
x = [1, 2, 3, 4, 5, 6, 7, 8, 9,10]  
plt.plot(x, accuracies, label = "different delta values")
plt.xlabel("Delta Value")  
plt.ylabel("Accuracy(%)")  
plt.title("Accuracy vs Delta Value")  
plt.xticks(x, deltas)  
plt.legend(loc='lower left')
plt.savefig("AccuracyPlot.png", dpi=300)


