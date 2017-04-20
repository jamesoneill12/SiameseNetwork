from SiameseNetwork.dtw import *
from sklearn.metrics.pairwise import manhattan_distances
from sklearn.metrics.pairwise import euclidean_distances
dist_fun = euclidean_distances
import matplotlib.pyplot as plt
from nltk.metrics.distance import edit_distance
import pandas as pd
from sklearn import linear_model
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor
import numpy as np
import cPickle as pickle


root = "C:/Users/1/James/Research/Projects/DTW_Similarity_Project/Datasets/Sentence Similarity/dataset-sts/data/sts/sick2014/"
trainpath = "SICK_train.txt"
testpath = "SICK_test_annotated.txt"

with open(r"testsword2vec.pkl", "rb") as input_file:
   word_vectors = pickle.load(input_file)

train = pd.read_csv(root+trainpath,sep="\t")
test = pd.read_csv(root+testpath,sep="\t")


def emb(corpus,word_vectors):
    vector = [] ; vectors = [] ; lines = []
    for line in corpus:
        pad = 30 - len(line.split())
        lines.append(line.split() + ['<\PAD>']*pad)
        for word in line.split():
            try:
                vector.append(word_vectors[word])
            except:
                vector.append(np.zeros(100))
        vectors.append(np.array(vector))
    return np.array(vectors)

print emb(train['sentence_A'].tolist(),word_vectors).shape

train_sentenceB = [line.split() for line in train['sentence_B'].tolist()]
test_sentenceA = [line.split() for line in test['sentence_A'].tolist()]
test_sentenceB = [line.split() for line in test['sentence_B'].tolist()]
y_train = train['relatedness_score']
y_test = test['relatedness_score']

dist_fun = edit_distance
X_train = []
X_test = []

for x,y in zip(train_sentenceA,train_sentenceB):
    dist1, cost1, acc1, path1 = dtw(x, y, dist_fun)
    X_train.append(dist1)
X_train = np.asarray(X_train)
X_train= X_train[:,np.newaxis]

for x, y in zip(test_sentenceA, test_sentenceB):
    dist2, cost2, acc2, path2 = dtw(x, y, dist_fun)
    X_test.append(dist2)
X_test = np.asarray(X_test)
X_test = X_train[:]

# Fit regression model
#regr_1 = DecisionTreeRegressor(max_depth=4)

#regr_2 = AdaBoostRegressor(DecisionTreeRegressor(max_depth=4),
#                           n_estimators=300, random_state=6)

#regr_1.fit(X_train, y_train)
#regr_2.fit(X_train, y_train)

# Create linear regression object
regr_1 = linear_model.LinearRegression()

# Train the model using the training sets
print X_train.shape
regr_1.fit(X_train, y_train)

# Predict
y_1 = regr_1.predict(X_test)
#y_2 = regr_2.predict(X_test)

# Plot the results
plt.figure()
plt.scatter(X_train, y_train, c="k", label="training samples")
plt.plot(X_test, y_1, c="g", label="n_estimators=1", linewidth=2)
#plt.plot(X_test, y_2, c="r", label="n_estimators=300", linewidth=2)
plt.xlabel("data")
plt.ylabel("target")
plt.title("Boosted Decision Tree Regression")
plt.legend()
plt.show()




