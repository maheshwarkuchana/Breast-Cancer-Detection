import pandas as pd
import numpy as np

df = pd.read_csv("data.csv")

from sklearn.preprocessing import LabelEncoder, scale
encoder = LabelEncoder()
df["Outcome"] = encoder.fit_transform(df["Outcome"])

X = df.iloc[:, 2:]
X = scale(X)
Y = df["Outcome"]

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, Y, random_state=20, test_size=0.2)

# from sklearn.linear_model import LogisticRegression
# clf = LogisticRegression(random_state=0)

# from sklearn.neural_network import MLPClassifier
# clf = MLPClassifier()

# from sklearn.neighbors import KNeighborsClassifier
# clf = KNeighborsClassifier()

# from sklearn.svm import SVC
# clf = SVC(kernel='linear', random_state=0)
# clf = SVC(kernel='rbf', random_state=0)

# from sklearn.naive_bayes import GaussianNB
# clf = GaussianNB()

# from sklearn.ensemble import RandomForestClassifier
# clf = RandomForestClassifier(n_estimators=10, criterion='entropy', random_state=0)

from keras.models import Sequential
from keras.layers import Dense
from keras.utils import np_utils

clf = Sequential()
clf.add(Dense(units=100, input_dim=30, activation='relu'))
clf.add(Dense(units=75, activation='relu'))
clf.add(Dense(units=2, activation='sigmoid'))

y_train = np_utils.to_categorical(y_train)

clf.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
clf.fit(X_train, y_train, epochs=40, batch_size=10)

y_pred = clf.predict(X_test)
y_pred = np.argmax(y_pred, axis=1)

# import pickle
# pickle.dump(clf, open("/Trained Models/NeuralNetwork_Keras_Model.pickle", "wb"))

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

print(confusion_matrix(y_true=y_test, y_pred=y_pred), end="\n\n")
print(accuracy_score(y_true=y_test, y_pred=y_pred), end="\n\n")
print(classification_report(y_true=y_test, y_pred=y_pred))
