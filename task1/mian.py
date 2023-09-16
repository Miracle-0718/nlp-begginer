import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

from feature import BagOfWords,Ngram
from softmax_regression import Softmax_regression

if __name__ == '__main__':
    #导入数据
    train_df = pd.read_csv('../data/train.tsv', sep='\t')
    X_data,y_data = train_df["Phrase"].values, train_df["Sentiment"].values

    test = 1000
    X_data = X_data[:test]
    y_data = y_data[:test]

    bag = BagOfWords(X_data)
    gram = Ngram(X_data,N=2)

    X_b = bag.get_feature_matrix()
    X_g = gram.get_feature_matrix()

    y = np.array(y_data).reshape((-1, 1))   
    X_train_b, X_test_b, y_train_b, y_test_b = train_test_split(X_b, y, test_size=0.2, random_state=42, stratify=y)   #按y中各类比例，分配给train和test
    X_train_g, X_test_g, y_train_g, y_test_g = train_test_split(X_g, y, test_size=0.2, random_state=42, stratify=y) 

    epochs = 100
    learning_rate = 1

    model1 = Softmax_regression()
    loss1 = model1.fit(X_train_b,y_train_b,learning_rate=learning_rate,epochs=epochs)
    plt.title('Bag of words')
    plt.plot(np.arange(len(loss1)), np.array(loss1))
    plt.show()
    print("Bow train {} test {}".format(model1.score(X_train_b, y_train_b), model1.score(X_test_b, y_test_b)))

    model2 = Softmax_regression()
    loss2 = model2.fit(X_train_g,y_train_g,learning_rate=learning_rate,epochs=epochs)
    plt.title('N - gram')
    plt.plot(np.arange(len(loss2)), np.array(loss2))
    plt.show()
    print("N-gram train {} test {}".format(model2.score(X_train_g, y_train_g), model2.score(X_test_g, y_test_g)))