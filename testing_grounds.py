import numpy as np
import spam_net_final as NN

training_spam = np.loadtxt(open("data/training_spam.csv"), delimiter=",")
testing_spam = np.loadtxt(open("data/testing_spam.csv"), delimiter=",")

training_spam_X = training_spam[:, 1:]
training_spam_Y = training_spam[:, 0].reshape(1, -1) # reshape to 1 row and n columns

testing_spam_X = testing_spam[:, 1:]
testing_spam_Y = testing_spam[:, 0].reshape(1, -1) # reshape to 1 row and n columns

def accuracy(predictions, true_classes):
    """
    Calculate the accuracy of predictions against true classes.
    """
    return np.mean(predictions == true_classes) * 100

classifier = NN.spamClassifier()
classifier.train(training_spam_X.T, training_spam_Y, iterations=1000, learning_rate=0.01, print_cost=True)
predictions = classifier.predict(testing_spam_X.T)

print(f"Accuracy: {accuracy(predictions, testing_spam_Y):.2f}%")