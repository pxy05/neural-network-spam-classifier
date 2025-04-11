import numpy as np

training_spam = np.loadtxt(open("data/training_spam.csv"), delimiter=",")

def estimate_log_class_conditional_likelihoods(data, alpha=1.0):
    """
    Given a data set with binary response variable (0s and 1s) in the
    left-most column and binary features (words), calculate the empirical
    class-conditional likelihoods, that is,
    log(P(w_i | c)) for all features w_i and both classes (c in {0, 1}).

    Assume a multinomial feature distribution and use Laplace smoothing
    if alpha > 0.

    :param data: a two-dimensional numpy-array with shape = [n_samples, 1 + n_features]

    :return theta:
        a numpy array of shape = [2, n_features]. theta[j, i] corresponds to the
        logarithm of the probability of feature i appearing in a sample belonging 
        to class j.
    """

    spam = np.array(data[data[0] == 0])
    ham = np.array(data[data[0] == 1])

    spam_features = np.array([])
    ham_features = np.array([])


    spam_features = np.sum(spam[:1], axis=0)
    ham_features = np.sum(ham[:1], axis=0)

    print ("Spam features:", spam_features)
    print ("Ham features:", ham_features)

estimate_log_class_conditional_likelihoods(training_spam, alpha=1.0)