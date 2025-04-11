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

    spam = data[data[:, 0] == 0]
    ham = data[data[:, 0] == 1]
    spam = spam[:, 1:]
    ham = ham[:, 1:]

    spam_features_sum = spam.sum(axis=0) + alpha
    ham_features_sum = ham.sum(axis=0) + alpha

    spam_prob_denominator = spam.shape[0] + alpha
    ham_prob_denominator = ham.shape[0] + alpha

    spam_probs = np.log(spam_features_sum/ spam_prob_denominator)
    ham_probs = np.log(ham_features_sum / ham_prob_denominator)

    theta = np.array([spam_probs, ham_probs])
            
    return theta

estimate_log_class_conditional_likelihoods(training_spam, alpha=1.0)