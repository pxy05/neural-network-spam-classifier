import numpy as np

training_spam = np.loadtxt(open("data/training_spam.csv"), delimiter=",")

def estimate_log_class_priors(data):
    """
    Given a data set with binary response variable (0s and 1s) in the
    left-most column, calculate the logarithm of the empirical class priors,
    that is, the logarithm of the proportions of 0s and 1s:
        log(p(C=0)) and log(p(C=1))

    :param data: a two-dimensional numpy-array with shape = [n_samples, 1 + n_features]
                 the first column contains the binary response (coded as 0s and 1s).

    :return log_class_priors: a numpy array of length two
    """

    priors = data[:, 0]

    spam_no = np.sum(data[:, 0] == 0)
    ham_no = np.sum( data[:, 0] == 1)
    total_no = spam_no + ham_no
    
    log_class_priors = np.array([np.log(spam_no / total_no), np.log(ham_no / total_no)])

    ### YOUR CODE HERE...
    return log_class_priors

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
    k       = data.shape[1]
    spam    = data[data[:, 0] == 0]
    ham     = data[data[:, 0] == 1]
    spam    = spam[:, 1:]
    ham     = ham[:, 1:]

    spam_n_cw   = spam.sum(axis=0) + alpha
    spam_n_c    = spam_n_cw.sum() + k*alpha

    ham_n_cw   = ham.sum(axis=0) + alpha
    ham_n_c    = ham_n_cw.sum() + k*alpha 

    spam_theta  = spam_n_cw / spam_n_c
    ham_theta   = ham_n_cw / ham_n_c

    
    theta = np.array([np.log(spam_theta), np.log(ham_theta)])
            
    return theta

def predict(new_data, log_class_priors, log_class_conditional_likelihoods):
    """
    Given a new data set with binary features, predict the corresponding
    response for each instance (row) of the new_data set.

    :param new_data: a two-dimensional numpy-array with shape = [n_test_samples, n_features].
    :param log_class_priors: a numpy array of length 2.
    :param log_class_conditional_likelihoods: a numpy array of shape = [2, n_features].
        theta[j, i] corresponds to the logarithm of the probability of feature i appearing
        in a sample belonging to class j.
    :return class_predictions: a numpy array containing the class predictions for each row
        of new_data.
    """
    ### YOUR CODE HERE...
    
    # scores = new_data @ log_class_conditional_likelihoods.T

    scores = np.dot(new_data, log_class_conditional_likelihoods.T)
    scores += log_class_priors

    class_predictions = np.argmax(scores, axis=1)

    return class_predictions


log_class_priors = estimate_log_class_priors(training_spam)
true_classes = training_spam[:, 0]

def runTests(this_alpha: float, test_num: int) -> float:
    log_class_conditional_likelihoods = estimate_log_class_conditional_likelihoods(training_spam, this_alpha)
    class_predictions = predict(training_spam[:, 1:], log_class_priors, log_class_conditional_likelihoods)
    training_set_accuracy = np.mean(np.equal(class_predictions, true_classes))
    print(f"Test {test_num}: {training_set_accuracy}.")
    return training_set_accuracy
a = 0
test = 1
tests = []
while a < 1.0:
    val     = runTests(this_alpha = a, test_num = test)
    test    += 1
    a       += 0.01
    tests.append(val)

best = np.argmax(tests)
print(f"Best alpha: {best}")


