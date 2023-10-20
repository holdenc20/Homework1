import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

def generate_samples(mean0, cov0, mean1, cov1):
    """Generates 10000 samples according to a given data distribution and saves the data.
    
    Args:
        mean0 (list of integers): the mean of L=0
        cov0 (_type_): the covariance matrix of L=0
        mean1 (_type_): the mean of L=1
        cov1 (_type_): the covariance matrix of L=1
    """
    
    rng = np.random.default_rng()
    
    size = 10000
    p0 = 0.35 #P(L=0) 
    p1 = 0.65 #P(L=1)
    
    count0 = p0 * size
    
    data0 = rng.multivariate_normal(mean = mean0, cov = cov0, size = int(count0))
    data1 = rng.multivariate_normal(mean = mean1, cov = cov1, size = size - int(count0))
    
    data0 = pd.DataFrame(data0, columns = ['x1', 'x2', 'x3', 'x4'])
    data1 = pd.DataFrame(data1, columns = ['x1', 'x2', 'x3', 'x4'])
    data0['label'] = 0
    data1['label'] = 1
    samples = pd.concat([data0, data1], ignore_index=True)
    return samples

def read_sample_data(file_path):
    """Reads the sample data from the file path provided.

    Args:
        file_path (string): the path to the file containing sample data

    Returns:
        pandas.DataFrame: the sample data
    """
    data = pd.read_csv(file_path, sep=',', header=0, index_col=0)
    return data

def write_data(samples, path):
    """Writes the sample data to the file path provided.

    Args:
        samples (pandas.DataFrame): the sample data
        path (string): the path to the file containing sample data
    """
    samples.to_csv(path, sep=',', header=True, index=True)

def expected_risk_minimization_classifier(samples, mean0, mean1, cov0, cov1, gamma):
    """Classifies the samples using the expected risk minimization classifier.

    Args:
        samples (pandas.DataFrame): the sample data
        mean0 (list of integers): the mean of L=0
        mean1 (list of integers): the mean of L=1
        cov0 (_type_): the covariance matrix of L=0
        cov1 (_type_): the covariance matrix of L=1
        gamma (float): the decision threshold

    Returns:
        pandas.DataFrame: the sample data with the predicted labels
    """
    predictions = []
    for i in range(len(samples)):
        sample = samples.iloc[i]
        likelihood = calculate_class_posteriors(sample, mean0, mean1, cov0, cov1)
        #apply decision threshold
        prediction = 0
        if likelihood > gamma:
            prediction = 1
        predictions.append(prediction)
        
    samples['prediction'] = predictions
    
    return samples

def calculate_class_posteriors(sample, mean0, mean1, cov0, cov1):
    """Calculates the class posteriors for a given sample.

    Args:
        sample (pandas.Series): a sample
        mean0 (list of integers): the mean of L=0
        mean1 (list of integers): the mean of L=1
        cov0 (matrix of integers): the covariance matrix of L=0
        cov1 (matrix of integers): the covariance matrix of L=1

    Returns:
        float: likelihood ratio test
    """
    x = sample[["x1", "x2", "x3", "x4"]].to_numpy()
    
    likelihood_class0 = multivariate_normal.pdf(x, mean0, cov0)
    likelihood_class1 = multivariate_normal.pdf(x, mean1, cov1)
    
    likelihood = likelihood_class1 / likelihood_class0
    return likelihood

def calculate_prediction_error(predictions):
    """Calculates the prediction error for a given set of predictions.

    Args:
        predictions (pandas.DataFrame): the sample data with the predicted labels

    Returns:
        integer: true positives
        integer: false positives
        integer: false negatives
        integer: true negatives
    """
    data0 = predictions[predictions['label'] == 0]
    data1 = predictions[predictions['label'] == 1]
    
    true_positives = len(data1[data1['prediction'] == 1])
    false_positives = len(data0[data0['prediction'] == 1])
    false_negatives = len(data1[data1['prediction'] == 0])
    true_negatives = len(data0[data0['prediction'] == 0])
    
    return true_positives, false_positives, false_negatives, true_negatives
    
if __name__ == "__main__":
    path = './sample_data1.csv'
    pred_path = './predictions1.csv'
    results_path = './results1.csv'
    results_path_fake = './results1_fake.csv'
    LDA_results_path = './LDA_results1.csv'
    mean0 = [-1, -1, -1, -1]
    mean1 = [1, 1, 1, 1]
    cov0 = [[2, -0.5, 0.3, 0], [-0.5, 1, -0.5, 0], [0.3, -0.5, 1, 0], [0, 0, 0, 2]]
    cov1 = [[1, 0.3, -0.2, 0], [0.3, 2, 0.3, 0], [-0.2, 0.3, 1, 0], [0, 0, 0, 3]]
    '''
    samples = generate_samples(mean0, cov0, mean1, cov1)
    write_data(samples, path)
    '''
    samples = read_sample_data(path)
    #1.A
    #1.A.1: Specify the minimum expected risk classification rule in the form of a likelihood-ratio test. - this will be in the report
    #1.A.2: Implement classifier 
    '''
    #examples prediction for gamma = 1.857
    gamma = 1.857
    predictions = expected_risk_minimization_classifier(samples, mean0, mean1, cov0, cov1, gamma)
    write_data(predictions, pred_path)
    
    calculated_error = calculate_prediction_error(predictions)
    print(calculated_error)
    '''
    
    '''
    #now for varying gamma 
    gamma = np.exp(np.linspace(-30, 30, 1000))
    results = pd.DataFrame(columns=['gamma', 'true_positives', 'false_positives', 'false_negatives', 'true_negatives', 'error'])
    for threshold in gamma:
        predictions = expected_risk_minimization_classifier(samples, mean0, mean1, cov0, cov1, threshold)
        calculated_error = calculate_prediction_error(predictions)
        true_positive = calculated_error[0]/(0.65 * 10000)
        false_positive = calculated_error[1]/(0.35 * 10000)
        false_negatives = calculated_error[2]/(0.65 * 10000)
        true_negatives = calculated_error[3]/(0.35 * 10000)
        error = false_positive * 0.35 + false_negatives * 0.65
        print("gamma: ", threshold, ", true pos: ", true_positive, ", false positive: ", false_positive, ", error: ",  error)
        results = results._append({'gamma': threshold, 'true_positives': true_positive, 'false_positives': false_positive, 'false_negatives': false_negatives, 'true_negatives': true_negatives, 'error': error}, ignore_index=True)
        
    write_data(results, results_path)
    '''
    
    
    results = read_sample_data(results_path)
    
    min_threshold = results[results['error'] == results['error'].min()]
    print("min threshold: ", min_threshold['gamma'].iloc[0])
    print("true pos at min: ", min_threshold['true_positives'].iloc[0])
    print("false pos at min: ", min_threshold['false_positives'].iloc[0])
    
    perror = min_threshold['false_positives'].iloc[0] * 0.35 + min_threshold['false_negatives'].iloc[0] * 0.65
    print ("P error at min: ", perror)
    
    theoretical_gamma = 0.35/0.65
    theoretical_pred = expected_risk_minimization_classifier(samples, mean0, mean1, cov0, cov1, theoretical_gamma)
    error = calculate_prediction_error(theoretical_pred)
    theoretical_false_positive = error[1]/(0.35 * 10000)
    theoretical_true_positive = error[0]/(0.65 * 10000)
    
    fig, ax = plt.subplots()
    ax.plot(results['false_positives'], results['true_positives'], label='ROC curve')
    ax.plot(min_threshold['false_positives'], min_threshold['true_positives'], 'ro', label='Experimental', markersize=10)
    ax.plot(theoretical_false_positive, theoretical_true_positive, 'go', label='Theoretical', markersize=10)
    
    
    ax.set_xlabel('False Positive Probability')
    ax.set_ylabel('True Positive Probability')
    ax.set_title('ROC Curve')
    
    ax.legend(title="Minimum Probability of Error", loc='lower right')
  
    
   
    plt.savefig('./roc_curve.png')
    plt.show()
    
    print("Theoretical gamma: ", theoretical_gamma, ", error: ", theoretical_false_positive * 0.65 + (1 - theoretical_true_positive) * 0.35)
    print("Experimental gamma: ", min_threshold['gamma'].iloc[0], ", error: ", min_threshold['error'].iloc[0])
    
    
    #Part B
    print("Part B")
    cov0_fake = np.diagonal(cov0) * np.identity(4)
    cov1_fake = np.diagonal(cov1) * np.identity(4)
    '''
    #now for varying gamma 
    gamma = np.exp(np.linspace(-30, 30, 1000))
    results = pd.DataFrame(columns=['gamma', 'true_positives', 'false_positives', 'false_negatives', 'true_negatives', 'error'])
    for threshold in gamma:
        predictions = expected_risk_minimization_classifier(samples, mean0, mean1, cov0_fake, cov1_fake, threshold)
        calculated_error = calculate_prediction_error(predictions)
        true_positive = calculated_error[0]/(0.65 * 10000)
        false_positive = calculated_error[1]/(0.35 * 10000)
        false_negatives = calculated_error[2]/(0.65 * 10000)
        true_negatives = calculated_error[3]/(0.35 * 10000)
        error = false_positive * 0.35 + false_negatives * 0.65
        print("gamma: ", threshold, ", true pos: ", true_positive, ", false positive: ", false_positive, ", error: ",  error)
        results = results._append({'gamma': threshold, 'true_positives': true_positive, 'false_positives': false_positive, 'false_negatives': false_negatives, 'true_negatives': true_negatives, 'error': error}, ignore_index=True)
        
    write_data(results, results_path_fake)
    '''
    results = read_sample_data(results_path_fake)
    
    min_threshold = results[results['error'] == results['error'].min()].median()
    print("min threshold: ", min_threshold['gamma'])
    print("true pos at min: ", min_threshold['true_positives'])
    print("false pos at min: ", min_threshold['false_positives'])
    
    perror = min_threshold['false_positives'] * 0.35 + min_threshold['false_negatives'] * 0.65
    print ("P error at min: ", perror)
    
    print("Theoretical gamma: ", theoretical_gamma, ", error: ", theoretical_false_positive * 0.65 + (1 - theoretical_true_positive) * 0.35)
    print("Experimental gamma: ", min_threshold['gamma'], ", error: ", perror)
    
    theoretical_gamma = 0.35/0.65
    theoretical_pred = expected_risk_minimization_classifier(samples, mean0, mean1, cov0_fake, cov1_fake, theoretical_gamma)
    error = calculate_prediction_error(theoretical_pred)
    theoretical_false_positive = error[1]/(0.35 * 10000)
    theoretical_true_positive = error[0]/(0.65 * 10000)
    
    fig, ax = plt.subplots()
    ax.plot(results['false_positives'], results['true_positives'], label='ROC curve')
    ax.plot(min_threshold['false_positives'], min_threshold['true_positives'], 'ro', label='Experimental', markersize=10)
    ax.plot(theoretical_false_positive, theoretical_true_positive, 'go', label='Theoretical', markersize=10)
    
    
    ax.set_xlabel('False Positive Probability')
    ax.set_ylabel('True Positive Probability')
    ax.set_title('ROC Curve')
    
    ax.legend(title="Minimum Probability of Error", loc='lower right')
    plt.savefig('./roc_curve_fake.png')
    plt.show()
    
    #Part C
    class0 = samples[samples['label'] == 0]
    class1 = samples[samples['label'] == 1]
    
    x0 = class0[["x1", "x2", "x3", "x4"]].to_numpy()
    x1 = class1[["x1", "x2", "x3", "x4"]].to_numpy()
    mean_class0 = np.mean(x0, axis=0)
    mean_class1 = np.mean(x1, axis=0)
    cov_class0 = np.cov(x0, rowvar=False)
    cov_class1 = np.cov(x1, rowvar=False)
    
    #Within class scatter matrix
    SW = 0.5 * cov_class0 + 0.5 * cov_class1
    
    #Between class scatter matrix
    m = 0.5 * mean_class0 + 0.5 * mean_class1
    SB = 0.5 * np.outer(mean_class0 - m, mean_class0 - m) + 0.5 * np.outer(mean_class1 - m, mean_class1 - m)
    
    #Generalized eigendeomposition
    eigvals, eigvecs = np.linalg.eig(np.linalg.inv(SW).dot(SB))
    
    #wLDA
    wLDA = eigvecs[:, np.argmax(eigvals)]
    #normalize wLDA
    wLDA = wLDA / np.linalg.norm(wLDA)
    print("wLDA: ", wLDA)

    results = pd.DataFrame(columns=['gamma', 'true_positives', 'false_positives', 'error'])
    gamma = np.linspace(-6, 6, 100) # should be -inf to inf
    for threshold in gamma:
        predictions = (np.dot(samples[["x1", "x2", "x3", "x4"]].to_numpy(), wLDA) < threshold).astype(int)
        samples['prediction'] = predictions
        tp, fp, fn, tn = calculate_prediction_error(samples)

        error = fp/(0.35 * 10000) * 0.35 + fn/(0.65 * 10000) * 0.65
        results = results._append({'gamma': threshold, 'true_positives': tp/(0.65 * 10000), 'false_positives': fp/(0.35 * 10000), 'error': error}, ignore_index=True)
        
        
    min_threshold = results[results['error'] == results['error'].min()].median()
    print(min_threshold)
    
    write_data(results, LDA_results_path)
    plt.figure(figsize=(8, 6))
    plt.plot(results['false_positives'], results['true_positives'], lw=2, label='ROC Curve (Fisher LDA)')
    plt.plot(min_threshold['false_positives'], min_threshold['true_positives'], 'ro', label='Experimental', markersize=10)

    plt.title('ROC Curve')
    plt.xlabel('False Positive Probability')
    plt.ylabel('True Positive Probability')
    plt.legend()
    plt.show()