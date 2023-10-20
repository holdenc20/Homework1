import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from sklearn import metrics

def write_data(samples, path):
    """Writes the sample data to the file path provided.

    Args:
        samples (pandas.DataFrame): the sample data
        path (string): the path to the file containing sample data
    """
    samples.to_csv(path, sep=',', header=True, index=True)


def read_sample_data(file_path):
    """Reads the sample data from the file path provided.

    Args:
        file_path (string): the path to the file containing sample data

    Returns:
        pandas.DataFrame: the sample data
    """
    data = pd.read_csv(file_path, sep=',', header=0, index_col=0)
    return data


def generate_samples(mean1, cov1, mean2, cov2, mean3a, cov3a, mean3b, cov3b, p1, p2, p3):
    """Generates 10000 samples according to a given data distribution and saves the data.
    
    Args:
        mean1 (array): The mean of the first class.
        cov1 (array): The covariance matrix of the first class.
        mean2 (array): The mean of the second class.
        cov2 (array): The covariance matrix of the second class.
        mean3a (array): The mean of the third class.
        cov3a (array): The covariance matrix of the third class.
        mean3b (array): The mean of the third class.
        cov3b (array): The covariance matrix of the third class.
    """
    
    rng = np.random.default_rng()
    
    size = 10000
    
    size1 = p1 * size
    size2 = p2 * size
    size3a = p3 * size / 2
    size3b = p3 * size / 2 
    
    
    data1 = rng.multivariate_normal(mean = mean1, cov = cov1, size = int(size1))
    data2 = rng.multivariate_normal(mean = mean2, cov = cov2, size = int(size2))
    data3a = rng.multivariate_normal(mean = mean3a, cov = cov3a, size = int(size3a))
    data3b = rng.multivariate_normal(mean = mean3b, cov = cov3b, size = int(size3b))
    
    data1 = pd.DataFrame(data1, columns = ['x1', 'x2', 'x3'])
    data2 = pd.DataFrame(data2, columns = ['x1', 'x2', 'x3'])
    data3a = pd.DataFrame(data3a, columns = ['x1', 'x2', 'x3'])
    data3b = pd.DataFrame(data3b, columns = ['x1', 'x2', 'x3'])
    
    data1['label'] = 1
    data2['label'] = 2
    data3a['label'] = 3
    data3b['label'] = 3
    samples = pd.concat([data1, data2, data3a, data3b], ignore_index=True)
    return samples

def map_classifier(samples, distribution, loss_matrix = [[0, 1, 1], [1, 0, 1], [1, 1, 0]]):
    """Classifies samples based on the label with the minimum loss.

    Args:
        samples (pandas.Dataframe): The sample data.
        distribution (pandas.Dataframe): The Gaussian distribution of the sample data classes.
        loss_matrix (list, optional): Loss matrix. Defaults to [[0, 1, 1], [1, 0, 1], [1, 1, 0]].

    Returns:
        _type_: _description_
    """
    decisions = []

    for index, sample in samples.iterrows():
        x = sample[['x1', 'x2', 'x3']].to_numpy()
        decision = np.argmin([loss(1, x, distribution, loss_matrix),
                            loss(2, x, distribution, loss_matrix),
                            loss(3, x, distribution, loss_matrix),
                            loss(3, x, distribution, loss_matrix)])
        
        decisions.append(decision + 1)
        
    samples['decision'] = decisions
    return samples

#Risk(D=d|x) = sum(l=1 to 3) LossMatrix d,l * P(L=l|x)
def loss(l, x, distribution, loss_matrix):
    """Determines the loss of a sample for a given distribution and loss matrix.

    Args:
        l (integer): Class label.
        x (numpy.array): Sample data.
        distribution (pandas.Dataframe): The Gaussian distribution of the sample data classes.
        loss_matrix (list): The loss matrix.

    Returns:
        float: The loss of the sample for a given label.
    """
    loss = 0
    
    #sum of all of the losses for each class for a given label
    for index, gaussian in distribution.iterrows():
        if index != l-1:
            loss += loss_matrix[l - 1][gaussian['label'] - 1] * gaussian['p'] * multivariate_normal.pdf(x, gaussian['mean'], gaussian['cov'])
       
    return loss

def calculate_correct_predictions(predictions):
    """Calculates the number of correct predictions.

    Args:
        predictions (pandas.Dataframe): The sample data with the predicted labels.

    Returns:
        integer: The number of correct predictions.
    """
    correct = 0
    for index, row in predictions.iterrows():
        if row['label'] == row['decision']:
            correct += 1
    return correct    
    
def generate_confusion_matrix(predictions):
    """Generates a confusion matrix for the sample data.

    Args:
        predictions (pandas.Dataframe): The sample data with the predicted labels.
    """
    confusion_matrix = metrics.confusion_matrix(predictions['label'], predictions['decision'])
    cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = [1, 2, 3])
    cm_display.plot()
    plt.show()

def plot_data(predictions):
    """Plots the sample data.

    Args:
        predictions (pandas.Dataframe): The sample data with the predicted labels.
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    #the color of the sample depends if the label = decision
    #the shape of the sample depends on the class label
    class11 = predictions[(predictions['label'] == 1) & (predictions['decision'] == 1)]
    class1x = predictions[(predictions['label'] == 1) & (predictions['decision'] != 1)]
    class22 = predictions[(predictions['label'] == 2) & (predictions['decision'] == 2)]
    class2x = predictions[(predictions['label'] == 2) & (predictions['decision'] != 2)]
    class33 = predictions[(predictions['label'] == 3) & (predictions['decision'] == 3)]
    class3x = predictions[(predictions['label'] == 3) & (predictions['decision'] != 3)]
    
    ax.scatter(class11['x1'], class11['x2'], class11['x3'], c='g', marker='o')
    ax.scatter(class1x['x1'], class1x['x2'], class1x['x3'], c='r', marker='o')
    ax.scatter(class22['x1'], class22['x2'], class22['x3'], c='g', marker='^')
    ax.scatter(class2x['x1'], class2x['x2'], class2x['x3'], c='r', marker='^')
    ax.scatter(class33['x1'], class33['x2'], class33['x3'], c='g', marker='s')
    ax.scatter(class3x['x1'], class3x['x2'], class3x['x3'], c='r', marker='s')
    
    ax.set_xlabel('X1')
    ax.set_ylabel('X2')
    ax.set_zlabel('X3')
    plt.show()

if __name__ == '__main__':
    samples_path = 'samples.csv'
    p1 = 0.3
    p2 = 0.3
    p3 = 0.4
    
    mean1 = [-30, -12, -10]
    mean2 = [12, 14, 24]
    mean3a = [45, -34, 7]
    mean3b = [1, 35, -35]
    
    scaler = np.identity(3) * 5
    
    cov1 = [[25, -5, 3], 
            [-5, 10, -5],
            [3, -5, 10]]
    cov2 = [[15, -5, 0], 
            [-5, 13, 0], 
            [0, 0, 22]]
    cov3a = [[10, 3, -2], 
            [3, 20, 3], 
            [-2, 3, 20]]
    cov3b = [[10, 3, -2], 
            [3, 20, 3], 
            [-2, 3, 20]] 
    
    cov1 = np.matmul(np.matmul(scaler, cov1), scaler)
    cov2 = np.matmul(np.matmul(scaler, cov2), scaler)
    cov3a = np.matmul(np.matmul(scaler, cov3a), scaler)
    cov3b = np.matmul(np.matmul(scaler, cov3b), scaler)
    
    
    '''
    #calculate standard deviation of each covariance matrix
    #all standard deviations are around 30
    sd_cov1 = np.std(cov1)
    sd_cov2 = np.std(cov2)
    sd_cov3a = np.std(cov3a)
    sd_cov3b = np.std(cov3b)
        
    print("Standard Deviation of Covariance Matrix 1:", sd_cov1)
    print("Standard Deviation of Covariance Matrix 2:", sd_cov2)
    print("Standard Deviation of Covariance Matrix 3a:", sd_cov3a)
    print("Standard Deviation of Covariance Matrix 3b:", sd_cov3b)
    
    #calculate distance between means 
    #all distances are 60-90
    dist12 = np.linalg.norm(np.array(mean1) - np.array(mean2))
    dist13a = np.linalg.norm(np.array(mean1) - np.array(mean3a))
    dist23a = np.linalg.norm(np.array(mean2) - np.array(mean3a))
    dist13b = np.linalg.norm(np.array(mean1) - np.array(mean3b))
    dist23b = np.linalg.norm(np.array(mean2) - np.array(mean3b))
    dist3a3b = np.linalg.norm(np.array(mean3a) - np.array(mean3b))
    print("Distance between mean1 and mean2:", dist12)
    print("Distance between mean1 and mean3a:", dist13a)
    print("Distance between mean2 and mean3a:", dist23a)
    print("Distance between mean1 and mean3b:", dist13b)
    print("Distance between mean2 and mean3b:", dist23b)
    print("Distance between mean3a and mean3b:", dist3a3b)
    '''
    '''
    
    samples = generate_samples(mean1, cov1, mean2, cov2, mean3a, cov3a, mean3b, cov3b, p1, p2, p3)
    
    write_data(samples, samples_path)
    '''
    samples = read_sample_data(samples_path)
    save_path = 'predictions.csv'
    
    #Part A - Minimum Probability of Error Classifier (0-1 loss)
    distribution = pd.DataFrame(columns=['p', 'mean', 'cov', 'label'])
    distribution = distribution._append({'p': p1, 'mean': mean1, 'cov': cov1, 'label': 1}, ignore_index=True)
    distribution = distribution._append({'p': p2, 'mean': mean2, 'cov': cov2, 'label': 2}, ignore_index=True)
    distribution = distribution._append({'p': p3/2, 'mean': mean3a, 'cov': cov3a, 'label': 3}, ignore_index=True)
    distribution = distribution._append({'p': p3/2, 'mean': mean3b, 'cov': cov3b, 'label': 3}, ignore_index=True)
    
    predictions = map_classifier(samples, distribution)
    print(predictions)
    correct = calculate_correct_predictions(predictions)
    print ("Percent Correct Predictions:", correct)
    
    write_data(predictions, save_path)
    
    predictions = read_sample_data(save_path)
    
    # plot the data
    plot_data(predictions)
 
    #confusion matrix
    generate_confusion_matrix(predictions)
    
    #Part B - Minimum Probability of Error Classifier (0-1 loss)
    loss_matrix2 = [[0, 10, 10], [1, 0, 10], [1, 1, 0]]
    predictions = map_classifier(samples, distribution, loss_matrix2)
    correct = calculate_correct_predictions(predictions)
    print ("Percent Correct Predictions:", correct)
    # plot the data
    plot_data(predictions)
 
    #confusion matrix
    generate_confusion_matrix(predictions)
    
    loss_matrix3 = [[0, 100, 100], [1, 0, 100], [1, 1, 0]]
    predictions = map_classifier(samples, distribution, loss_matrix3)
    correct = calculate_correct_predictions(predictions)
    print ("Percent Correct Predictions:", correct)
    # plot the data
    plot_data(predictions)
 
    #confusion matrix
    generate_confusion_matrix(predictions)