from ucimlrepo import fetch_ucirepo 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import scipy.stats as stats
from scipy.stats import multivariate_normal
from sklearn import metrics
import string

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

def read_txt_data(file_path):
    """Reads the sample data from the file path provided.

    Args:
        file_path (string): the path to the file containing sample data

    Returns:
        pandas.DataFrame: the sample data
    """
    data = pd.read_csv(file_path, sep=' ', header=0, index_col=0)
    return data


def calculate_gaussians(samples, label='quality', reg = 0.0001, start=3, end=10):
    """Calculates the Gaussian distributions for the sample data.

    Args:
        samples (pandas.Dataframe): The sample data.
        label (str, optional): _description_. Defaults to 'quality'.
        reg (float, optional): _description_. Defaults to 0.0001.
        start (int, optional): _description_. Defaults to 3.
        end (int, optional): _description_. Defaults to 10.

    Returns:
        _type_: _description_
    """
    distributions = pd.DataFrame(columns=['l', 'p', 'mean', 'cov'])
    total_samples = len(samples)
    for i in range(start, end, 1):
        class_data = samples[samples[label] == i]
        class_data = class_data.drop([label], axis=1)
        mean = class_data.mean(axis=0)
        cov = class_data.cov()
        #might need to add regularization to cov
        cov = cov + np.identity(len(cov)) * reg
        distributions = distributions._append({'l': i, 'p': len(class_data) / total_samples,'mean': mean, 'cov': cov}, ignore_index=True)
    
    return distributions
        


def map_classifier(samples, distribution, loss_matrix, start=3, end=10, label='quality'):
    """Classifies samples based on the label with the minimum loss.

    Args:
        samples (pandas.Dataframe): The sample data.
        distribution (pandas.Dataframe): The Gaussian distribution of the sample data classes.
        loss_matrix (list, optional): Loss matrix. Defaults to [[0, 1, 1], [1, 0, 1], [1, 1, 0]].

    Returns:
        pandas.Dataframe: The sample data with the predicted labels.
    """
    decisions = []
    for index, sample in samples.iterrows():
        print(index / len(samples))
        x = sample.drop([label])
        decision = 0
        min_val = math.inf
        for i in range(start, end, 1):
            l = loss(i, x, distribution, loss_matrix, start)
            if l < min_val:
                min_val = l
                decision = i
        
        decisions.append(decision)
        
    samples['decision'] = decisions
    return samples

def loss(l, x, distribution, loss_matrix, start):
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
    for _, gaussian in distribution.iterrows():
        mean = gaussian['mean']
        cov = gaussian['cov']
        
        loss += loss_matrix[l - start][gaussian['l'] - start] * gaussian['p'] * multivariate_normal.pdf(x, mean, cov)
    
    return loss

def calculate_correct_predictions(predictions, label ='quality'):
    """Calculates the number of correct predictions.

    Args:
        predictions (pandas.Dataframe): The sample data with the predicted labels.

    Returns:
        integer: The number of correct predictions.
    """
    correct = 0
    for index, row in predictions.iterrows():
        if row[label] == row['decision']:
            correct += 1
    return correct    
    
    
def generate_confusion_matrix(predictions, label ='quality', labels = [3, 4, 5, 6, 7, 8, 9]):
    """Generates a confusion matrix for the sample data.

    Args:
        predictions (pandas.Dataframe): The sample data with the predicted labels.
    """
    confusion_matrix = metrics.confusion_matrix(predictions[label], predictions['decision'])
    cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = labels)
    cm_display.plot()
    plt.show()

def plot_data(data, subset=['fixed_acidity', 'volatile_acidity', 'citric_acid'], start=3, end=10, label='quality'):
    """Plots the sample data.

    Args:
        predictions (pandas.Dataframe): The sample data with the predicted labels.
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    #the color of the sample depends if the label = decision
    #the shape of the sample depends on the class label
    markers = ['o', '^', 's', 'd', 'v', '<', '>']
    for i in range(3, 10, 1):
        class11 = data[(data[label] == i) & (data['decision'] == i)]
        class1x = data[(data[label] == i) & (data['decision'] != i)]
        ax.scatter(class11[subset[0]], class11[subset[1]], class11[subset[2]], c='g', marker=markers[i - 3])
        ax.scatter(class1x[subset[0]], class1x[subset[1]], class1x[subset[2]], c='r', marker=markers[i - 3])
    
    ax.set_xlabel(subset[0])
    ax.set_ylabel(subset[0])
    ax.set_zlabel(subset[0])
    plt.show()


def generate_loss_matrix(distributions, alpha=0.01):
    """
    Generate a loss matrix with regularization.

    Args:
        distributions (pd.Dataframe): The sample data distributions.
        alpha (float): A small real number (0 < alpha < 1) for regularization. Default is 0.01.

    Returns:
        np.array: The generated loss matrix.
    """
    '''average_cov_matrix = np.mean(distributions['cov'].to_numpy(), axis=0)
    
    regularization_term = alpha * np.trace(average_cov_matrix) / np.linalg.matrix_rank(average_cov_matrix)
    loss_matrix = np.ones_like(average_cov_matrix) - np.identity(average_cov_matrix.shape[0])
    return loss_matrix * regularization_term'''
    
    num_classes = len(distributions)
    loss_matrix = np.zeros((num_classes, num_classes))
    
    for i in range(num_classes):
        cov_i = distributions['cov'][i]        
        reg_term_i = alpha * np.trace(cov_i) / np.linalg.matrix_rank(cov_i)
        
        for j in range(num_classes):
            if i != j:
                cov_j = distributions['cov'][j]
                reg_term_j = alpha * np.trace(cov_j) / np.linalg.matrix_rank(cov_j)
                loss_matrix[i][j] = (reg_term_i + reg_term_j) / 2
    return loss_matrix

if __name__ == "__main__":
    
    # fetch dataset 
    #wine_quality = fetch_ucirepo(id=186) 
    
    save_path = 'winequality.csv'
    
    # data (as pandas dataframes) 
    #X = wine_quality.data.features
    #y = wine_quality.data.targets
    
    #samples = pd.concat([X, y], axis=1)
    #write_data(samples, save_path)
    '''samples = read_sample_data(save_path)
    distributions = calculate_gaussians(samples)
    loss_matrix = generate_loss_matrix(distributions)
    samples = map_classifier(samples, distributions, loss_matrix)
    write_data(samples, 'winequality_predictions.csv')
    print(loss_matrix)
    samples = read_sample_data('winequality_predictions.csv')
    correct = calculate_correct_predictions(samples)
    print(correct / len(samples))
    generate_confusion_matrix(samples)
    plot_data(samples)'''
    
    #Human Activity Recognition
    x_test_path = './homework1/question3/UCI HAR Dataset/UCI HAR Dataset/test/X_test.txt'
    y_test_path = './homework1/question3/UCI HAR Dataset/UCI HAR Dataset/test/y_test.txt'
    x_train_path = './homework1/question3/UCI HAR Dataset/UCI HAR Dataset/train/X_train.txt'
    y_train_path = './homework1/question3/UCI HAR Dataset/UCI HAR Dataset/train/y_train.txt'
    features_path = './homework1/question3/UCI HAR Dataset/UCI HAR Dataset/features.txt'
    '''
    features = pd.read_csv(features_path, sep=' ', header=None, names=['Index', 'Feature'])

    x_train = pd.read_csv(x_train_path, delim_whitespace=True, header=None)
    y_train = pd.read_csv(y_train_path, delim_whitespace=True, header=None, names=['Activity'])

    x_test = pd.read_csv(x_test_path, delim_whitespace=True, header=None)
    y_test = pd.read_csv(y_test_path, delim_whitespace=True, header=None, names=['Activity'])

    # Add feature labels to the training and testing data
    x_train.columns = features['Feature'].values
    x_test.columns = features['Feature'].values

    samples = pd.concat([x_train, y_train], axis=1)
    distributions = calculate_gaussians(samples, 'Activity', 0.01, start=1, end=7)
    print(distributions)
    loss_matrix = generate_loss_matrix(distributions)
    print(loss_matrix)
    
    testing_data = pd.concat([x_test, y_test], axis=1)
    testing_data = map_classifier(testing_data, distributions, loss_matrix, start=1, end=7, label='Activity')
    write_data(testing_data, 'activity_predictions_withtesting.csv')
    '''
    testing_data = read_sample_data('activity_predictions_withtesting.csv')
    correct = calculate_correct_predictions(testing_data, 'Activity')
    generate_confusion_matrix(testing_data, 'Activity', [1, 2, 3, 4, 5, 6])
    plot_data(testing_data, ['tBodyAcc-mean()-X', 'tBodyAcc-mean()-Y', 'tBodyAcc-mean()-Z'], start=1, end=7, label='Activity')
    