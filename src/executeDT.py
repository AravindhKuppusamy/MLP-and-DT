"""
File: executeDT.py
Language: Python 3.5.1
Author: Aravindh Kuppusamy ( axk8776@rit.edu )
        Karan Jariwala( kkj1811@rit.edu )
        Deepak Sharma ( ds5930@rit.edu )
Description: Classifies the data based on the provided decision tree
"""

from trainDT import Tree, loadData, classify
import sys
import numpy as np
import matplotlib.pyplot as plt

__author__ = "Aravindh Kuppusamy, Deepak Sharma, Karan Jariwala"


DATA_TRAIN = 'train_data.csv'
DATA_TEST = "test_data.csv"
CLASSES = 4
CLASS_LIST = ["Bolts", "Nuts", "Rings", "Scraps"]


def treeAsList(treeFile):
    """
    Given the csv file containing the tree in DFS traversal, it returns it as a list.
    :param treeFile: csv file containing the tree in DFS traversal
    :return: list of nodes in the tree in DFS traversal order
    """
    with open(treeFile) as f:
        nodeList = f.read().strip().split(',')
        return nodeList


def createTree(nodeList, node):
    """
    Given the csv file containing the tree in DFS traversal, it returns it as a list.
    :param nodeList: list of nodes in the tree in DFS traversal order
    :param node: Root node of the tree
    :return: remaining nodes in list of nodes in the tree in DFS traversal order
    """
    if len(nodeList) == 0:
        return
    if nodeList[0] == '!':
        return nodeList[1:]

    splitAtt, splitVal, value = nodeList[0].strip().split('-')
    nodeList = nodeList[1:]

    if splitAtt == splitVal == '$':
        node.setSplit("Base case")
        node.setValue(value)
        return nodeList[2:]
    else:
        node.setSplit((splitAtt, splitVal))

    leftChild = Tree()
    node.setLesser(leftChild)
    nodeList = createTree(nodeList, leftChild)

    rightChild = Tree()
    node.setGreater(rightChild)
    nodeList = createTree(nodeList, rightChild)

    return nodeList


def printTree(node, depth=0):
    """
    Given the root node of the tree it prints the tree.
    :param depth: depth of current node in the tree
    :param node: Root or current node of the tree
    """
    if node:
        print("  " * depth, node.split, "CLASS:", node.value)
        printTree(node.lesser, depth + 2)
        printTree(node.greater, depth + 2)


def decision_boundary(treeRoot, figure, data_file):
    """
    It plots a graph of decision boundary and data points
    :param treeRoot: Root node of the decision tree
    :param figure: figure in plot
    :param data_file: data file
    :return: decision plot
    """
    decision_plot = figure.add_subplot(111)
    attribute, label, _ = loadData(data_file)
    attribute, label = np.array(attribute), np.array(label)
    classes = [1, 2, 3, 4]
    colors_box = ['y', 'b', 'g', 'm']
    marker_box = ['*', '+', 'x', 'o']
    step = .001

    x1_corr, x2_corr = np.meshgrid(np.arange(0, 1, step), np.arange(0, 1, step))

    Y_predicted = []
    for i in range(x1_corr.shape[0]):
        Y_predicted.append([])
        for j in range(x1_corr.shape[1]):
            sample = [x1_corr[i][j], x2_corr[i][j]]
            predicted = classify(np.array(sample), treeRoot)
            Y_predicted[i].append(predicted)

    decision_plot.contourf(x1_corr, x2_corr, np.array(Y_predicted))

    for index in classes:
        x1 = [attribute[i][0] for i in range(len(attribute[:]))
              if label[i] == index]
        x2 = [attribute[i][1] for i in range(len(attribute[:]))
              if label[i] == index]
        decision_plot.scatter(x1, x2, color=colors_box[index - 1], marker=marker_box[index - 1]
                              , label=CLASS_LIST[index - 1], s=100)

    decision_plot.legend(loc='upper right')
    decision_plot.set_xlabel("Six fold Rotational Symmetry")
    decision_plot.set_ylabel("Eccentricity")
    decision_plot.set_title("Decision boundary")
    return decision_plot


def get_confusion_matrix(rootNode, data_file):
    """
    Construct confusion matrix by predicting class for given set of data and MLP
    :param rootNode: root node of the decision tree
    :param data_file: CSV data file
    :return: confusion matrix List of List
    """
    attribute, label, _ = loadData(data_file)
    attribute = np.array(attribute)

    confusion_matrix = []

    for _ in range(CLASSES):
        confusion_matrix.append([])
        for _ in range(CLASSES):
            confusion_matrix[-1].append(0)
    for sample_counter in range(attribute.shape[0]):
        actual_class = label[sample_counter]
        predicted_class = classify(attribute[sample_counter], rootNode)
        confusion_matrix[int(predicted_class) - 1][int(actual_class) - 1] += 1

    print_data(confusion_matrix)
    return confusion_matrix


def print_data(confusion):
    """
    Prints the results achieved after classification
    :param confusion: A confusion matrix
    :return: None
    """
    confusion = np.array(confusion)
    digit = 5
    print("\n--------------------------------- Confusion Matrix ----------------------------------")
    print("Pre/Act ", "*\t",
          padding("Class 1 ", digit), "\t|\t",
          padding("Class 2 ", digit), "\t|\t",
          padding("Class 3 ", digit), "\t|\t",
          padding("Class 4 ", digit), "\t*\tTotal")
    for counter in range(4):
        print("Class " + str(counter + 1), " *\t",
              padding(confusion[counter][0], digit), "\t\t|\t",
              padding(confusion[counter][1], digit), "\t\t|\t",
              padding(confusion[counter][2], digit), "\t\t|\t",
              padding(confusion[counter][3], digit), "\t\t*\t",
              padding(np.sum(confusion[counter]), digit))
    print("-------------------------------------------------------------------------------------")
    print("Total    *\t",
          padding(np.sum(confusion[:, 0]), digit), "\t\t|\t",
          padding(np.sum(confusion[:, 1]), digit), "\t\t|\t",
          padding(np.sum(confusion[:, 2]), digit), "\t\t|\t",
          padding(np.sum(confusion[:, 3]), digit), "\t\t*\t",
          padding(np.sum(confusion[:, 0:4]), digit))
    print("-------------------------------------------------------------------------------------\n")


def padding(input_num, digit):
    """
    Helper method for padding the input number so that program can maintain
    symmetry in tables
    :param input_num: a number
    :param digit: number of digits required
    :return: Padded value
    """
    padded = str(input_num)
    for padding_counter in range(digit - len(str(input_num))):
        padded += " "
    return padded


def misclassification(dataFile, rootNode):
    """
    Report number of correctly and incorrectly classified samples in the provided data set
    :param dataFile: datafile/data set
    :param rootNode: trained root node of decision tree
    :return: Number of misclassified and correctly classified samples
    """
    miss_prediction = [0, 0, 0, 0]
    correct_prediction = [0, 0, 0, 0]
    correct = 0
    incorrect = 0
    attribute, label, _ = loadData(dataFile)
    attribute = np.array(attribute)
    for sample_index in range(attribute.shape[0]):

        prediction = classify(attribute[sample_index], rootNode)
        if int(label[sample_index]) == int(prediction):
            correct_prediction[label[sample_index] - 1] += 1
            correct += 1
        else:
            miss_prediction[label[sample_index] - 1] += 1
            incorrect += 1
    accuracy = correct / (correct + incorrect)
    mean_per_class_accuracy = 0
    for counter in range(CLASSES):
        correct = correct_prediction[counter]
        incorrect = miss_prediction[counter]
        mean_per_class_accuracy += correct / (CLASSES * (correct + incorrect))
    print("\n--------------------------------- Recognition Rate ----------------------------------")
    print("Total Accuracy           : ", accuracy)
    print("Mean Per Class Accuracy  : ", mean_per_class_accuracy)
    print("-------------------------------------------------------------------------------------\n")

    return miss_prediction, correct_prediction


def profit(confusion_matrix):
    """
    Calculates and prints the profit.
    :param confusion_matrix: confusion matrix
    :return: None
    """
    profit_values = [[20, -7, -7, -7], [-7, 15, -7, -7], [-7, -7, 5, -7], [-3, -3, -3, -3]]
    total_profit = 0
    for counter in range(CLASSES):
        for inner_counter in range(CLASSES):
            total_profit += confusion_matrix[counter][inner_counter] * \
                            profit_values[counter][inner_counter]

    print("\n-------------------------------------- Profit ---------------------------------------")
    print("Total Profit     : ", total_profit)
    print("-------------------------------------------------------------------------------------\n")


def main():
    """
    Main method
    return: None
    """
    if len(sys.argv) != 2:
        print("FAILED: Please provide the proper python syntax")
        print("USAGE: python3 executeDT <DT_file.csv>")
        sys.exit(1)

    traversalList = treeAsList(sys.argv[1])
    rootNode = Tree()
    createTree(traversalList, rootNode)
    print("\n********************************* DECISION TREE *************************************\n")
    printTree(rootNode)
    print("\n*************************************************************************************")

    print("\n************************************* TRAINING **************************************\n")
    misclassification(DATA_TRAIN, rootNode)
    confusion_matrix = get_confusion_matrix(rootNode, DATA_TRAIN)
    profit(confusion_matrix)

    print("\n************************************* TESTING ***************************************\n")
    misclassification(DATA_TEST, rootNode)
    confusion_matrix = get_confusion_matrix(rootNode, DATA_TEST)
    profit(confusion_matrix)
    print("\n*************************************** END *****************************************\n")

    figure = plt.figure()
    decision_boundary(rootNode, figure, DATA_TEST)
    plt.show()


if __name__ == "__main__":
    main()
