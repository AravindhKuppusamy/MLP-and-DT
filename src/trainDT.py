"""
File: trainDT.py
Language: Python 3.5.1
Author: Aravindh Kuppusamy ( axk8776@rit.edu )
        Deepak Sharma ( ds5930@rit.edu )
        Karan Jariwala( kkj1811@rit.edu )
Description: Trains the decision tree based on the given samples
"""

# Importing python module
import math
import operator
import matplotlib.pyplot as plt
import numpy as np


__author__ = "Aravindh Kuppusamy, Deepak Sharma, and Karan Jariwala"


# Global Constants
DATA_TRAIN = 'train_data.csv'
DATA_TEST = "test_data.csv"
CLASS = 4
CLASS_LIST = ["Bolts", "Nuts", "Rings", "Scraps"]
SIGNIFICANCE = 0.01


class Tree:
    """
    Represents a node in the tree
    """

    __slots__ = ('value', 'greater', 'lesser',
                 'split', 'classCount', 'isLeaf')

    def __init__(self):
        """
        Initializing parameters requires for the node
        :return: None
        """
        self.value = None
        self.greater = None
        self.lesser = None
        self.split = None
        self.classCount = [0, 0, 0, 0]
        self.isLeaf = False

    def setGreater(self, node):
        """
        Sets the given node to greater link of the current node
        :param node: A node in the tree
        :return: None
        """
        self.greater = node

    def setLesser(self, node):
        """
        Sets the given node to lesser link of the current node
        :param node: A node in the tree
        :return: None
        """
        self.lesser = node

    def setValue(self, value):
        """
        Sets the current node to the given class value and assigns it as a leaf node
        :param value: Class of the current node
        :return: None
        """
        self.value = value
        self.isLeaf = True

    def setSplit(self, split):
        """
        Sets the split attribute and split value to the current node
        :param split: A tuple containing the split attribute and split value
        :return: None
        """
        self.split = split


def loadData(file):
    """
    Given the file, it returns the list of input attributes, class of samples,
     and combination of both
    :param file : File containing the training samples
    :return     : list of input attributes, class of samples, and combination of both
    """
    inputData = list()
    outputData = list()
    combinedData = list()
    with open(file) as f:
        for line in f:
            sample = line.strip().split(",")
            inputData += [[float(sample[0])] + [float(sample[1])]]
            outputData += [int(sample[2])]
            combinedData += [[float(sample[0])] + [float(sample[1])] + [int(sample[2])]]

    return inputData, outputData, combinedData


def writeFile(filePointer, curNode):
    """
    Given the root node of the tree and the file points it writes the tree in the
     file in DFS traversal order.
    :param filePointer: Pointer to the file on which the tree is to be written
    :param curNode: Root or current node of the tree
    """
    if curNode is None:
        filePointer.write("!,")
        return

    if not curNode.isLeaf:
        towrite = str(curNode.split[0]) + '-' + str(curNode.split[1]) + '-' + \
                  str(curNode.value) + ','
    else:
        towrite = '$' + '-' + '$' + '-' + str(curNode.value) + ','

    filePointer.write(towrite)
    writeFile(filePointer, curNode.lesser)
    writeFile(filePointer, curNode.greater)


def printTree(node, depth=0):
    """
    Given the root node of the tree it prints the tree.
    :param depth: depth of current node in the tree
    :param node: Root or current node of the tree
    """
    if node:
        print("  " * depth, node.classCount, node.split, "CLASS:", node.value)
        printTree(node.lesser, depth + 2)
        printTree(node.greater, depth + 2)


def chiSquareTest(node, sign):
    """
    Performs the chi-square test on the given node
    :param node: node of the tree to check for pruning
    :param sign: significance to be considered on chi-square testing.
    :return: Decision on whether the node to be pruned or not
    """
    leftClassHat = list()
    rightClassHat = list()
    leftTemp = sum(node.lesser.classCount) / sum(node.classCount)
    rightTemp = sum(node.greater.classCount) / sum(node.classCount)

    for c in node.classCount:
        leftClassHat.append(c * leftTemp)
        rightClassHat.append(c * rightTemp)

    delta = 0
    for c in range(len(node.classCount)):
        if leftClassHat[c] != 0:
            delta += (node.lesser.classCount[c] - leftClassHat[c]) ** 2 / leftClassHat[c]
        if rightClassHat[c] != 0:
            delta += (node.greater.classCount[c] - rightClassHat[c]) ** 2 / rightClassHat[c]
    return delta < 11.345 if sign == 0.01 else delta < 7.815


def pruning(node, sign, nodes, leaves, count=0):
    """
    Prunes the decision tree based on chi-square testing.
    :param node: Root or current node of the tree
    :param sign: significance to be considered on chi-square testing.
    :param nodes: No. of nodes in tree
    :param leaves: No. of leaves in tree
    :param count: Depth of the current node in the tree
    :return: Split list of samples
    """
    if not node.isLeaf:
        nodes, leaves = pruning(node.lesser, sign, nodes, leaves, count + 1)
        nodes, leaves = pruning(node.greater, sign, nodes, leaves, count + 1)
        if node.lesser.isLeaf and node.greater.isLeaf:
            if chiSquareTest(node, sign):
                node.lesser = node.greater = None
                node.isLeaf = True
                node.value, _ = max(enumerate(node.classCount), key=operator.itemgetter(1))
                node.value += 1
                leaves.remove(count + 1)
                leaves.remove(count + 1)
                leaves.append(count)
                return nodes - 2, leaves

    return nodes, leaves


def classCounter(samples):
    """
    Calculates the number of samples in each class
    :param samples: Samples to be classified
    :return: A list of number of samples in each class
    """
    counterList = [0, 0, 0, 0]
    for sample in samples:
        counterList[sample[-1] - 1] += 1
    return counterList


def splitOnIndex(examples, attribute, index):
    """
    Splits the samples based on the given attribute's index
    :param examples: Samples to be classified
    :param attribute: attribute based on which samples are to be split
    :param index: Index of the given attribute
    :return: Split list of samples and the value on which it is split
    """
    midValue = (examples[index][attribute] + examples[index + 1][attribute]) / 2
    greater, lesser = list(), list()
    for i in range(len(examples)):
        if examples[i][attribute] > midValue:
            greater.append(examples[i])
        else:
            lesser.append(examples[i])
    return lesser, greater, midValue


def splitOnValue(examples, attribute, value):
    """
    Splits the samples based on the given attribute's value
    :param examples: Samples to be classified
    :param attribute: attribute based on which samples are to be split
    :param value: Value of the given attribute
    :return: Split list of samples
    """
    greater, lesser = list(), list()
    for i in range(len(examples)):
        if examples[i][attribute] > value:
            greater.append(examples[i])
        else:
            lesser.append(examples[i])
    return lesser, greater


def entropy(samples):
    """
    Calculates the entropy of the given samples.
    :param samples: Samples to be classified
    :return: Entropy of samples.
    """
    counterList = classCounter(samples)
    probability = [x / sum(counterList) for x in counterList]
    return -sum([x * math.log2(x) for x in probability if x != 0])


def informationGain(examples, attributes, currentEntropy):
    """
    Learns the decision tree classification based on the given samples.
    :param examples: Samples to be classified
    :param attributes: attributes of samples
    :param currentEntropy: entropy of the current node
    :return: information gain calculated based on attributes.
    """
    infoGain = [[] for _ in range(attributes)]
    for attribute in range(attributes):
        examples.sort(key=operator.itemgetter(attribute))

        for index in range(len(examples) - 1):
            lesser, greater, midValue = splitOnIndex(examples, attribute, index)
            lesserEntropy = entropy(lesser)
            greaterEntropy = entropy(greater)

            remainder = ((len(lesser) / len(examples)) * lesserEntropy) + (
                (len(greater) / len(examples)) * greaterEntropy)
            infoGain[attribute] += [(midValue, currentEntropy - remainder)]
    return infoGain


def DTLearning(examples, attributes, parentExamples, curNode, nodes=1, leaves=list(), count=0):
    """
    Learns the decision tree classification based on the given samples.
    :param examples: Samples to be classified
    :param attributes: attributes of samples
    :param parentExamples: Parent samples of the current samples in the current recursion.
    :param curNode: Root or current node of the tree
    :param nodes: No. of nodes in the tree
    :param leaves: No. of leaves in the tree
    :param count: depth of the node at current recursion
    :return: Root node of the tree along with count of total number of nodes and leaves.
    """
    if len(examples) == 0:
        index, _ = max(enumerate(classCounter(parentExamples)), key=operator.itemgetter(1))
        return index + 1

    if all(examples[0][-1] == example[-1] for example in examples):
        curNode.value = str(examples[0][-1])
        curNode.split = "Base case"
        curNode.isLeaf = True
        leaves.append(count)
        return None, nodes, leaves

    currentEntropy = entropy(examples)
    infoGain = informationGain(examples, attributes, currentEntropy)

    maxList = [max(infoGain[index], key=operator.itemgetter(1)) for index in range(attributes)]
    if maxList[0][1] >= maxList[1][1]:
        splitAtt, splitVal = 0, maxList[0][0]
    else:
        splitAtt, splitVal = 1, maxList[1][0]
    lesser, greater = splitOnValue(examples, splitAtt, splitVal)

    curNode.split = (splitAtt + 1, splitVal)
    for childExamples in [lesser, greater]:
        childNode = Tree()
        if childExamples == lesser:
            childNode.classCount = classCounter(lesser)
            curNode.lesser = childNode
        else:
            childNode.classCount = classCounter(greater)
            curNode.greater = childNode

        classification, nodes, leaves = DTLearning(childExamples, attributes,
                                                   examples, childNode,
                                                   nodes + 1, leaves, count + 1)

        if isinstance(classification, int):
            nodes -= 1 if childExamples == lesser else 2
            leaves.append(count)
            curNode.lesser = None
            curNode.greater = None
            curNode.split = "Base case"
            curNode.value = classification
            break
    return curNode, nodes, leaves


def classify(sample, curNode):
    """
    Classifies the given sample.
    :param sample: Sample to be classified
    :param curNode: Root or current node of the tree
    :return: class of the given sample
    """
    while curNode.value is None:
        splitAtt, splitVal = curNode.split
        if sample[int(splitAtt) - 1] > float(splitVal):
            curNode = curNode.greater
        else:
            curNode = curNode.lesser
    return curNode.value


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
        decision_plot.scatter(x1, x2, label=CLASS_LIST[index - 1], marker=marker_box[index-1],
                              color=colors_box[index-1], s=100)

    decision_plot.legend(loc='upper right')
    decision_plot.set_xlabel("Six fold Rotational Symmetry")
    decision_plot.set_ylabel("Eccentricity")
    decision_plot.set_title("Decision boundary")
    return decision_plot


def printSummary(nodes, leaves):
    """
    Prints the No. of nodes, leaves and max, min and avg depth of the tree
    :param nodes: Total No. of nodes in the tree
    :param leaves: No. of leaves in the tree
    :return: None
    """
    print("\n------------------------------ SUMMARY -----------------------------")
    print("No. of Nodes     : ", nodes)
    print("No. of Leaf Nodes: ", len(leaves))
    print("Maximum Depth    : ", max(leaves))
    print("Minimum Depth    : ", min(leaves))
    print("Average Depth    : ", sum(leaves) / len(leaves))
    print("----------------------------------------------------------------------\n")


def main():
    """
    Main method
    return: None
    """
    _, _, samples = loadData(DATA_TRAIN)
    decisionTree = Tree()

    print("\n---------------------------- DECISION TREE ---------------------------\n")
    root, nodes, leaves = DTLearning(samples, 2, samples, decisionTree)
    printTree(root)
    printSummary(nodes, leaves)

    f = open('DTree.csv', 'w')
    writeFile(f, root)
    f.close()

    figure = plt.figure()
    decision_boundary(root, figure, DATA_TRAIN)
    plt.show()

    print("\n------------------------ PRUNED DECISION TREE ------------------------\n")
    pNodes, pLeaves = pruning(root, SIGNIFICANCE, nodes, leaves)
    printTree(root)
    printSummary(pNodes, pLeaves)

    f = open('PDTree.csv', 'w')
    writeFile(f, root)
    f.close()

    figure = plt.figure()
    decision_boundary(root, figure, DATA_TRAIN)
    plt.show()

    print("------------------------------- END ----------------------------------\n")


if __name__ == "__main__":
    main()
