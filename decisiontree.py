import numpy as np
import copy


class TreeNode(object):
    def __init__(self, classtype=None, leftnode=None, rightnode=None, attribute=None, threshold=None):
        self.__classtype = classtype
        self.__leftnode = leftnode
        self.__rightnode = rightnode
        self.__attribute = attribute
        self.__threshold = threshold

    def set_classtype(self, classtype):
        self.__classtype = classtype

    def add_leftnode(self, leftnode):
        self.__leftnode = leftnode

    def add_rightnode(self, rightnode):
        self.__rightnode = rightnode

    def set_attribute(self, attribute):
        self.__attribute = attribute

    def set_threshold(self, threshold):
        self.__threshold = threshold

    def get_classtype(self):
        return self.__classtype

    def get_leftnode(self):
        return self.__leftnode

    def get_rightnode(self):
        return self.__rightnode

    def get_attribute(self):
        return self.__attribute

    def get_threshold(self):
        return self.__threshold

    def classification(self,data):
        """
        :param self: must be root node
        :param data: a 1D list like [5.1,3.5,1.4,0.2,'Iris-setosa']
        :return: the prediction class of the data
        """
        if self.__attribute == None:
            classtype = self.__classtype
        else:
            if data[self.__attribute] < self.__threshold:
                classtype = self.__leftnode.classification(data)
            else:
                classtype = self.__rightnode.classification(data)
        return classtype


def ComputeEntropy(dataset):
    """
    Compute the entropy.
    :param dataset: a 2D list,each row of which is like [some attribute value, classtype]
    :return: the entropy computed
    """
    p1_num = 0
    p2_num = 0
    p3_num = 0
    for i in range(len(dataset)):
        if dataset[i][1] == 'Iris-setosa':
            p1_num += 1
        if dataset[i][1] == 'Iris-versicolor':
            p2_num += 1
        if dataset[i][1] == 'Iris-virginica':
            p3_num += 1
    p1 = p1_num / (p1_num + p2_num + p3_num)
    p2 = p2_num / (p1_num + p2_num + p3_num)
    p3 = p3_num / (p1_num + p2_num + p3_num)
    ent = 0
    if p1 != 0:
        ent = ent - p1*np.log2(p1)
    if p2 != 0:
        ent = ent - p2 * np.log2(p2)
    if p3 != 0:
        ent = ent - p3 * np.log2(p3)
    return ent


def ComputeGain(dataset, attribute):
    """
    Compute the gain of some attribute.
    :param dataset: a 2D list,each row of which is like [5.1,3.5,1.4,0.2,'Iris-setosa']
    :param attribute: an attribute
    :return: the best gain and its corresponding threshold
    """
    attri_value = [row[attribute] for row in dataset]
    attri_value = list(set(attri_value))
    attri_value.sort()
    thresholds = []
    for i in range(len(attri_value) - 1):
        thresholds.append((attri_value[i] + attri_value[i+1]) / 2)
    max_gain = 0
    max_threshold = 0
    for threshold in thresholds:
        dataset_sum = []
        dataset1 = []
        dataset2 = []
        for i in range(len(dataset)):
            dataset_sum.append([dataset[i][attribute], dataset[i][4]])
            if dataset[i][attribute] < threshold:
                dataset1.append([dataset[i][attribute], dataset[i][4]])
            else:
                dataset2.append([dataset[i][attribute], dataset[i][4]])
        gain = ComputeEntropy(dataset_sum) - (len(dataset1)/len(dataset))*ComputeEntropy(dataset1) - (len(dataset2)/len(dataset))*ComputeEntropy(dataset2)
        if gain > max_gain:
            max_gain = copy.deepcopy(gain)
            max_threshold = copy.deepcopy(threshold)
    return max_gain, max_threshold


def BestAttribute(dataset, attributeset):
    """
    find the best attribute.
    :param dataset: a 2D list,each row of which is like [5.1,3.5,1.4,0.2,'Iris-setosa']
    :param attributeset: a list of attribute
    :return:best attribute and its threshold
    """
    largest_gain = 0
    best_attribute = None
    best_threshold = 0
    for attribute in attributeset:
        gain_attri, threshold = ComputeGain(dataset, attribute)
        if gain_attri > largest_gain:
            largest_gain = copy.deepcopy(gain_attri)
            best_attribute = copy.deepcopy(attribute)
            best_threshold = copy.deepcopy(threshold)
    return best_attribute, best_threshold


def TreeGenerate(dataset, attributeset):
    """
    Generate a tree by this function.
    :param dataset: a 2D list,each row of which is like [5.1,3.5,1.4,0.2,'Iris-setosa']
    :param attributeset: a list of attribute
    :return: a node
    """
    # generate a node
    node = TreeNode()

    # find most class
    iris_setosa = 0
    iris_versicolor = 0
    iris_virginica = 0
    for i in range(len(dataset)):
        if 'Iris-setosa' == dataset[i][4]:
            iris_setosa += 1
        if 'Iris-versicolor' == dataset[i][4]:
            iris_versicolor += 1
        if 'Iris-virginica' == dataset[i][4]:
            iris_virginica += 1
    most_num = iris_setosa
    most_class = 'Iris-setosa'
    if iris_versicolor > most_num:
        most_num = iris_versicolor
        most_class = 'Iris-versicolor'
    if iris_virginica > most_num:
        most_num = iris_virginica
        most_class = 'Iris-virginica'

    # if all the samples belong to the same same class,set classtype and return
    # you can reduce the parameter alpha to weaken the effect of over-fitting
    alpha = 1.0
    if alpha*(len(dataset)) <= most_num:
        node.set_classtype(most_class)
        return node

    # if attribute set is empty,set classtype and return
    if len(attributeset) == 0:
        node.set_classtype(most_class)
        return node

    # find the best attribute
    best_attribute, threshold = BestAttribute(dataset, attributeset)
    node.set_attribute(best_attribute)
    node.set_threshold(threshold)

    # bulid branchs
    dataset1 = []
    dataset2 = []
    for i in range(len(dataset)):
        if dataset[i][best_attribute] < threshold:
            dataset1.append(dataset[i])
        else:
            dataset2.append(dataset[i])
    # If attributes are discrete,replace replace attributeset with sub_attributeset
    # sub_attributeset = copy.deepcopy(attributeset)
    # sub_attributeset.remove(best_attribute)
    node_son1 = TreeGenerate(dataset1, attributeset)
    node_son2 = TreeGenerate(dataset2, attributeset)
    node.add_leftnode(node_son1)
    node.add_rightnode(node_son2)
    return node


def ReadData(filename):
    """
    Read dataset from "filename".
    :param filename: a string of a file,which contains dataset
    :return: train dataset and test dataset
    """
    f = open(filename, 'r')
    dataset = f.readlines()
    f.close()
    for i in range(len(dataset)):
        dataset[i] = dataset[i].rstrip().split(',')
        for j in range(4):
            dataset[i][j] = float(dataset[i][j])
    train_dataset = []
    test_dataset = []
    for data in dataset:
        for i in range(50):
            if i < 10:
                train_dataset.append(data)
            else:
                test_dataset.append(data)

    return train_dataset, test_dataset


def ComputeAccuracy(test_dataset, root_node):
    """
    Compute accuracy of classification of the Decision Tree .
    :param test_dataset: test dataset or validation dataset
    :param root_node: the root node of the Decision Tree
    :return: accuracy from [0,1]
    """
    TP = 0
    for data in test_dataset:
        classtype = root_node.classification(data)
        if classtype == data[4]:
            TP += 1
    acc = TP / len(test_dataset)
    return acc


def main():
    # read data
    train_dataset, test_dataset = ReadData('data.txt')
    attributeset = [0, 1, 2, 3]     # correlate to ['sepal length','sepal width','petal length','petal width']

    # generate tree
    root_node = TreeGenerate(train_dataset, attributeset)

    # compute accuracy
    acc = ComputeAccuracy(test_dataset, root_node)
    print('Accuracy:', acc)


if __name__ == '__main__':
    main()