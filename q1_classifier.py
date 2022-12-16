import math
import numpy as np
from scipy import stats
import pickle 
import sys
import getopt

def read_data(data):
    """ 
    input----
    data: path of data file

    return---
    data_feature: numpy array with data

    """
    data_feature = []
    file = open(data)
    for line in file:
        feature_vector = [int(x) for x in line.split()]
        data_feature.append(feature_vector)
    # print("Data File:", data)
    # print("Number of Rows: ", len(data_feature),": Number of Colums",len(data_feature[0]))
    return np.array(data_feature)

def read_label(data):
    """ 
    input----
    data: path of data file

    return---
    data_label: numpy array with labels

    """
    data_label = []
    file = open(data)
    for line in file:
        label_vector = [int(x) for x in line.split()]
        data_label.extend(label_vector)
    # print("Data File:", data)
    # print("Number of Rows: ", len(data_label),": Number of Colums: 1")
    return np.array(data_label)

#node class
class Node():
    def __init__(self,name):
        """ 
        input----
        name: feature name of the node

        """
        self.name = name   #assign the name to node
        self.dict = {}     #initiate a dictionary that will contain child of node
        self.leaf = False   #set leaf value to show if the node is leaf node or not
        self.label = -1     # set the label of the  node
        
    def update_dict(self, val, child):              # add the child to the node's dictionary
        """ 
        input----
        val: value of the feature
        child: feature node being assigned to val
        """
        if val not in self.dict.keys():
            self.dict[val] = child 
    def getNext(self, val):
        """ 
        input----
        val: value of the feature

        return : node assigned to value val
        """
        return self.dict[val]
    
class Tree():
    def __init__(self,p):           # initialize number of nodes, number of leafs and chi square threshold 
        """ 
        input----
        p: threshold value for the chi-squared stopping criteria

        """
        self.number_nodes = 0
        self.number_leaf = 0
        self.threshold = p
        self.root = None

    def data_part(self, idx, val, data, label):            #partition data on th bases of particular value in the feature
        """ 
        input----
        idx(int): feature number of training data
        val(int): value of the feature
        data(numpy array): training data
        label(numpy array): lables of training data

        return : data corresponding to particular value of the feature, label corresponding to particular value of the feature
        """
        sub_data = []
        sub_label = []

        for i in range(0,data.shape[0]):
            if data[i][idx]==val:
                sub_data.append(data[i])
                sub_label.append(label[i])
        return np.array(sub_data), np.array(sub_label)
    def find_unique(self, data, idx):                       # find the unique values of the feature
        """
        input----
        idx(int): feature number of training data
        data(numpy array): training data

        return---
        unique values of feature in data
        """
        unique = np.unique(data[:,idx])
        return unique
    def check_value(self, label):                           # check if all the labels are same and if not what is the max label in data
        """
        input----
        label(numpy array): labels data

        return---
        label with maximum count, (bool){True: All Same label, False: Different lables}
        """
        same = False
        max_lab = None
        pos = np.count_nonzero(label == 1)
        neg = np.count_nonzero(label == 0)
        
        if pos==0 or neg==0:
            same = True
            if pos>neg:
                max_lab = 1
            else:
                max_lab = 0
        else:
            if (neg/pos) > (32193.0/7807.0):
                max_lab = 0
            else:
                max_lab = 1
        return max_lab, same
    def get_entropy(self, label):                   #get the entropy of the labels
        """
        input--
        labels: labels in the training data

        return--
        entropy of labels
        """
        pos = np.count_nonzero(label == 1)
        neg = np.count_nonzero(label == 0)
        total = label.shape[0]

        en = 0
        if pos!=0:
            en = en - (pos/total)*math.log(pos/total, 2)
        if neg!=0:
            en = en - (neg/total)*math.log(neg/total, 2)
        return en
    
    def max_entropy_gain(self, train, label, visited): #get the feature with the maximum gain
        """
        data(numpy array): training data
        label(numpy array): lables of training data
        visited: dictionary of feature visited

        return--
        index of the feature with maximum entropy
        """
        feats = train.shape[1]
    
        entropy = self.get_entropy(label)
        max_gain = -999
        max_idx = -99
        total = train.shape[0]
        
        for i in range(0, feats):                                           #compare the gain for each feature and choose the one with maximum value
            
            if visited[i]:
                continue;
            values= self.find_unique(train,i)
            ent = 0
            for val in values:
                count = np.count_nonzero(train[:,i]==val)
                prob = count/total
                _, sub_label = self.data_part(i, val, train, label)
                ent = ent + self.get_entropy(sub_label)*prob
            
            gain = entropy-ent

            if gain> max_gain:
                max_gain = gain
                max_idx = i
        # print("Max gain", max_gain)
        return max_idx

    def construct_tree(self, train, label, visited):                            # construct the tree
        # print("********************************************")
        """
        data(numpy array): training data
        label(numpy array): lables of training data
        visited: dictionary of feature visited
        
        return---
        root node for the tree
        """
        root = Node(-1)
        max_label, same = self.check_value(label)
        self.number_nodes+=1
        if max_label == -1:
            print("WRONG")
        if False not in visited:
            root.leaf = True
            root.label = max_label
            self.number_leaf+=1
            return root
        elif same:
            root.leaf = True
            root.label = max_label
            self.number_leaf+=1
            

            return root
        else:
            max_gain_idx = self.max_entropy_gain(train, label, visited) #find the feature with maximum information gain
            
            # print("Best Feature", max_gain_idx)

            root.name = max_gain_idx

            datasets={}
            values = self.find_unique(train, max_gain_idx)
            for val in values:                                                              # partition data w.r.t each value in the features and store in a dict
                sub_train, sub_label = self.data_part(max_gain_idx, val, train, label)
                new_visited = visited[:]
                new_visited[max_gain_idx] = True
                datasets[val] = [sub_train, sub_label, new_visited]
            pos = np.count_nonzero(label == 1)
            neg = np.count_nonzero(label == 0)
            # print("Pos: ", pos)
            # print("Neg: ", neg)
            S = 0.0                                                                         # calculate chi square
            for v in datasets.values():
                pi = pos * (len(v[1])/float(train.shape[0]))
                ni = neg * (len(v[1])/float(train.shape[0]))
                r_pi = np.count_nonzero(v[1] == 1)
                r_ni = np.count_nonzero(v[1] == 0)

                temp = 0
                if r_pi!=0:
                    temp+=pow(r_pi - pi,2)/r_pi
                if r_ni!=0:
                    temp+=pow(r_ni - ni, 2)/r_ni

                S+=temp
            
            p_v = 1-stats.chi2.cdf(S, len(datasets))                                # chi square value
            # print("chi-sq", p_v)
            if p_v<self.threshold:
                for k in datasets.keys():
                    node = self.construct_tree(datasets[k][0],datasets[k][1],datasets[k][2])
                    root.update_dict(k, node)
            else:
                root.leaf = True
                root.label = max_label
                self.number_leaf+=1
                
                return root
        
        return root

    def getPredictions(self, data, labels):
        """
        data(numpy array): testing data
        label(numpy array): lables of testing data
        return--
        accuracy, precision, recall, predictions of tree
        
        """
        print("\n--------------------------------Getting Predictions---------------------")

        node = self.root
        predictions = []
        for instance in data:
            pred = self.predictOne(instance, node)
            predictions.append(pred)
        true_p = 0
        true_n = 0
        false_p = 0
        false_n = 0
        count = 0
        for (pred, lab) in zip(predictions, labels):
            count +=1
            if pred==1 and lab==1:
                true_p+=1
            elif pred==1 and lab==0:
                false_p+=1
            elif pred==0 and lab==1:
                false_n+=1
            elif pred==0 and lab==0:
                true_n+=1
            else:
                print("something is wrong")
        accuracy = (true_n + true_p) / (count)
        precision = true_p/(true_p + false_p)
        recall = true_p/(true_p + false_n)

        print("Accuracy is :", accuracy)
        print("Precision is :", precision)
        print("Recall : ", recall)

        return accuracy, precision, recall, predictions



    def predictOne(self, sample, node):
        """
        sample: instance from testing data
        node: node to check for the value

        return--
        label value of the node
        """
        if node.leaf:
            return node.label
        
        # print("Name :", node.name, ": ", type(node.name)," :", node.leaf," : ", node.label)

        val = sample[node.name]
        
        if val not in node.dict.keys():
            return 0
        
        nextNode = node.getNext(val)
        
        predict = self.predictOne(sample, nextNode)

        return predict

def saveModel(model):
    """
    model: constructTree object
    """
    filehandler = open('modelID3', 'wb') 
    pickle.dump(model, filehandler)
    print("Model dumped as => [modelID3]")

def getModel(model_name):
    """
    model_name: path of model pkl file

    return--
    constructTree object
    """
    print("\n--------------------------------Using saved model: ", model_path,"--------------------------")

    filehandler = open(model_name, 'rb') 
    model = pickle.load(filehandler)

    return model

def savePrediction(predictions, path):
    """
    predictions: prediction of model on test data
    path: path of .csv to save
    """
    print("\n--------------------------Saving output in: ",path,"-----------------------------------")
    with open(path, 'w', encoding='UTF8') as f:
        for p in predictions:
            f.write(str(p)+"\n")

def data_retrieve(train_data_path,train_label_path,test_data_path,test_label_path):
    """
    train_data_path: path of the training data file
    train_label_path: path of the training label file
    test_data_path: path of the testing data file
    test_label_path: path of the testing label file

    return--
    train_data: numpy array of training data
    train_label: numpy array of testing label
    test_data: numpy array of training data
    test_label: numpy array of testing label
    """
    print("\n-------------------------Reading Training data-------------------------")
    train_data = read_data(train_data_path)
    train_label = read_label(train_label_path)

    train_data = np.array(train_data)
    train_label = np.array(train_label)
    print("\n-------------------------Reading Testing data-------------------------")
    test_data = read_data(test_data_path)
    test_label = read_label(test_label_path)

    test_data = np.array(test_data)
    test_label = np.array(test_label)

    return train_data,train_label, test_data, test_label
            

train_data_path = ""
test_data_path = ""
train_label_path =""
test_label_path = ""
output_path =""
model_path =""
p= None

#---getting arugments--------
options = ["p=", "f1=", "f2=", "o=", "t="]
op = "p:f1:f2:o:t:"
try:
    opts, args = getopt.getopt(sys.argv[1:],op, options )
except getopt.GetoptError:
    print ('Wrong arguments: please follow this pattern')
    print("python q1_classifier.py -p <pvalue> -f1 <train_dataset> -f2 <test_dataset> -o <output_file> -t <decision_tree>")

    sys.exit(2)

for opt, arg in opts:
    if opt in ['-p','--p']:
        p = float(arg)
    elif opt in ['-1','--f1']:
        train = arg
    elif opt in ['-2','--f2']:
        test = arg
    elif opt in ['-o','--o']:
        output_path = arg
    elif opt in ['-t','--t']:
        model_path = arg

#--parsing arguments----
train_data_path = train
train_label_path = train[:5]+'labs'+train[-4:]

test_data_path = test
test_label_path = test[:4]+'labs'+test[-4:]

print(train_data_path,train_label_path,test_data_path,test_label_path,p, output_path)
train_data,train_label, test_data, test_label = data_retrieve(train_data_path,train_label_path,test_data_path,test_label_path)

if model_path:

    model = getModel(model_path) # get saved model
    root = model.root

    print("Name", root.name)
    print("Number of nodes ", model.number_nodes)
    print("Number of leave nodes", model.number_leaf)

    acc, pre, rec, predictions = model.getPredictions(test_data, test_label) # get predictions from saved model
    
    savePrediction(predictions,output_path) #save prediction to output_path
    
    exit()


print("\n---------------------------Start the training---------------------------")
visited = [False for i in range(0, train_data.shape[0])]
tree = Tree(p)

root = tree.construct_tree(train_data, train_label, visited) # get root of the tree
tree.root = root

print("\n---------------------------Training Done---------------------------")
print("Name", root.name)
print("Number of nodes ", tree.number_nodes)
print("Number of leave nodes", tree.number_leaf)

# print("\n---------------------------Saving Model---------------------------")
saveModel(tree) # save the model as tree object

# print("\n---------------------------Getting Predictions---------------------------")
acc, pre, rec, predictions = tree.getPredictions(test_data, test_label)# get prediction on the test data set

# print("\n---------------------------Saving .csv output---------------------------")
savePrediction(predictions,output_path)# save prediction in output_path file




