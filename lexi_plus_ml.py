#-------------Project Code for CSC 522------------------#
#----------Last Modified: April 20,2018 ----------------#

#----importing packages-------_#
import random
import math
import os
import numpy as np
import pandas as pd
import time
import io
import operator

start_time = time.time()

os.getcwd()
#------------------Change path-----------------#
os.chdir('/Users/jagadeesh/PycharmProjects/ALDA')
os.getcwd()

import nltk
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.tokenize import RegexpTokenizer, word_tokenize
from pandas import DataFrame
from sklearn.model_selection import train_test_split


# ----- FUNCTIONS ------ #

#----------Data Preprocessing functions--------- #

#NLTK
def nltk_process(sample_text):
    # reading an input text file
    # input_file = open('/Users/jagadeesh/PycharmProjects/ALDA/testdoc.txt','r')
    # sample_text = input_file.read()
    # input_file.close()


    # word tokenizer, but it includes the punctuation
    # tokenized = word_tokenize(sample_text)

    # tokenizing words using RegexpTokenizer of nltk, which by default removes all punctuation.
    tokenizer = RegexpTokenizer(r'\w+')
    tokenized = tokenizer.tokenize(sample_text)
    length_input = len(tokenized)  # used later in for loop which splits pos_tags

    # print(tokenized)

    # converted each word to its root word using WordNetLemmatizer of nltk.
    lemma = WordNetLemmatizer()
    lemma_words = map(lemma.lemmatize, tokenized)

    pos_tagged = []

    # function for assigning pos_tags
    def pos_tagging():
        try:
            for i in lemma_words:
                words = nltk.word_tokenize(i)
                for j in words:
                    tagged = nltk.pos_tag(words)
                    pos_tagged.append(tagged)
        except Exception as e:
            print(str(e))

    # function call for performing pos_tags
    pos_tagging()

    # print(pos_tagged)

    # splitting the pos_tags into two seperate lists for further processing
    list_df = DataFrame.from_records(pos_tagged)
    words = []
    pos_final = []
    for i in range(0, length_input):
        for j in range(0, 2):
            if j == 0:
                words.append(list_df[0][i][j])
            if j == 1:
                pos_final.append(list_df[0][i][j])

    # converted as table if required for further processing
    pos_table = np.column_stack((words, pos_final))
    # pos_table=pd.DataFrame(words,pos_final)
    return pos_table


def f(row):
    if row['Sentiment'] == 'positive':
        val = 1
    elif row['Sentiment'] == 'negative':
        val = -1
    else:
        val = 0.5
    return val


def ads(st):
    merge = pd.merge(st, pd.DataFrame(dic), how='left', left_on='Word', right_on='Token', sort=False)
    merge = merge.fillna(0)
    merge2 = pd.merge(merge, pd.DataFrame(intense), how='left', left_on='Word', right_on='Token', sort=False)
    merge2 = merge2.fillna(0)

    for i in range(len(merge2.index) - 1):
        merge2['Polarity_x'][i + 1] = merge2['Polarity_x'][i + 1] * (1 + merge2['Polarity_y'][i])
        merge2['Polarity'] = merge2['Polarity_x'].sum()

        merge3 = pd.merge(merge2, pd.DataFrame(bucket), how='left', left_on='POS', right_on='POS', sort=False)
        merge4 = pd.DataFrame(merge3.groupby(by=['Bucket']).agg({'Polarity_x': 'sum'}).transpose())
        merge4['index'] = range(1, len(merge4) + 1)
        merge5 = pd.DataFrame({'Polarity': []})
        merge5.set_value(1, 'Polarity', merge2['Polarity'][1])
        merge5['index'] = range(1, len(merge5) + 1)
        merge5 = pd.merge(merge4, pd.DataFrame(merge5), how='left', left_on='index', right_on='index', sort=False)

        return merge5


#-------num of pos,neg,abs,words----------#
negative_words = io.open('negabuse.txt', encoding='ISO-8859-1').read().splitlines()
positive_words = io.open('positive-words.txt',encoding='ISO-8859-1').read().splitlines()


def pos_words(sentence_input):
    #Calculating postive words
    numPosWords = 0
    for word in sentence_input:
        if word in positive_words:
            numPosWords += 1
    # print(numPosWords)
    return(numPosWords)

def neg_words(sentence_input):
    #Calculating postive words
    numNegWords = 0
    for word in sentence_input:
        if word in negative_words:
            numNegWords += 1
    # print(numNegWords)
    return(numNegWords)


##finding correlation of each feature with the class label
# print('Adjective')
# print(np.corrcoef(ads_df_final_1['Adjective'], ads_df_final_1['label']))
# print('Adverb')
# print(np.corrcoef(ads_df_final_1['Adverb'], ads_df_final_1['label']))
# print('Noun')
# print(np.corrcoef(ads_df_final_1['Noun'], ads_df_final_1['label']))
# print('Verb')
# print(np.corrcoef(ads_df_final_1['Verb'], ads_df_final_1['label']))
# print('Polarity')
# print(np.corrcoef(ads_df_final_1['Polarity'], ads_df_final_1['label']))
# print('Num_Positive')
# print(np.corrcoef(ads_df_final_1['Num_Positive'], ads_df_final_1['label']))
# print('Num_Negative')
# print(np.corrcoef(ads_df_final_1['Num_Negative'], ads_df_final_1['label']))

###################################################################
#-----------------Naive Bayes Classification code-----------------#
###################################################################

# Define function to split dataset with ratio
def Datasplit(data, Ratiosplit):
    X = data.ix[:, 'Polarity':'label'].values
    train, test = train_test_split(X, test_size=(1 - Ratiosplit))
    train = train.tolist()
    test = test.tolist()
    return [train, test]

# Calculating the mean and standard deviation for every class
def summ(dataset):
    summaries = [(np.mean(numbers), np.var(numbers)) for numbers in zip(*dataset)]
    del summaries[-1]
    return summaries

def Classwisesumm(dataset):
    dataclass = {}
    for i in range(len(dataset)):
        loop = dataset[i]
        if (loop[-1] not in dataclass):
            dataclass[loop[-1]] = []
        dataclass[loop[-1]].append(loop)

    measures = {}
    for clas, vals in dataclass.items():
        measures[clas] = summ(vals)
    return measures

# Defining the Gaussian Probability Density Function
def prob(x, mean, sd):
    var = float(sd) ** 2
    pi = 3.1415926
    denom = (2 * pi * var) ** .5
    num = math.exp(-(float(x) - float(mean)) ** 2 / (2 * var))
    return num / denom

# Classify the test data based on likelihood ratio
def Classify(s, testin):
    def makeprediction(s, testin):
        ClassProb = {}
        for cv, cs in s.items():
            ClassProb[cv] = 1
            for i in range(len(cs)):
                mean, stdev = cs[i]
                x = testin[i]
                ClassProb[cv] *= prob(x, mean, stdev)  # Calculating class probability
        labelv = []
        llprob = []
        likelihood = []
        labelfinal = None
        bp = -10
        for i in ClassProb.keys():
            labelv.append(i)
        for i in ClassProb.values():
            llprob.append(i)
        for i in range(len(llprob)):
            likelihood.append(llprob[i] / sum(llprob))  # Calculating likelihood
        for i in range(len(labelv)):
            if labelfinal is None or likelihood[i] > bp:
                bp = likelihood[i]  # Assigning likelihood
                labelfinal = labelv[i]
        return labelfinal

    classifications = []
    for i in range(len(testin)):
        result = makeprediction(s, testin[i])
        classifications.append(result)
    return classifications

# Running the classifier and calculating the accuracy of predictions
def NVB(dataset):
    Ratiosplit = 0.7  #Split ratio for test and train dataset
    training, test = Datasplit(dataset, Ratiosplit)  # Calling split function
    print('{0} rows are split into {1} rows for training and {2} rows for testing'.format(len(dataset), len(training),
                                                                                          len(test)))
    model = Classwisesumm(training)
    classification = Classify(model, test)
    Diagonal = 0
    for x in range(len(test)):  # Calculating accuracy of classification
        if test[x][-1] == classification[x]:
            Diagonal += 1
    accuracy = (Diagonal / float(len(test))) * 100.0
    print('Accuracy obtained through Naive Bayes = {0}%'.format(accuracy))

###################################################################
#-------------------Decision Tree Algorithm ----------------------#
###################################################################
def dtree(dataset):
    data = dataset
    class classifier:
        def __init__(self, true=None, false=None, column=-1, split=None, result=dict()):
            self.true = true  # it is used to identify the true decision nodes
            self.false = false  # it is used to identify the false decision nodes
            self.column = column  # the column index of criteria that is being tested
            self.split = split  # it assigns the split point based on entropy
            self.result = result  # it stores the result in the form of an dictionary

    def key_labels(data):  # This function is used to return the unique class labels from the data
        keys = []
        for i in range(0, len(data)):  # For loop is used to store unique class/keys from dataset
            if data.iloc[i, len(data.columns) - 1] in keys:
                continue
            else:
                keys.append(data.iloc[i, len(data.columns) - 1])
        return keys

    def entropy_cal(data,
                    keys):  # This function is used to return the entropy of parent node / data and the count of class
        count = [0] * len(keys)
        ent = [0] * len(keys)
        lgth = len(data)
        for i in range(0, len(data)):  # A for loop is used to calculate the count of each class in a dataset
            for j in range(0, len(keys)):
                if data.iloc[i, len(data.columns) - 1] == keys[j]:
                    count[j] += 1
            for j in range(0, len(keys)):  # A for loop is used to calculate the entropy of data
                if count[j] != 0:
                    ent[j] = (count[j] / float(lgth)) * np.log2(count[j] / float(lgth))
        ent = [-x for x in ent]
        entropy = sum(ent)
        return entropy, count

    def entropy_data(data,
                     keys):  # This function is used to return the entropy of parent node / data and the count of class
        len_keys = len(keys)
        ent_dict = {}
        split_dict = {}
        # print(data)
        if len(data.columns) > 1:  # If loop is executed if the number of columns in dataset is greater than 1

            for i in range(0, len(
                    data.columns) - 1):  # A for loop is executed to carry out entropy calculations for each attribute
                if len(data[data.columns[
                    i]].unique()) > 1:  # If the number of unique values in an attribute is greater than one, then the if loop is executed
                    entropy_min = 2  # Initially the minimum value of entropy is set as the maximum possible entropy value, i.e. log(n) where n is the number of unique classes
                    test = data[data.columns[[i, len(
                        data.columns) - 1]]]  # A dataset is created having the selected attribute and its label values to find the split point
                    max_value = test.iloc[:, 0].max()
                    min_value = test.iloc[:, 0].min()
                    y = min_value  # the initial value of split point is set as the minimum value of the selected attribute
                    while (
                        y < max_value):  # a while loop is executed untill the split point value reaches the maximum value of that attribute
                        left = []  # an empty list is created to store the labels of the dataset with value <= split point
                        right = []  # an empty list is created to store the labels of the dataset with value > the split point
                        for x in range(0, len(test)):
                            if test.iloc[x, 0] <= y:
                                left.append(test.iloc[x, 1])
                            else:
                                right.append(test.iloc[x, 1])

                        split_pair = [left, right]
                        ent_pair = [0, 0]
                        for sp_val in range(0,
                                            2):  # A for loop is created to calculate the entropy of the left split and right split
                            count = [0] * len_keys
                            prop = [0] * len_keys
                            for sp_ct in range(0, len(split_pair[sp_val])):
                                lgth = len(split_pair[sp_val])
                                for j in range(0, len_keys):
                                    if split_pair[sp_val][sp_ct] == keys[j]:
                                        count[j] += 1
                            for j in range(0, len_keys):
                                if count[j] != 0:
                                    prop[j] = (count[j] / float(lgth)) * np.log2(count[j] / float(lgth))
                            ent_pair[sp_val] = sum(prop)

                        ent_pair = [-x for x in ent_pair]
                        entropy = (ent_pair[0] * len(left) / float(len(test))) + (ent_pair[1] * len(right) / float(
                            len(test)))  ##The resultant entropy of the split is stored in "entropy"
                        if entropy < entropy_min:
                            entropy_min = entropy
                            splitvalue = y
                        y = y + 1
                    # print(entropy_min, splitvalue)
                    ent_dict[test.columns[0]] = entropy_min
                    split_dict[test.columns[
                        0]] = splitvalue  # attribute is stored as a key and the split point is stored as its value in the dictionary split_dict
        return ent_dict, split_dict

    def parent_node(dictionary,
                    entropy):  # the function created which returns the attribute and its gain to determine the node with the maximum gain and assign it as a parent node
        gain = {}
        for k, v in dictionary.items():
            gain[k] = entropy - v
        attribute = max(gain, key=gain.get)
        return attribute, gain[attribute]

    def split_data(data, attribute,
                   split_dict):  # the function created splits the dataset while checking the values of the best split point and returns the points and splits the data into two
        left = data[data[attribute] <= split_dict[attribute]]
        right = data[data[attribute] > split_dict[attribute]]
        return left, right

    def unique_values(data1, data2):  # This function is used to find the unique count of values from each rows
        left_count = 0
        right_count = 0
        for i in (0, len(
                data1.columns) - 1):  # for loop is used to store count of the unique values from each attribute of data1
            left_idx = len(data1[data1.columns[i]].unique())
            left_count = left_count + left_idx
        for i in (
        0, len(data2.columns) - 1):  # for loop is used to store the count of unique values from each attribute of data2
            right_idx = len(data2[data2.columns[i]].unique())
            right_count = right_count + right_idx
        return left_count, right_count

    def complete_tree(data,
                      keys):  # This function is used to store the rules/decisions in an object through a class called "classifier"
        entropy, count = entropy_cal(data, keys)  # The entropy of parent node is stored in the variable "entropy"
        ent_dict, split_dict = entropy_data(data, keys)
        attribute, gain = parent_node(ent_dict,
                                      entropy)  # the attribute and gain is stored using the function parent_node
        if gain > 0:  # An if loop is executed when the gain will be positive
            left, right = split_data(data, attribute,
                                     split_dict)  # split_data function is used to split the dataset into two based on the attribute and split value
            left_count, right_count = unique_values(left,
                                                    right)  # the count of unique values from each attribute is calculated
            if left_count > len(
                    left.columns) - 1:  # if each attribute has only one value, then the if loop is not executed
                true = complete_tree(left, keys)
            if right_count > len(right.columns) - 1:
                false = complete_tree(right, keys)
            if left_count > len(left.columns) - 1 and right_count > len(right.columns) - 1:
                return classifier(true=true, false=false, column=attribute,
                                  split=split_dict[attribute])  # the attribute and split value is stored in the class
            else:
                return classifier(result=dict(
                    zip(keys, count)))  # the count of each class at the leaf node is stored as results in a class
        else:
            return classifier(result=dict(zip(keys, count)))

    keys = key_labels(data)  # key stores the value of the uique key labels in the lat column

    a = complete_tree(data, keys)  # a will return the stored value when we call the intialised object

    ###################################################################
     #----------------------BUILDING THE MODEL------------------------#
    ###################################################################

    from sklearn.model_selection import \
        train_test_split  # using sci-kit learn only to train and test the data uniformly
    train, test = train_test_split(data, test_size=0.25)
    test22 = test.iloc[:, :-1]
    test_data = test22.to_dict('records')

    def traverse(a,
                 each_row):  # the given function is used to assign the labels of the decision tree to our test data by comapring the dictinoary structure of instance a
        if (a.column == -1):
            return max(a.result.iteritems(), key=operator.itemgetter(1))[0]
        if (each_row[a.column] < a.split):
            return traverse(a.true, each_row)
        else:
            return traverse(a.false, each_row)

    for each_row in test_data:  # looping to assign the labels obtained from the instance a
        label_value = traverse(a, each_row)
        each_row['Label'] = label_value

    test_data2 = pd.DataFrame(test_data)  # manipulating the dataset

    labels_test_data = test_data2[
        test_data2.columns[[2]]]  # extracting labels of the test data to later cross validate

    labels_of_original = test[
        test.columns[[-1]]]  # extracting the labels of the test data to extract later for cross validation

    count = 0  # initialising count = 0 to count the number of correct predictions
    for i in range(0, len(labels_test_data)):
        if labels_test_data.iloc[i, 0] == labels_of_original.iloc[i, 0]:
            count = count + 1
    correct_predictions = count  # correct predicitons will give the number of correct predictions

    accuracy = correct_predictions / float(len(labels_test_data))
    print('Accuracy obtained through Decision Tree = {0}%'.format(accuracy))

###################################################################
# #----------------Comaprisons among Classifiers------------------#
###################################################################
# from sklearn.linear_model import LogisticRegression
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.svm import SVC, LinearSVC, NuSVC
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
# from sklearn.naive_bayes import GaussianNB
# from sklearn.metrics import accuracy_score
# import pandas as pd
# from sklearn.model_selection import train_test_split
# import matplotlib.pyplot as plt
#
# data = pd.read_csv('features_update_2.csv',encoding='utf=8')
#
# best_classifier = [
#     LogisticRegression(C=0.000000001,solver='liblinear',max_iter=200),
#     KNeighborsClassifier(3),
#     SVC(kernel="rbf", C=0.025, probability=True),
#     DecisionTreeClassifier(),
#     RandomForestClassifier(n_estimators=200),
#     AdaBoostClassifier(),
#     GaussianNB()]
#
# train, test = train_test_split(data, test_size=0.3)
#
# # x_train = train[['Adjective', 'Adverb', 'Noun', 'Verb', 'Polarity','Num_Positive','Num_Negative']]
# x_train = train[['Polarity', 'Num_Positive', 'Num_Negative']]
#
# y_train = train['label']
# y_train = y_train.astype('int')
#
# # x_test = test[['Adjective', 'Adverb', 'Noun', 'Verb', 'Polarity','Num_Positive','Num_Negative']]
# x_test = test[['Polarity','Num_Positive','Num_Negative']]
#
# y_test = test['label']
# y_test = y_test.astype('int')
#
# Accuracy=[]
# Model=[]
# for classifier in best_classifier:
#     try:
#         fit = classifier.fit(x_train,y_train)
#         pred = fit.predict(x_test)
#     except Exception:
#         print(classifier)
#     accuracy = accuracy_score(pred,y_test)
#     Accuracy.append(accuracy)
#     Model.append(classifier.__class__.__name__)
#     print('Accuracy of '+classifier.__class__.__name__+'is '+str(accuracy))
#
# # ----- Plotting the accuracies on a bar chart-------#
# Index = [1,2,3,4,5,6,7]
# plt.bar(Index,Accuracy)
# plt.xticks(Index, Model, rotation = 10)
# plt.ylabel('Accuracy')
# plt.xlabel('Model')
# plt.title('Accuracies of Models')
# plt.show()
#
#
# #--------Building the confusion matrix----------#
# # Classifier = [
# #     DecisionTreeClassifier(),
# #     ]
# #
# # Accuracy=[]
# # Model=[]
# #
# # for classifier in Classifier:
# #     try:
# #         fit = classifier.fit(x_train,y_train)
# #         pred = fit.predict(x_test)
# #     except Exception:
# #         # fit = classifier.fit(dense_features,train['sentiment'])
# #         # pred = fit.predict(dense_test)
# #         print(classifier)
# #     accuracy = accuracy_score(pred,y_test)
# #     Accuracy.append(accuracy)
# #     Model.append(classifier.__class__.__name__)
# #     # print('Accuracy of '+classifier.__class__.__name__+'is '+str(accuracy))
# #     y_actu = pd.Series(y_test, name='Actual')
# #     y_pred = pd.Series(pred, name='Predicted')
# #     df_confusion = pd.crosstab(y_actu, y_pred)
# #     print(df_confusion)

###################################################################
#--------Data input and calling preprocessing functions-----------#
###################################################################
data = pd.read_csv('tweetylabel.csv', encoding='utf=8')
dic = pd.read_csv('dic.csv', encoding='utf=8')
intense = pd.read_csv('intense.csv', encoding='utf=8')
bucket = pd.read_csv('bucket.csv', encoding='utf=8')
data.columns = ['Tweet', 'Sentiment']
output_nltk = []
for i in range(len(data)):
    sample_text = data['Tweet'][i]
    output = nltk_process(sample_text)
    output_nltk.append(output)

ads_df = pd.DataFrame({'Adjective': [], 'Adverb': [], 'Noun': [], 'Verb': [], 'Polarity': []})
pos_neg_df = pd.DataFrame()
# pos_neg_bad_list = []

for i in range(len(output_nltk)):
    nltkout_temp = pd.DataFrame(output_nltk[i])
    nltkout_temp.columns = ['Word', 'POS']
    nltkout_temp['Word'] = pd.DataFrame(nltkout_temp['Word'].str.lower())
    pos = pos_words(nltkout_temp['Word'])
    neg = neg_words(nltkout_temp['Word'])
    pos_neg_df = pos_neg_df.append({'Num_Positive': pos, 'Num_Negative': neg}, ignore_index=True)
    ads_df = ads_df.append(ads(nltkout_temp))
    ads_df = ads_df.fillna(0)

# print(pos_neg_bad_df)
ads_df['index'] = range(1, len(ads_df) + 1)
data['index'] = range(1, len(data) + 1)
data['label'] = data.apply(f, axis=1)
ads_df_final = pd.merge(ads_df, pd.DataFrame(data), how='left', left_on='index', right_on='index', sort=False)
ads_df_final = ads_df_final[['Adjective', 'Adverb', 'Noun', 'Verb', 'Polarity', 'label']]

pos_neg_df['index'] = range(1, len(pos_neg_df) + 1)
ads_df_final['index'] = range(1, len(ads_df_final) + 1)
ads_df_final_1 = pd.merge(ads_df_final, pos_neg_df, how='left', left_on='index', right_on='index', sort=False)
ads_df_final_1 = ads_df_final_1[['Adjective', 'Adverb', 'Noun', 'Verb', 'Polarity', 'Num_Positive', 'Num_Negative', 'label']]

 # ----Calling the Niave Bayes Classifier--------#
NVB(ads_df_final_1)
dtree(ads_df_final_1)

print("--- %s minutes ---" % ((time.time() - start_time)/60))
