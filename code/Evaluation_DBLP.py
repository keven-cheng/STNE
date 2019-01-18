# -*- coding:utf-8 -*-
__author__ = 'chengxiaotao'
__date__ = '2018/8/24 上午5:51'
__product__ = 'PyCharm'
__filename__ = 'Evalute_DBLP'

### 读入文件
import gensim
import gensim.utils as ut
from sklearn.cross_validation import train_test_split
from sklearn.svm import LinearSVC,SVC
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from gensim.models.doc2vec import Doc2Vec
from sklearn.neighbors import KNeighborsClassifier
from sklearn import preprocessing
from collections import namedtuple

train_size = 0.7  #Percentage of training samples
random_state = 2
NetworkSentence = namedtuple('NetworkSentence', 'vectors tags labels time index') # 向量；序号ID；标签；索引


# read the file and save vectors
def readNetworkData(dir, stemmer=0):#dir, directory of network dataset
    allindex = {}
    allvectors = []
    labelset = set()
    with open(dir + '/disciplinary2_auth_label_time.txt') as f1, open(dir + '/inter_disciplinary2_auth_label_time.txt') as f2:

        for l1 in f1:
            # tokens = ut.to_unicode(l1.lower()).split()；大小写的规范化处理
            if stemmer == 1:
                l1 = gensim.parsing.stem_text(l1)
            else:
                l1 = l1.lower()
            tokens = ut.to_unicode(l1).split(sep='|')

            vectors = tokens[1]
            # print(vectorss)
            tags = [tokens[0]]  # ID of each document, for doc2vec model
            index = len(allvectors)
            allindex[tokens[0]] = index  # A mapping from documentID to index, start from 0
            time = tokens[3]
            labels = tokens[2]  # class label
            labelset.add(labels)
            allvectors.append(NetworkSentence(vectors, tags, labels, time, index))

        for l2 in f2:
            if stemmer == 1:
                l1 = gensim.parsing.stem_text(l2)
            else:
                l1 = l2.lower()
            tokens2 = ut.to_unicode(l1).split(sep='|')

            vectors = tokens2[1]
            # print(vectorss)
            tags = [tokens2[0]]  # ID of each document, for doc2vec model
            index = len(allvectors)
            allindex[tokens2[0]] = index  # A mapping from documentID to index, start from 0
            time = tokens2[3]
            labels = tokens2[2]  # class label
            labelset.add(labels)
            allvectors.append(NetworkSentence(vectors, tags, labels, time, index))

    return allvectors, allindex, list(labelset)


# save the lable file to txt
def text_save(filename, data):
    file = open(filename,'a')
    for i in data:
        tags = i[1]   # 用户标记
        label = i[2]   # 用户标签
        time = i[3]  # 标记时刻
        s = tags[0] + ' ' + label + ' ' + str(time)  # 原来的time后面有一个换行，所以就不用加了；
        file.write(s)
    file.close()
    print('标签文件保存成功')



def evaluation(train_vecs, test_vecs, train_y, test_y, classifierStr='SVM', normalize=0):

    if classifierStr == 'KNN':
        print('Training NN classifier...')
        classifier = KNeighborsClassifier(n_neighbors=1)
    else:
        print('Training SVM classifier...')

        classifier = LinearSVC()

    train_vec = []
    for i in train_vecs:
        tokens = ut.to_unicode(i[0]).split(sep=' ')
        templist = []
        for j in range(64):
            templist.append(float(tokens[j]))  #将字符串数字转换为float类型
        train_vec.append(templist)

    # print(train_vec[1])

    test_vec = []
    for i in test_vecs:
        tokens = ut.to_unicode(i[0]).split(sep=' ')
        templist = []
        for j in range(64):
            templist.append(float(tokens[j]))  #将字符串数字转换为float类型
        test_vec.append(templist)

    # print(test_vec[1])

    classifier.fit(train_vec, train_y)
    y_pred = classifier.predict(test_vec)   ##  给出测试向量的类别预测；最好我要知道该向量在个类别上的预测概率

    cm = confusion_matrix(test_y, y_pred)

    print(cm)
    acc = accuracy_score(test_y, y_pred)
    print(acc)

    macro_f1 = f1_score(test_y, y_pred,pos_label=None, average='macro')
    micro_f1 = f1_score(test_y, y_pred,pos_label=None, average='micro')

    per = len(train_y) * 1.0 /(len(test_y)+len(train_y))
    print('Classification method:'+classifierStr+'(train, test, Training_Percent): (%d, %d, %f)' % (len(train_y),len(test_y), per ))
    print('Classification Accuracy=%f, macro_f1=%f, micro_f1=%f' % (acc, macro_f1, micro_f1))
    #print(metrics.classification_report(test_y, y_pred))

    return acc, macro_f1, micro_f1


if __name__ == '__main__':
    directory = '../LABELS/DBLP' #data directory
    allvectors, allindex, classlabels = readNetworkData(directory)  # 读入数据并将数据放到一起；
    print('%d users' % len(allvectors))
    print('%d classes' % len(classlabels))
    vector_list = allvectors[:]  # 这个list中应该包含ID 数据 标签值的，方便用于切分；for reshuffling per pass
    # text_save('DBLP_label.txt',vector_list)

    train, test = train_test_split(vector_list, train_size=train_size, random_state=random_state)

    train_vecs = [[user.vectors] for user in train]  # 加[]后可以表示，
    test_vecs = [[user.vectors] for user in test]

    train_y = [user.labels for user in train]
    test_y = [user.labels for user in test]

    print('train y: , test y: ', len(train_y), len(test_y))

    acc, macro_f1, micro_f1 = evaluation(train_vecs, test_vecs, train_y, test_y, classifierStr='SVM', normalize=0)

