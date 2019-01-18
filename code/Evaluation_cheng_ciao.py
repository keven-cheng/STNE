# -*- coding:utf-8 -*-
__author__ = 'chengxiaotao'
__date__ = '2018/8/29 上午9:42'
__product__ = 'PyCharm'
__filename__ = 'Evaluation_cheng'

### 读入文件
import gensim
import gensim.utils as ut
from sklearn.cross_validation import train_test_split
from sklearn.svm import LinearSVC, SVC
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from collections import namedtuple
from itertools import islice

train_size = 0.6   # Percentage of training samples
random_state = 2
NetworkSentence = namedtuple('NetworkSentence', 'tags vectors labels')  # 向量；序号ID；标签；索引


# read the file and save it to dict and return；提取标签信息，生成标签字典；
def readLabelsData(dir):
    labelDict = dict()
    with open(dir) as f1:
        for l1 in f1:
            tokens = ut.to_unicode(l1).split(sep=' ')
            labelDict[tokens[0]] = tokens[1]
    return labelDict


# dir, directory of embedding dataset；提取生成的嵌入表示数据
def readEmbeddingData(dir, labelDict, stemmer=0):
    index = []
    tags = []
    Embeddings = []
    count = 0
    with open(dir+'/spatiotemporal_114.stwalkone') as f1:   # spatiotemporal_window_4.stwalktwo
        for l1 in islice(f1, 1, None):      # 数字1控制从第几行开始读取，islice(seq,start,stop,step)
            # tokens = ut.to_unicode(l1.lower()).split()；大小写的规范化处理
            if stemmer == 1:
                l1 = gensim.parsing.stem_text(l1)
            else:
                l1 = l1.lower()
            tokens = ut.to_unicode(l1).split(sep=' ')  # 空格切分

            index = len(tags)
            tag = tokens[0].split('_')[0]      # 对于ciao数据集，用户名形式为1344_4 ，需要去掉后面的时间段标记信息；根据_进行切分，取前半段
            if tag in labelDict:
                label = labelDict[tag]   # 如果有这个标签，则给出它的标签信息；增加一个如果该ID的标签找不到的异常处理

                verctors = []
                for i in range(len(tokens[1:])-1):   # make the embeddings insert into the vectors
                    verctor = float(tokens[i+1])
                    verctors.append(verctor)

                tags.append(tag)
                Embeddings.append(NetworkSentence(tag, verctors, label))
            else:
                count = count+1
        print('%s label not found, end' % count)   # 统计有多少标签找不到的，找不到的也就没法评测

    return index, tags, Embeddings


# save the lables file to txt
def text_save(filename, data):
    file = open(filename, 'a')
    for i in data:
        tags = i[0]    # 用户标记;现在是第0个
        vector = i[1]  # list类型数据
        label = i[2]   # 用户标签

        s = tags+' '+str(vector)+' '+str(label)+'\n'  # 原来的文件label后面有一个换行，去掉后文末再加一个；
        file.write(s)
    file.close()
    print('%s 测评文件保存成功' % filename)


# 计算各项分类指标
def evaluation(train_vecs, test_vecs, train_y, test_y, classifierStr='SVM', normalize=0):

    if classifierStr == 'KNN':
        print('Training NN classifier...')
        classifier = KNeighborsClassifier(n_neighbors=1)
    else:
        print('Training LinearSVC classifier...')
        classifier = LinearSVC()
        # class_weight = dict({'0': 2.7, '1': 4})   # 设置类别权重
        # classifier = RandomForestClassifier(bootstrap=True, class_weight=class_weight)

    classifier.fit(train_vecs, train_y)
    y_pred = classifier.predict(test_vecs)   # 给出测试向量的类别预测；最好我要知道该向量在个类别上的预测概率
    cm = confusion_matrix(test_y, y_pred)

    print(cm)
    acc = accuracy_score(test_y, y_pred)
    recall = recall_score(test_y, y_pred, pos_label='1')  # 不知为啥设置pos_label='1'recall为0
    # print(acc, recall)

    macro_f1 = f1_score(test_y, y_pred, pos_label=None, average='macro')
    micro_f1 = f1_score(test_y, y_pred, pos_label=None, average='micro')

    per = len(train_y) * 1.0 /(len(test_y)+len(train_y))
    print('Classification method:'+classifierStr+'(train, test, Training_Percent): (%d, %d, %f)' % (len(train_y),
                                                                                                    len(test_y), per))
    print('Classification Accuracy=%f, Recall=%f, macro_f1=%f, micro_f1=%f' % (acc, recall, macro_f1, micro_f1))

    return acc, macro_f1, micro_f1


if __name__ == '__main__':
    directory_label = '../LABELS/CIAO_label.txt'              # data directory
    directory_embedding = '../ciao/output_stwalkone'          # 嵌入表示数据位置
    # label_dict = dict()

    label_dict = readLabelsData(directory_label)  # 读入数据并将数据放到一起；
    index, tags, Embeddings = readEmbeddingData(directory_embedding, label_dict)

    print('This dataset have %d users totally.' % len(label_dict))
    # print('%d classes' % len(label_dict.values(0)))

    vector_list = Embeddings[:]  # 这个list中应该包含ID 数据 标签值的，方便用于切分；for reshuffling per pass
    # text_save('Ciao_ceshi.txt', vector_list)

    train, test = train_test_split(vector_list, train_size=train_size, random_state=random_state)

    train_vecs = [user.vectors for user in train]  # 加[]后可以表示，注意这里的user.vectors是否用加括号
    test_vecs = [user.vectors for user in test]

    train_y = [user.labels for user in train]
    test_y = [user.labels for user in test]

    train_neg = train_y.count('0')
    train_pos = train_y.count('1')
    print('train y: %d, test y: %d' % (len(train_y), len(test_y)))
    print('Train negative sample: %d, Train positive sample: %d' % (train_neg, train_pos))

    acc, macro_f1, micro_f1 = evaluation(train_vecs, test_vecs, train_y, test_y, classifierStr='SVM', normalize=0)