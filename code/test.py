# -*- coding:utf-8 -*-
__author__ = 'chengxiaotao'
__date__ = '2018/8/26 下午6:27'
__product__ = 'PyCharm'
__filename__ = 'test'

# 用于测试实现将字符串中的一串数字转为各自一个个的浮点数；

import gensim.utils as ut

list = [['0.280260 -0.003215 -0.114778 -0.118665'],['1 -0.003215 -0.114778 -0.118665'],['2 -0.003215 -0.114778 -0.118665']]
# list嵌套list中的第一个元素；
list2 = []
for i in list:
    tokens = ut.to_unicode(i[0]).split(sep=' ')
    list3 = []
    for j in range(4):
        list3.append(float(tokens[j]))
        print(tokens[j])
    list2.append(list3)

print(list[1:])

