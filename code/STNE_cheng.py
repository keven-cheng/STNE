# -*- coding:utf-8 -*-
#####################
# RWR_STNE
# Learns trajectory representation of graph nodes by considering spatial and temporal neighbors together
# revised by Chengxiaotao
#####################

import random
from gensim.models import Word2Vec
import networkx as nx
import sys
from multiprocessing import cpu_count
import time
import argparse
import os


# 修改STwalk的时空图创建部分；由过去时刻的串联式改为发散式
def createSpaceTimeGraph(G_list, time_window, start_node, time_step):
    """
     time step is necessary because we want representation only for last time step and
     we will create the space-time graph for [time_step, time_step-1,time_step-2,...,time_step-time_window]
    """
    G = G_list[-1]  # 将最后一个时刻图作为当前图；
    for time1 in range(1, time_window + 1):    # 在一定时间窗口内
        past_node = start_node.split("_")[0] + "_" + str(time_step - time1)   # 新建一个过去时间间隔内的该节点的多个副节点，加_4;_3等区分
        # TODO 下面这句话写的有问题，当使用DBLP数据集时应该是start_node;而使用Ciao数据集时可以用past_node，节点标记不同
        if past_node not in list(G_list[time_window+1 - time1].nodes()):  # if start_node not in G_list:
            #print G_list[time_step - time1].nodes()
            #print("not exist in G_list")
            continue
        else:
            G.add_edge(start_node, past_node)  # 将新老时刻节点间建立连边；每一个同最初的节点建立连边

            G_past = G_list[-time1 - 1]        # 上一时刻的图节点

            # considering first level neighbors；一阶邻居点
            past_neighbors = list(G_past.neighbors(past_node))
            temp = []

            # considering second level neighbors
            for elt in past_neighbors:
                temp = temp + list(G_past.neighbors(elt))  # 不断将邻居的邻居节点放到temp中来

            # merging list of level-1 and level-2 neighbors；邻居节点为1阶和2阶之和
            past_neighbors = past_neighbors + temp
            past_neighbors.append(past_node)

            # subgraph of G_past containing nodes from "past_neighbors" and edges between those nodes.使用这一子图函数，将这些节点提取出
            past_subgraph = G_past.subgraph(past_neighbors)

            # merge current graph with past subgraphs；
            G = nx.compose(G, past_subgraph)  # 合并两个图，相同节点-边进行合并；其实就是这段时间内的所有点边的集合
            start_node = past_node
    return G


# 定义随机游走的方式；创新改变的地方
def random_walk(SpaceTimegraph, path_length, rand=random.Random(0), start=None):
    """ Returns a truncated random walk.
        path_length: Length of the random walk.
        removed = alpha: probability of restarts.
        start: the start node of the random walk.
        rand = random.Random(0) 表示rand代表一个随机数，从序列元素中随机选一个
    """
    G = SpaceTimegraph
    if start:
        path = [start]    # 将start节点放入path的这个list中；
    else:
        sys.exit("ERROR: Start node not mentioned for random_walk")

    while len(path) < path_length:
        cur = path[-1]  # 当前是路径中最后一个；即取出游走的最后一个节点作为当前节点
        if len(G[cur]) > 0:   # 这个G[cur]具体什么意思
            # TODO 选择邻居节点的方式可以有所改变；重启的概率未应用
            path.append(rand.choice(list(G[cur])))  # 这个代码是从当前节点邻居中随机选一个放入path中，
            # print(G[cur])  # 便于自己理解G[cur]里存的是什么，就是cur节点的当前的对应邻居节点，G图中用的是字典的数据类型
        else:
            break
    return path


# 关键是在这个时间图中如何产生游走序列walks，后面的问题就是直接采用word2vec了；
def create_vocab(G_list, num_restart, path_length, nodes, time_step, rand=random.Random(0), time_window=1):
    walks = []

    nodes = list(nodes)

    # number of path is equal to number of restarts per node
    for cnt in range(num_restart):
        rand.shuffle(nodes)
        start = time.time()  # 程序开始执行时间
        for node in list(nodes):
            G = createSpaceTimeGraph(G_list, time_window, node, time_step)  # 主要这里的创建时间空间图需要用到G_list；
            walks.append(random_walk(SpaceTimegraph=G, path_length=path_length, rand=rand, start=node))
    print("Vocabulary created")
    return walks


def STNE(input_direc, output_file, number_restart, walk_length, representation_size, time_step,
            time_window_size, workers, vocab_window_size):
    """
    This function generates representation for all nodes in space-time-graph of all nodes of graph at t=time_step
    however we will consider only representations of nodes present in graph at t = time_step;
    每一区间段内进行一次STNE表示；
    """
    if time_window_size > time_step:
        sys.exit("ERROR: time_window_size(=" + str(time_window_size) + ") cannot be more than time_step(=" + str(
            time_step) + "):")

    # G_list中存放的是时间窗为5的，5个时间片段的图数据
    G_list = [nx.read_graphml(input_direc + "/graph_" + str(i) + ".graphml") for i in
              range(time_step - time_window_size, time_step + 1)]

    # get list of nodes
    nodes = G_list[-1].nodes()
    print("Creating vocabulary...")  # 生成节点的游走序列
    walks = create_vocab(G_list, num_restart=number_restart, path_length=walk_length, nodes=nodes,
                         rand=random.Random(0), time_step=time_step, time_window=time_window_size)

    # time-step is decremented by 1, because, time steps are from 0 to time_step-1=total time_step length
    print("Generating representation...")
    model = Word2Vec(walks, size=representation_size, window=vocab_window_size, min_count=0, workers=workers)

    model.wv.save_word2vec_format(output_file)
    print("Representation File saved: " + output_file)
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', action='store', dest='dataset', help='Dataset')
    arg = parser.parse_args()
    print('dataset =', arg.dataset)

    if arg.dataset not in ["epinion","dblp",'ciao']:
        print("Invalid dataset.\nAllowed datsets are epinion, dblp, ciao")
        sys.exit(0)
    # by default we use Epinion dataset for experiment
    direc = "../epinion"
    max_timestep = 109 # Epinion dataset has 0 to 109 graphs

    if arg.dataset == "ciao":
        direc = "../"+arg.dataset
        max_timestep = 114
    elif arg.dataset == "dblp":
        direc = "../"+arg.dataset
        max_timestep = 44

    number_restart = 40
    walk_length = 10
    representation_size = 64
    vocab_window_size = 5
    time_window_size = 5  # number of previous time steps graphs plus current time step graph

    if not os.path.exists(direc+"/output_STNE"):  # 如果不存在就创建这一目录
        os.makedirs(direc+"/output_STNE")

    workers = cpu_count()
    seq = []  # 存放的采样的时间点序列

    for t in range(0, max_timestep+1):
        if (t + 1) % time_window_size == 0:
            seq.append(t)       # 生成一定间隔的时间段并拼接

    for t in seq:
        print("\n Generating " + str(representation_size) + " dimension embeddings for nodes")
        time_step = t
        start = time.time()
        STNE(input_direc=direc + "/input_graphs",
                output_file=direc + "/output_STNE/spatiotemporal_" + str(time_step) + ".stne",
                number_restart=number_restart,
                walk_length=walk_length, vocab_window_size=vocab_window_size,
                representation_size=representation_size, time_step=time_step, time_window_size=time_window_size - 1,
                workers=workers)  # 根据不同的时刻t进入STWalk1进行计算
