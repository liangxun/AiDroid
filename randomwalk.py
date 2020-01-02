"""
多进程随机游走生成序列语料
"""
import numpy as np
import os
from multiprocessing import Pool
import random
import time
from utils import load_adjs
from config import *


class Wakler:
    """
    random walk in HIN
    return: corpus for skip-gram
    """
    def __init__(self, data_path):
        self.adjs_fromApk, self.adjs_toApk = load_adjs(data_path)
        self.num_metapath = len(self.adjs_fromApk)
        self.num_apks = len(self.adjs_fromApk[0])
    
    def choice_node(self, node, metapath=None):
        if metapath is None:
            """
            current node's type is apk:
                1. choice one metapath
                2. sample node within choosed metapath
            """
            probability = np.array([len(self.adjs_fromApk[i][node]) for i in range(self.num_metapath)])
            probability = probability / sum(probability)
            # print(node, probability)
            metapath = np.random.choice(metapaths, p=probability)
            next_node = random.sample(self.adjs_fromApk[metapath][node], 1)[0]
            return next_node, metapath
        else:
            """
            current node's type is not apk:
                sample node within given metapath
            """
            next_node = random.sample(self.adjs_toApk[metapath][node], 1)[0]
            return next_node
    
    def gen_path(self, seed_apk, length):
        """
        生成一条随机游走序列
        """
        cnt = 0
        node = seed_apk
        walkpath = ['apk{}'.format(node),]
        while(cnt < length):
            # current node's type is apk
            node, metapath = self.choice_node(node)
            walkpath.append('{}{}'.format(prefix[metapath],node))

            # current node's type is others
            node = self.choice_node(node, metapath)
            walkpath.append('apk{}'.format(node))
            cnt += 2

        return walkpath

    def wrapper_gen_path(self, seed, length):
        pid = os.getpid()
        print("pid{} start".format(pid))
        cnt = 0
        with open('./data/corpus/pid{}.txt'.format(pid), 'w') as f:
            for node in seed:
                print("[pid{}] walk[{}] start from {}".format(pid, cnt, node))
                cnt += 1
                path = self.gen_path(node, length)
                f.write(" ".join(path)+'\n')
        print("pid{} end".format(pid))

    def run(self, epoch=20, length=100):
        seeds = self.gen_seeds(epoch)
        print("start random walk")
        p = Pool()
        for i in range(epoch):
            p.apply_async(self.wrapper_gen_path, args=(seeds[i], length))
        p.close()
        p.join()
        print("all subprocess done")

    def gen_seeds(self, epoch):
        apks = list(range(self.num_apks))
        seeds = []
        for _ in range(epoch):
            np.random.shuffle(apks)
            seeds.append(apks)
        return seeds


if __name__ == '__main__':
    data_dir = './data'
    corpus_path = './data/corpus'
    W = Wakler(data_dir)
    W.run(epoch=20, length=100)
