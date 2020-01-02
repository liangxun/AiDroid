"""
Hin2img过程
将每个apk节点表示成一张图片. 64x64
"""
import numpy as np
import re
from collections import deque
import pickle as pkl
from gensim.models import KeyedVectors
from utils import load_adjs
from config import *


delPrefix = re.compile('[^0-9]') #去掉字母，保留node序号
getPrefix = re.compile('[0-9]') #去掉数字,保留前缀

def add_prefix(metapath, node):
    return '{}{}'.format(prefix[metapath], node)

def del_prefix(word):
    return delPrefix.sub('', word)

def get_prefix(word):
    return getPrefix.sub('', word)


class Hin2Img:
    """
    represent each apk as an image.
    """
    def __init__(self, data_path, wv_path):
        self.adjs_fromApk, self.adjs_toApk = load_adjs(data_path)
        self.num_metapath = len(self.adjs_fromApk)
        self.num_apks = len(self.adjs_fromApk[0])
        
        self.wv = KeyedVectors.load(wv_path)
        self.d = self.wv['apk1'].shape[0]
        self.t = self.d
        self.k = 2
    
    def apk2img(self, apk_id):
        img = []
        word = 'apk{}'.format(apk_id)
        # print(word)
        img.append(self.wv[word])
        neighs = self.bfs(word)
        for word in neighs:
            try:
                img.append(self.wv[word])
            except Exception as e:
                print(e)
                pass
        while(len(img) < self.t):
            img.append(np.zeros_like(img[0]))
        return np.array(img)
    
    def bfs(self, start_word):
        neighs_1hop = []
        neighs_2hop = []

        # 1-order neighbors: other type nodes
        for i in metapaths:
            for node in self.adjs_fromApk[i][int(del_prefix(start_word))]:
                neighs_1hop.append(add_prefix(i, node))

        # 2-order neighbors: apks
        for word in neighs_1hop:
            type = predix2mpid[get_prefix(word)]
            for node in self.adjs_toApk[type][int(del_prefix(word))]:
                neighs_2hop.append('apk{}'.format(node))
        
        neighs = neighs_1hop + neighs_2hop
        neighs = neighs[:self.t-1] if len(neighs) > (self.t-1) else neighs
        return neighs


def gen_img(apk_id):
    img = H2I.apk2img(apk_id)
    return img

H2I = Hin2Img(data_path, wv_path)

if __name__ == '__main__':
    apk_ids = list(range(H2I.num_apks))
    import multiprocessing as mp
    pool = mp.Pool(processes=15)
    images = pool.map(gen_img, apk_ids)    
    with open('./data/images', 'wb') as f:
        pkl.dump(images, f)
    print(len(images))
    