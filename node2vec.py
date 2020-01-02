"""
使用gensim/word2vec为节点生成embeddings
"""
from gensim.models import Word2Vec
from gensim.models.word2vec import PathLineSentences, LineSentence


def node2vec(corpus_dir, save_wv):
    print("load corpus")
    walks = PathLineSentences(corpus_dir)
    print("train word2vec model")
    model = Word2Vec(walks, size=64)
    print("save model.wv")
    model.wv.save(save_wv)

if __name__ == '__main__':
    corpus_dir = "./data/corpus"
    save_wv = "./data/KeyedVectors.wv"
    node2vec(corpus_dir, save_wv)
