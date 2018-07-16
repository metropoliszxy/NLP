#coding:utf-8
from gensim import corpora, models, similarities
import jieba

import sys
reload(sys)
sys.setdefaultencoding('utf-8')


#手工写个文本列表    
sentences = ["我喜欢吃土豆","土豆是个百搭的东西","我不喜欢今天雾霾的北京"] 

# 描述
# append() 方法用于在列表末尾添加新的对象。
# 语法
# append()方法语法：
# list.append(obj)

words=[]
for doc in sentences:
    words.append(list(jieba.cut(doc)))
print words
print "此时输出的格式为unicode"

dic = corpora.Dictionary(words)
print dic
print dic.token2id
print "可以看到各个词或词组在字典中的编号\n"

print "为了方便看，我给了个循环输出："
for word,index in dic.token2id.iteritems():
    print word + " 编号为:"+ str(index)

print "\n词典生成好之后，就开始生成语料库了	BOW词袋模型"
corpus = [dic.doc2bow(text) for text in words]
print corpus

print "\n此时，得到了语料库，接下来做一个TF-IDF变换"
# 可以理解成 将用词频向量表示一句话 变换成为用词的重要性向量表示一句话
# （TF-IDF变换：评估一字词对于一个文件集或一个语料库中的其中一份文件的重要程度。字词的重要性随着它在文件中出现的次数成正比增加，但同时会随着它在语料库中出现的频率成反比下降。）
tfidf = models.TfidfModel(corpus)

vec = [(0, 1), (4, 1)]
print tfidf[vec]
print "\n"

corpus_tfidf = tfidf[corpus]
for doc in corpus_tfidf:
    print doc

#print "\n vec是查询文本向量，比较vec和训练中的三句话相似度 \n"

index = similarities.SparseMatrixSimilarity(tfidf[corpus], num_features=14)
print index
#sims = index[tfidf[vec]]
#print list(enumerate(sims))