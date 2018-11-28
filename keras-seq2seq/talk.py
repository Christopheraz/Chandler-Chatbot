
import os
from keras.models import load_model
from gensim.models import word2vec
import numpy as np
import jieba

os.chdir('D:\PycharmProjects\chatbot\chatbot')

chat_model = load_model('../MyModel/v2/model5000.h5')
wordVector_model = word2vec.Word2Vec.load('../word_vector/Word60.model')

while(True):
    question = input('输入问题：')
    question = question.strip()
    #que_list = jieba.cut(question)
    que_vector = [wordVector_model[w] for w in que_list if w in wordVector_model.vocab]

    # 获得单词的维度
    word_dim = len(que_vector[0])

    # 将每一句话，删减/补足 到仅有15个单词
    sentend = np.ones((word_dim,), dtype=np.float32)
    if len(que_vector) > 14:
        que_vector[14:] = []
        que_vector.append(sentend)
    else:
        for i in range(15 - len(que_vector)):
            que_vector.append(sentend)

    que_vector = np.array([que_vector])

    # 预测答句
    predictions = chat_model.predict(que_vector)
    answer_list = [wordVector_model.most_similar([predictions[0][i]])[0][0] for i in range(15)]
    answer = ''.join(answer_list)
    print(answer)
