import os
import re
import jieba
from gensim.models import word2vec
import numpy as np
from keras.models import Sequential
from keras.layers.recurrent import LSTM
from keras.layers import RepeatVector, Dense, TimeDistributed

# 设置当前的工作目录
os.chdir('')

# 为语料做分词处理
def word_segment():

    # 打开语料文本
    inputFile_NoSegment = open('', 'rb')
    outputFile_Segment = open('',
                              'w', encoding='utf-8')

    # 读取语料文本中的每一行文字
    lines = inputFile_NoSegment.readlines()

    # 为每一行文字分词
    for i in range(len(lines)):
        line = lines[i]
        if line:
            line = line.strip()
            seg_list = jieba.cut(line)

            segments = ''
            for word in seg_list:
                segments = segments + ' ' + word
            segments += '\n'
            segments = segments.lstrip()

            # 将分词后的语句，写进文件中
            outputFile_Segment.write(segments)

    inputFile_NoSegment.close()
    outputFile_Segment.close()


# 将问-答语句输入X-Y中
def XY():

    # 读取分词后的对话文本
    f = open('../data/chatterbot_segment.txt', 'r', encoding='utf-8')
    subtitles = f.read()

    X = []
    Y = []

    # 将对话文本按段落切分
    subtitles_list = subtitles.split('E')

    # 将“问句”放入X中，将“答句”放入Y中
    for q_a in subtitles_list:
        # 检验段落中，是否含有"问-答"句；如果有，则分别追加到 X 和 Y 中
        if re.findall('.*M.*M.*', q_a, flags=re.DOTALL):
            q_a = q_a.strip()
            q_a_pair = q_a.split('M')

            X.append(q_a_pair[1].strip())
            Y.append(q_a_pair[2].strip())

    f.close()

    return X, Y


# 将X和Y中的词语，转换为词向量，并将问答句的长度统一
def XY_vector(X, Y):

    # 导入训练好的词向量
    model = word2vec.Word2Vec.load('../word_vector/Word60.model')

    # 将X-Y转换为词向量X_vector、Y_vector
    X_vector = []
    for x_sentence in X:
        x_word = x_sentence.split(' ')

        x_sentvec = [model[w] for w in x_word if w in model.vocab]
        X_vector.append(x_sentvec)

    Y_vector = []
    for y_sentence in Y:
        y_word = y_sentence.split(' ')
        y_sentvec = [model[w] for w in y_word if w in model.vocab]
        Y_vector.append(y_sentvec)

    # 计算词向量的维数
    word_dim = len(X_vector[0][0])

    # 设置结束词
    sentend = np.ones((word_dim,), dtype=np.float32)

    # 将问-答句的长度统一
    for sentvec in X_vector:
        if len(sentvec) > 14:
            # 将第14个词之后的全部内容删除，并将第15个词换为sentend
            sentvec[14:] = []
            sentvec.append(sentend)
        else:
            # 将不足15个词的句子，用sentend补足
            for i in range(15 - len(sentvec)):
                sentvec.append(sentend)

    for sentvec in Y_vector:
        if len(sentvec) > 15:
            sentvec[14:] = []
            sentvec.append(sentend)
        else:
            for i in range(15 - len(sentvec)):
                sentvec.append(sentend)

    return X_vector, Y_vector


# 搭建seq2seq模型
def seq2seq(X_vector, Y_vector):
    # 将 X_vector、Y_vector 转化为数组形式
    X_vector = np.array(X_vector, dtype=np.float32)
    Y_vector = np.array(Y_vector, dtype=np.float32)

    # 手动切分数据为：训练集、测试集
    pos = 80
    X_train, X_test = X_vector[:pos], X_vector[pos:]
    Y_train, Y_test = Y_vector[:pos], Y_vector[pos:]

    timesteps = X_train.shape[1]
    word_dim = X_train.shape[2]
    print(X_train.shape)
    print(timesteps)
    print(word_dim)
    print(X_train.shape[1:])

    # 构建一个空容器
    model = Sequential()

    # 编码
    model.add(LSTM(output_dim=word_dim, input_shape=X_train.shape[1:], return_sequences=False))

    # 将问句含义进行复制
    model.add(RepeatVector(timesteps))

    # 解码
    model.add(LSTM(output_dim=word_dim, return_sequences=True))

    # 添加全连接层
    model.add(TimeDistributed(Dense(output_dim=word_dim, activation="linear")))

    # 编译模型
    model.compile(loss='mse', optimizer='Adam', metrics=['accuracy'])

    # 训练、保存模型
    model.fit(X_train, Y_train, nb_epoch=5000, validation_data=(X_test, Y_test))
    model.save('../MyModel/v2/model5000.h5')

    return model

if __name__ == '__main__':
    word_segment()
    X, Y = XY()
    X_vector, Y_vector = XY_vector(X, Y)
    model = seq2seq(X_vector, Y_vector)
