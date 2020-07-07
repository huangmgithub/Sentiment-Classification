import numpy as np
import pandas as pd
import jieba
import re
import pickle
from hanziconv import HanziConv

np.random.seed(42)
import gc
from sklearn.metrics import roc_auc_score
from keras.models import Model
from keras.layers import Input, Embedding, Dense, Conv2D
from keras.layers import Reshape, Flatten, Concatenate, Dropout, SpatialDropout1D, GRU, Bidirectional
from keras import initializers, regularizers, constraints
from keras.engine.topology import Layer
from keras import backend as K
from keras.losses import categorical_crossentropy
from keras.metrics import categorical_accuracy

from keras.preprocessing import text, sequence
from keras.callbacks import Callback
from gensim.models import KeyedVectors

stopwords_path = "./data/chinese_stopwords.txt"

class StringFilterZH():
    """
    中文分词，过滤标点，去除停用词的类
    """
    def __init__(self):
        self.stopwords = stopwords_path

    def get_stopwords(self):
        """获得停用词"""
        with open(self.stopwords, 'r', encoding='utf-8') as f:
            return set([line.strip() for line in f])

    def cut(self, string):
        """切分词"""
        return list(jieba.cut(string))

    def token(self, string):
        """匹配文字"""
        return re.findall(r'\w+', string)

    def to_simplified(self, string):
        """繁体转换成简体"""
        return HanziConv.toSimplified(string)

    def to_tokens(self, string, use_stopwords=True):
        """
        繁体转换为简体，分词，去除停用词，去除标点保留文字
        :param string:
        :param use_stopwords:
        :return: 词与词空格分隔开
        """
        stopwords = self.get_stopwords()
        string = self.to_simplified(string)
        tokens = ""
        for word in self.cut("".join(self.token(string))):
            if use_stopwords:
                if word not in stopwords:
                    tokens += word + " "
            else:
                tokens += word + " "
        return tokens.strip()

def reformat(labels):
    """label[-1,-2,1,2]转换成[0,1,0,0,1,0,0,0...]"""
    labels = (np.arange(-2, 2) == labels.reshape(-1, 1)).astype(np.float32)
    return labels.flatten() #

# 实例化文本清洗
string_filter = StringFilterZH()

def data_extract(data):
    y_label = data.iloc[:, 2:len(data.columns)].values
    # 6大类的标签
    y_label_l1 = np.stack((np.any(y_label[:, 0:3] != -2, axis=1).astype(np.float32),
                               np.any(y_label[:, 3:7] != -2, axis=1).astype(np.float32),
                               np.any(y_label[:, 7:10] != -2, axis=1).astype(np.float32),
                               np.any(y_label[:, 10:14] != -2, axis=1).astype(np.float32),
                               np.any(y_label[:, 14:18] != -2, axis=1).astype(np.float32),
                               np.any(y_label[:, 18:20] != -2, axis=1).astype(np.float32))).T
    # 20小类的标签
    y_label_l2 = np.zeros((y_label.shape[0], 80))
    for i in range(y_label_l2.shape[0]):
        y_label_l2[i] = reformat(y_label[i])
    y = np.hstack((y_label_l2, y_label_l1)) # label 86

    # 处理文本内容
    x = data["content"].apply(string_filter.to_tokens).values

    return x, y

class ProcessSequence:
    def __init__(self):
        self.vocab_size = 35000 # 词汇表的大小
        self.max_len = 300 # 序列最大长度
        self.embed_size = 200 # 词嵌入大小
        self.embedding_path = r"./model/word2vec/wiki.zh.model"

    def to_sequence(self, *args):
        """
        返回序列化文本，Embedding词典
        :param args: (train， valid， test)
        :return:
        """
        tokenizer = text.Tokenizer(num_words=self.vocab_size)

        corpus = []
        for i in args: corpus += list(i)
        tokenizer.fit_on_texts(corpus) # 词表

        data = []
        for i in args:
            x_data = tokenizer.texts_to_sequences(i)
            x_data = sequence.pad_sequences(x_data, maxlen=self.max_len) # pad sequence
            data.append(x_data)

        model_wv = KeyedVectors.load(self.embedding_path, mmap='r') # Load 预训练的词向量
        word_index = tokenizer.word_index
        nb_words = min(self.vocab_size, len(word_index))
        embedding_matrix = np.zeros((nb_words, self.embed_size)) # embedding表
        for word, i in word_index.items():
            if i >= self.vocab_size: continue
            try:
                embedding_vector = model_wv[word]
            except:
                embedding_vector = None
            if embedding_vector is not None: embedding_matrix[i] = embedding_vector
        del model_wv # 清除内存
        _ = gc.collect() # 垃圾回收

        data.append(embedding_matrix)
        return tuple(data)

class RocAucEvaluation(Callback):
    """RocAuc性能评估"""
    def __init__(self, validation_data=(), interval=1):
        super(Callback, self).__init__()

        self.interval = interval
        self.X_val, self.y_val = validation_data

    def on_epoch_end(self, epoch, logs={}):
        if epoch % self.interval == 0:
            y_pred = self.model.predict(self.X_val, verbose=0)
            score = roc_auc_score(self.y_val, y_pred)
            print("\n ROC-AUC - epoch: %d - score: %.6f \n" % (epoch+1, score))

class LossHistory(Callback):
    """记录训练时loss acc变化"""
    def on_train_begin(self, logs={}):
        self.losses = {'batch':[], 'epoch':[]}
        self.accuracy = {'batch':[], 'epoch':[]}
        self.val_loss = {'batch':[], 'epoch':[]}
        self.val_acc = {'batch':[], 'epoch':[]}
    def on_batch_end(self, batch, logs={}):
        keys = list(logs.keys())
        self.losses['batch'].append(logs.get(keys[0]))
        self.accuracy['batch'].append(logs.get(keys[1]))
        self.val_loss['batch'].append(logs.get(keys[2]))
        self.val_acc['batch'].append(logs.get(keys[3]))
    def on_epoch_end(self, batch, logs={}):
        keys = list(logs.keys())
        self.losses['epoch'].append(logs.get(keys[0]))
        self.accuracy['epoch'].append(logs.get(keys[1]))
        self.val_loss['epoch'].append(logs.get(keys[2]))
        self.val_acc['epoch'].append(logs.get(keys[3]))

class Attention(Layer):
    '''自定义attention层，基于Hierarchical Attention Networks for Document Classification'''
    def __init__(self, step_dim,
                 W_regularizer=None, b_regularizer=None,
                 W_constraint=None, b_constraint=None,
                 bias=True, **kwargs):
        self.supports_masking = True
        self.init = initializers.get('glorot_uniform')

        self.W_regularizer = regularizers.get(W_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.b_constraint = constraints.get(b_constraint)

        self.bias = bias
        self.step_dim = step_dim
        self.features_dim = 0
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3
        # Embed_size
        self.W = self.add_weight(shape=(input_shape[-1],),
                                 initializer=self.init,
                                 name='{}_W'.format(self.name),
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)
        self.features_dim = input_shape[-1]

        if self.bias:
            # c*filters
            self.b = self.add_weight(shape=(input_shape[1],),
                                     initializer='zero',
                                     name='{}_b'.format(self.name),
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint)
        else:
            self.b = None
        self.built = True

    def compute_mask(self, input, input_mask=None):
        return None

    def call(self, x, mask=None):
        features_dim = self.features_dim
        step_dim = self.step_dim
        # 对CNN输出进行线性变换
        eij = K.reshape(K.dot(K.reshape(x, (-1, features_dim)),
                        K.reshape(self.W, (features_dim, 1))), (-1, step_dim))

        if self.bias:
            eij += self.b
        # Tanh
        eij = K.tanh(eij)

        a = K.exp(eij)

        if mask is not None:
            a *= K.cast(mask, K.floatx())
        # Softmax变换
        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())

        a = K.expand_dims(a)
        # CNN输出加权
        weighted_input = x * a
        return K.sum(weighted_input, axis=1)

    def compute_output_shape(self, input_shape):
        return input_shape[0],  self.features_dim


class Model_GRU:
    """
    基于双层双向GRU模型, 6大类
    """
    def __init__(self):
        self.hidden_size_1 = 128
        self.hidden_size_2 = 64

        self.vocab_size = 35000
        self.max_len = 300
        self.embed_size = 200

        self.epochs = 2
        self.batch_size = 64
        self.model = None

    def build_model(self, embedding_matrix):
        """
        建立模型
        :param embedding_matrix:
        :return:
        """
        inp = Input(shape=(self.max_len,))
        x = Embedding(self.vocab_size, self.embed_size, weights=[embedding_matrix])(inp)
        x = SpatialDropout1D(0.2)(x)

        GRU1 = Bidirectional(GRU(self.hidden_size_1, return_sequences=True, recurrent_dropout=0.2,
                                 input_shape=(self.max_len, self.embed_size)))(x)
        GRU2 = Bidirectional(GRU(self.hidden_size_1, return_sequences=False, recurrent_dropout=0.2,
                                 input_shape=(self.max_len, self.hidden_size_1)))(GRU1)

        z = Dropout(0.2)(GRU2)

        fc = Dense(self.hidden_size_2, activation="relu")(z)

        output = Dense(6, activation="sigmoid")(fc)

        model = Model(inputs=inp, outputs=output)
        model.compile(loss="binary_crossentropy",
                      optimizer="adam",
                      metrics=["accuracy"])
        return model

    def train(self, x_train, y_train, x_valid, y_valid, embedding_matrix):
        """
        训练模型
        :param x_train:
        :param y_train:
        :param x_valid:
        :param y_valid:
        :param embedding_matrix:
        :return:
        """
        self.model = self.build_model(embedding_matrix)
        history = LossHistory()
        self.model.fit(x_train, y_train[:, 80:], batch_size=self.batch_size,
                       epochs=self.epochs, validation_data=(x_valid, y_valid[:, 80:]),
                       callbacks=[history])
        return history

class Model_CNN_Attention:
    """基础TextCNN模型，20小类"""
    def __init__(self):
        self.vocab_size = 35000
        self.max_len = 300
        self.embed_size = 200

        self.num_filters = 32
        self.filter_size = [1,2,3,5]

        self.epochs = 3
        self.batch_size = 128
        self.model  = None

        self.drop = 0.5

    def build_model(self, embedding_matrix):
        """
        建立模型
        :param embedding_matrix:
        :return:
        """
        inp = Input(shape=(self.max_len,))
        x = Embedding(self.vocab_size, self.embed_size, weights=[embedding_matrix])(inp)
        x = SpatialDropout1D(0.2)(x)
        x = Reshape((self.max_len, self.embed_size, 1))(x)

        conv_0 = Conv2D(self.num_filters, kernel_size=(self.filter_size[0], self.embed_size), padding="valid",
                        kernel_initializer="normal", activation="relu")(x)
        conv_1 = Conv2D(self.num_filters, kernel_size=(self.filter_size[1], self.embed_size), padding="valid",
                        kernel_initializer="normal", activation="relu")(x)
        conv_2 = Conv2D(self.num_filters, kernel_size=(self.filter_size[2], self.embed_size), padding="valid",
                        kernel_initializer="normal", activation="relu")(x)
        conv_3 = Conv2D(self.num_filters, kernel_size=(self.filter_size[3], self.embed_size), padding="valid",
                        kernel_initializer="normal", activation="relu")(x)

        z = Concatenate(axis=1)([conv_0, conv_1, conv_2, conv_3]) # (b, c, embed_size, filters)
        print(z.ndim)
        print(z.shape)
        a_shape = (int(z.shape[1]), int(z.shape[-1])) # (embed_size, filters)
        z1 = Reshape((-1, a_shape[1]))(z) # (b, c*filters, embed_size)

        outputs = []
        for i in range(20): # 20小类
            z = Attention(a_shape[0])(z1)
            z = Dropout(0.1)(z)
            z = Dense(64, activation="relu")(z)
            out = Dense(4, activation="softmax")(z)
            outputs.append(out)
        output = Concatenate()(outputs)
        model = Model(inputs=inp, outputs=output)

        def loss_a(y_true, y_pred):
            """
            计算损失
            :param y_true:
            :param y_pred:
            :return:
            """
            loss_sum = 0
            for i in range(0, 80, 4):
                loss_sum += categorical_crossentropy(y_true[:, i:i+4], y_pred[:, i:i+4])
            return loss_sum

        def acc(y_true, y_pred):
            """
            计算准确率
            :param y_true:
            :param y_pred:
            :return:
            """
            a = 0

            for i in range(0, 80, 4):
                a += categorical_accuracy(y_true[:, i:i+4], y_pred[:, i:i+4])
            return a / 20

        model.compile(loss=loss_a,
                      optimizer="adam",
                      metrics=[acc])

        return model

    def train(self, x_train, y_train, x_valid, y_valid, embedding_matrix):
        """
        训练模型
        :param x_train:
        :param y_train:
        :param x_valid:
        :param y_valid:
        :param embedding_matrix:
        :return:
        """
        self.model = self.build_model(embedding_matrix)
        history = LossHistory()
        self.model.fit(x_train, y_train[:, :80], batch_size=self.batch_size,
                       epochs=self.epochs, validation_data=(x_valid, y_valid[:, :80]),
                       callbacks=[history])

class Model_GRU_Attention:
    """基础GRU模型"""

    def __init__(self):
        self.vocab_size = 35000
        self.max_len = 300
        self.embed_size = 200

        self.epochs = 3
        self.batch_size = 64
        self.model = None

        self.drop = 0.5

    def build_model(self, embedding_matrix):
        """
        建立模型
        :param embedding_matrix:
        :return:
        """
        inp = Input(shape=(self.max_len,))
        x = Embedding(self.vocab_size, self.embed_size, weights=[embedding_matrix])(inp)
        x = SpatialDropout1D(0.2)(x)
        x1 = Bidirectional(GRU(128, return_sequences=True))(x)

        outputs = []
        for i in range(20):  # 20小类
            z = Attention(self.max_len)(x1)
            z = Dropout(0.1)(z)
            z = Dense(64, activation="relu")(z)
            out = Dense(4, activation="softmax")(z)
            outputs.append(out)
        output = Concatenate()(outputs)
        model = Model(inputs=inp, outputs=output)

        def loss_a(y_true, y_pred):
            """
            计算损失
            :param y_true:
            :param y_pred:
            :return:
            """
            loss_sum = 0
            for i in range(0, 80, 4):
                loss_sum += categorical_crossentropy(y_true[:, i:i + 4], y_pred[:, i:i + 4])
            return loss_sum

        def acc(y_true, y_pred):
            """
            计算准确率
            :param y_true:
            :param y_pred:
            :return:
            """
            a = 0

            for i in range(0, 80, 4):
                a += categorical_accuracy(y_true[:, i:i + 4], y_pred[:, i:i + 4])
            return a / 20

        model.compile(loss=loss_a,
                      optimizer="adam",
                      metrics=[acc])

        return model

    def train(self, x_train, y_train, x_valid, y_valid, embedding_matrix):
        """
        训练模型
        :param x_train:
        :param y_train:
        :param x_valid:
        :param y_valid:
        :param embedding_matrix:
        :return:
        """
        self.model = self.build_model(embedding_matrix)
        history = LossHistory()
        self.model.fit(x_train, y_train[:, :80], batch_size=self.batch_size,
                       epochs=self.epochs, validation_data=(x_valid, y_valid[:, :80]),
                       callbacks=[history])

if __name__ == "__main__":
    data_files = [
        "./data/train_set.csv",
        "./data/valid_set.csv",
        "./data/test_set.csv"
    ]

    # 文本处理，标签onehot编码
    x_train, y_train = data_extract(pd.read_csv(data_files[0]))
    x_valid, y_valid = data_extract(pd.read_csv(data_files[1]))
    x_test, _ = data_extract(pd.read_csv(data_files[2]))

    x_train, y_train = x_train[:5000], y_train[:5000]
    x_valid, y_valid = x_valid[:500], y_valid[:500]

    # 序列化文本
    process_sequence = ProcessSequence()
    x_train, x_valid, x_test, embedding_matrix = process_sequence.to_sequence(x_train, x_valid, x_test)

    print(x_train.shape, y_train.shape)


    # pickle_file = 'comments_pickle'
    # with open(pickle_file, 'rb') as f:
    #     save = pickle.load(f)
    #     x_train = save['X_train']
    #     y_train = save['y_train']
    #     x_valid = save['X_valid']
    #     y_valid = save['y_valid']
    #     x_test = save['X_test']
    #     y_test = save['y_test']
    #     embedding_matrix = save['embedding_matrix']
    #     del save  # hint to help gc free up memory

    model_1 = Model_GRU()
    model_1.train(x_train, y_train, x_valid, y_valid, embedding_matrix)
    test_pred_1 = model_1.model.predict(x_test)

    # model_2 = Model_CNN_Attention()
    model_2 = Model_GRU_Attention()
    model_2.train(x_train, y_train, x_valid, y_valid, embedding_matrix)
    test_pred_2 = model_2.model.predict(x_test)








