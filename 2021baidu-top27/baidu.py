import json
import numpy as np
from bert4keras.backend import keras, K, batch_gather,search_layer
from bert4keras.layers import Loss
from bert4keras.layers import LayerNormalization
from bert4keras.tokenizers import Tokenizer
from bert4keras.models import build_transformer_model
from bert4keras.optimizers import Adam, extend_with_exponential_moving_average
from bert4keras.snippets import sequence_padding, DataGenerator
from bert4keras.snippets import open, to_array
from keras.layers import Input, Dense, Lambda, Reshape
from keras.models import Model
from tqdm import tqdm
import os
import tensorflow as tf
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
train=True  #如果是训练就选择True，如果是预测就选择False
maxlen = 220
batch_size =24
epoch=5
config_path = './chinese_roberta_wwm_ext_L-12_H-768_A-12/bert_config.json'
checkpoint_path = './chinese_roberta_wwm_ext_L-12_H-768_A-12/bert_model.ckpt'
dict_path = './chinese_roberta_wwm_ext_L-12_H-768_A-12/vocab.txt'
model_save = 'best_model_v3.weights'
# https://github.com/core-power/2021baidu-TOP27/blob/main/baidu.py
def load_data(filename):
    """加载数据
    单条格式：{'text': text, 'spo_list': [(s, p, o)]}
    """
    D = []
    with open(filename, encoding='utf-8') as f:
        for l in f:
            l = json.loads(l)
            D.append({
                'text': l['text'],
                'spo_list': [(spo['subject'], spo['predicate'], spo['object']['@value'])
                             for spo in l['spo_list']]
            })
    return D
def load_test_data(filename):
    """加载数据
    单条格式：{'text': text, 'spo_list': [(s, p, o)]}
    """
    D = []
    with open(filename, encoding='utf-8') as f:
        for l in f:
            l = json.loads(l)
            D.append({
                'text': l['text']
            })
    return D
# 解析为 {test: , spo_list:}
train_data = load_data('duie_train.json')
valid_data = load_data('duie_dev.json')
predicate2id, id2predicate = {}, {}

with open('duie_schema.json') as f:
    for l in f:
        l = json.loads(l)
        if l['predicate'] not in predicate2id:
            id2predicate[len(predicate2id)] = l['predicate']
            predicate2id[l['predicate']] = len(predicate2id)

# 建立分词器
tokenizer = Tokenizer(dict_path, do_lower_case=True)


def search(pattern, sequence):
    """从sequence中寻找子串pattern
    如果找到，返回第一个下标；否则返回-1。
    """
    n = len(pattern)
    for i in range(len(sequence)):
        if sequence[i:i + n] == pattern:
            return i
    return -1


class data_generator(DataGenerator):
    """数据生成器
    """
    def __iter__(self, random=False):
        batch_token_ids, batch_segment_ids = [], []
        batch_subject_labels, batch_subject_ids, batch_object_labels = [], [], []
        for is_end, d in self.sample(random):#随机取样
            token_ids, segment_ids = tokenizer.encode(d['text'], maxlen=maxlen)
            # 整理三元组 {s: [(o, p)]}
            spoes = {}
            for s, p, o in d['spo_list']:
                s = tokenizer.encode(s)[0][1:-1]
                p = predicate2id[p]
                o = tokenizer.encode(o)[0][1:-1]
                s_idx = search(s, token_ids)
                o_idx = search(o, token_ids)
                if s_idx != -1 and o_idx != -1:
                    s = (s_idx, s_idx + len(s) - 1) # s,e 为实体的起始和结束位置
                    o = (o_idx, o_idx + len(o) - 1, p) # (o_start, o_end, p)
                    if s not in spoes:
                        spoes[s] = []
                    spoes[s].append(o) #format {s: [(o, p)]} ex. {(0, 1): [(2, 3, 0)]}
            if spoes:
                # subject标签
                subject_labels = np.zeros((len(token_ids), 2))
                for s in spoes:
                    subject_labels[s[0], 0] = 1
                    subject_labels[s[1], 1] = 1
                # 随机选一个subject（这里没有实现错误！这就是想要的效果！！）
                start, end = np.array(list(spoes.keys())).T
                start = np.random.choice(start)
                end = np.random.choice(end[end >= start])
                subject_ids = (start, end)
                # 对应的object标签
                object_labels = np.zeros((len(token_ids), len(predicate2id), 2)) # 三维 token，微词的个数，起始标识[0,0]
                for o in spoes.get(subject_ids, []):
                    object_labels[o[0], o[2], 0] = 1
                    object_labels[o[1], o[2], 1] = 1
                # 构建batch
                batch_token_ids.append(token_ids)
                batch_segment_ids.append(segment_ids)
                batch_subject_labels.append(subject_labels) # 多个subject的开始和结束位置
                batch_subject_ids.append(subject_ids) # 一个subject的开始和结束位置 ?这个实验方式不太好？
                batch_object_labels.append(object_labels)
                if len(batch_token_ids) == self.batch_size or is_end:
                    batch_token_ids = sequence_padding(batch_token_ids)
                    batch_segment_ids = sequence_padding(batch_segment_ids)
                    batch_subject_labels = sequence_padding(batch_subject_labels)
                    batch_subject_ids = np.array(batch_subject_ids)# 随机选的起始id （s,e）
                    batch_object_labels = sequence_padding(batch_object_labels)
                    yield [
                        batch_token_ids, batch_segment_ids,
                        batch_subject_labels, batch_subject_ids,
                        batch_object_labels
                    ], None
                    batch_token_ids, batch_segment_ids = [], []
                    batch_subject_labels, batch_subject_ids, batch_object_labels = [], [], []


def extract_subject(inputs):
    """根据subject_ids从output中取出subject的向量表征
    """
    output, subject_ids = inputs
    start = batch_gather(output, subject_ids[:, :1])
    end = batch_gather(output, subject_ids[:, 1:])
    subject = K.concatenate([start, end], 2)
    return subject[:, 0]


# 补充输入
subject_labels = Input(shape=(None, 2), name='Subject-Labels')
subject_ids = Input(shape=(2,), name='Subject-Ids')
object_labels = Input(shape=(None, len(predicate2id), 2), name='Object-Labels')

# 加载预训练模型
bert = build_transformer_model(
    config_path=config_path,
    checkpoint_path=checkpoint_path,
    return_keras_model=False,
    model='roberta'
)

# ##### 1. 预测subject (s, e) 位置unit=2
output = Dense(
    units=2, activation='sigmoid', kernel_initializer=bert.initializer
)(bert.model.output)
subject_preds = Lambda(lambda x: x**2)(output)#这里为啥2次幂，不用sigmoid呢

subject_model = Model(bert.model.inputs, subject_preds) # subject预测线路

# #### 2. 传入subject，预测object
# 通过Conditional Layer Normalization将subject融入到object的预测中
output = bert.model.layers[-2].get_output_at(-1)  # 自己想为什么是-2而不是-1 # 一层transformer的输出值，理论上来说都可以作为句向量。但真正进行使用，选取倒数第二层来作为句向量的效果是最好的，因为Bert的最后一层的值太接近于训练任务目标，Bert前面几层transformer的可能还未充分的学习到语义特征 ，因此选择Bert倒数第二层作为句向量是比较合适的。
subject = Lambda(extract_subject)([output, subject_ids])
output = LayerNormalization(conditional=True)([output, subject])  # 合并bert的输出和subject的输出
output = Dense(
    units=len(predicate2id) * 2, # 谓词的个数*2 因为起始和结束
    activation='sigmoid',
    kernel_initializer=bert.initializer
)(output)
output = Lambda(lambda x: x**4)(output)#这里为啥4次幂，不用sigmoid呢
object_preds = Reshape((-1, len(predicate2id), 2))(output) # 谓词格式化，起始标识[0,0]

object_model = Model(bert.model.inputs + [subject_ids], object_preds) # object模型预测线路


class TotalLoss(Loss):
    """subject_loss与object_loss之和，都是二分类交叉熵
    """
    def compute_loss(self, inputs, mask=None):
        subject_labels, object_labels = inputs[:2]
        subject_preds, object_preds, _ = inputs[2:]
        if mask[4] is None:
            mask = 1.0
        else:
            mask = K.cast(mask[4], K.floatx())
        # sujuect部分loss
        subject_loss = K.binary_crossentropy(subject_labels, subject_preds)
        subject_loss = K.mean(subject_loss, 2)
        subject_loss = K.sum(subject_loss * mask) / K.sum(mask)
        # object部分loss
        object_loss = K.binary_crossentropy(object_labels, object_preds)
        object_loss = K.sum(K.mean(object_loss, 3), 2)
        object_loss = K.sum(object_loss * mask) / K.sum(mask)
        # 总的loss
        return subject_loss + object_loss

# 3. 增加loss计算 subject_loss + object_loss
subject_preds, object_preds = TotalLoss([2, 3])([
    subject_labels, object_labels, subject_preds, object_preds,
    bert.model.output
])

# 4. 训练模型 # 模型完整线路
train_model = Model(
    bert.model.inputs + [subject_labels, subject_ids, object_labels],
    [subject_preds, object_preds]
)

#AdamEMA = extend_with_exponential_moving_average(Adam, name='AdamEMA')
optimizer = Adam(learning_rate=1e-5)
train_model.compile(optimizer=optimizer)

def extract_spoes(text):
    """抽取输入text所包含的三元组
    """
    tokens = tokenizer.tokenize(text, maxlen=maxlen)
    mapping = tokenizer.rematch(text, tokens)
    token_ids, segment_ids = tokenizer.encode(text, maxlen=maxlen)
    token_ids, segment_ids = to_array([token_ids], [segment_ids])
    # 抽取subject
    subject_preds = subject_model.predict([token_ids, segment_ids])
    start = np.where(subject_preds[0, :, 0] > 0.5)[0]
    end = np.where(subject_preds[0, :, 1] > 0.4)[0]
    subjects = []
    for i in start:
        j = end[end >= i]
        if len(j) > 0:
            j = j[0]
            subjects.append((i, j))
    if subjects:
        spoes = []
        token_ids = np.repeat(token_ids, len(subjects), 0)
        segment_ids = np.repeat(segment_ids, len(subjects), 0)
        subjects = np.array(subjects)
        # 传入subject，抽取object和predicate
        object_preds = object_model.predict([token_ids, segment_ids, subjects])
        for subject, object_pred in zip(subjects, object_preds):
            start = np.where(object_pred[:, :, 0] > 0.5)
            end = np.where(object_pred[:, :, 1] > 0.4)
            for _start, predicate1 in zip(*start):
                for _end, predicate2 in zip(*end):
                    if _start <= _end and predicate1 == predicate2:
                        spoes.append(
                            ((mapping[subject[0]][0],
                              mapping[subject[1]][-1]), predicate1,
                             (mapping[_start][0], mapping[_end][-1]))
                        )
                        break
        return [(text[s[0]:s[1] + 1], id2predicate[p], text[o[0]:o[1] + 1])
                for s, p, o, in spoes]
    else:
        return []

class SPO(tuple):
    """用来存三元组的类
    表现跟tuple基本一致，只是重写了 __hash__ 和 __eq__ 方法，
    使得在判断两个三元组是否等价时容错性更好。
    """
    def __init__(self, spo):
        self.spox = (
            tuple(tokenizer.tokenize(spo[0])),
            spo[1],
            tuple(tokenizer.tokenize(spo[2])),
        )

    def __hash__(self):
        return self.spox.__hash__()

    def __eq__(self, spo):
        return self.spox == spo.spox

def evaluate(data):
    """评估函数，计算f1、precision、recall
    """
    X, Y, Z = 1e-10, 1e-10, 1e-10
    f = open('dev_pred.json', 'w', encoding='utf-8')
    pbar = tqdm()
    for d in data:
        R = set([SPO(spo) for spo in extract_spoes(d['text'])])
        T = set([SPO(spo) for spo in d['spo_list']])
        X += len(R & T)
        Y += len(R)
        Z += len(T)
        f1, precision, recall = 2 * X / (Y + Z), X / Y, X / Z
        pbar.update()
        pbar.set_description(
            'f1: %.5f, precision: %.5f, recall: %.5f' % (f1, precision, recall)
        )
        s = json.dumps({
            'text': d['text'],
            'spo_list': list(T),
            'spo_list_pred': list(R),
            'new': list(R - T),
            'lack': list(T - R),
        },
                       ensure_ascii=False,
                       indent=4)
        f.write(s + '\n')
    pbar.close()
    f.close()
    return f1, precision, recall

class Evaluator(keras.callbacks.Callback):
    """评估与保存
    """
    def __init__(self):
        self.best_val_f1 = 0.

    def on_epoch_end(self, epoch, logs=None):
        #optimizer.apply_ema_weights()
        f1, precision, recall = evaluate(valid_data)
        if f1 >= self.best_val_f1:
            self.best_val_f1 = f1
            train_model.save_weights(model_save)
        #optimizer.reset_old_weights()
        print(
            'f1: %.5f, precision: %.5f, recall: %.5f, best f1: %.5f\n' %
            (f1, precision, recall, self.best_val_f1)
        )

def pred(data):
    f = open('duie.json', 'w', encoding='utf-8')
    with open("../baidu_data/object_type.json", 'r', encoding='UTF-8') as x:
        dict1 = json.load(x)
    with open("../baidu_data/subject_type.json", 'r', encoding='UTF-8') as y:
        dict2 = json.load(y)
    with open("../baidu_data/inWork.json", 'r', encoding='UTF-8') as t:
        dict3 = json.load(t)
    for i in data:
        try:
            R = list(set([SPO(spo) for spo in extract_spoes(i['text'])]))
        except:R=[]
        x = []
        for j in R:
            type1=dict1.get(j[1],'-1')
            type2=dict2.get(j[1],'-1')
            inwork1=dict3.get(j[2],'-1')
            #inwork2=dict3.get(dict1.get(j[2],'-1'),'-1')
            if inwork1!='-1' :
                x.append({'subject':j[0],'predicate':j[1],'object':{'@value':j[2],'inWork':inwork1},'object_type':type1,'subject_type':type2})
            else :
                 x.append({'subject':j[0],'predicate':j[1],'object':{'@value':j[2]},'object_type':type1,'subject_type':type2})
            # elif inwork1=='-1' and inwork2!='-1':
            #      x.append({'subject':j[0],'predicate':j[1],'object':{'@value':j[2]},'object_type':{'@value':type1,'inWork':inwork2},'subject_type':type2})
            # elif inwork1=='-1' and inwork2=='-1':
            #      x.append({'subject':j[0],'predicate':j[1],'object':{'@value':j[2]},'object_type':{'@value':type1},'subject_type':type2})
        s = json.dumps({
            'text': i['text'],
            'spo_list': x,
        },
            ensure_ascii=False)
        f.write(s +'\n')
    f.close()
if __name__ == '__main__':

    if train == True:
        train_generator = data_generator(train_data, batch_size)
        dev_generator = data_generator(valid_data, batch_size)
        evaluator = Evaluator()
        #train_model.load_weights('best_model_v2.weights')
        train_model.fit(
            train_generator.forfit(),
            steps_per_epoch=len(train_generator),
            epochs=epoch,
            callbacks=[evaluator]
        )
        # train_model.load_weights('best_model_v2.weights')
        # train_model.fit(
        # dev_generator.forfit(),
        # steps_per_epoch=len(dev_generator),
        # epochs=20,
        # )
        # train_model.save_weights('best_model_v2.weights')
    else:
        train_model.load_weights(model_save)
        test_data = load_test_data('../baidu_data/duie_test2.json')
        pred(test_data)