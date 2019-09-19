# -*- coding: utf-8 -*-
import json
import time
import tensorflow as tf
print(tf.__version__)

config = {
    'sequence_length': 300,    # 文本长度，当文本大于该长度则截断
    'num_classes': 20,         # 文本分类数
    'vocab_size': 6000,        # 字典大小
    'embedding_size': 300,     # embedding词向量维度
    'device': '/cpu:0',        # 设置device
    'batch_size':256,         # batch大小
    'num_epochs': 200,           # epoch数目
    'evaluate_every': 100,     # 每隔多少步打印一次验证集结果
    'checkpoint_every': 100,   # 每隔多少步保存一次模型
    'num_checkpoints': 5,      # 最多保存模型的个数
    'allow_soft_placement': True,   # 是否允许程序自动选择备用device
    'log_device_placement': False,  # 是否允许在终端打印日志文件
    'train_test_dev_rate': [0.8, 0.1, 0.1],   # 训练集，测试集，验证集比例
    'data_path': './data/result.txt',    # 数据路径  格式：标签\t文本
    'learning_rate': 0.005,             # 学习率
    'vocab_path': './vocabs',           # 保存词典路径
}
def load_json(json_file_path):
    with open(json_file_path, 'r', encoding='utf-8') as f:
        return json.loads(f.read(), encoding='utf-8')
def padding(X, y, config, word_to_index, label_to_index):
    sequence_length = config['sequence_length']
    num_classes = config['num_classes']
    input_x = []
    for line in X:
        temp = []
        for item in list(line):
            temp.append(word_to_index.get(item, 0))
        input_x.append(temp[:sequence_length] + [0] * (sequence_length - len(temp)))
    if not y:
        return input_x

    input_y = []
    for item in y:
        temp = [0] * num_classes
        temp[label_to_index[item]] = 1
        input_y.append(temp)
    return input_x, input_y
class Predict():
    def __init__(self, config, model_path='./runs/1568858850/checkpoints/model.bin-1300', word_to_index='./vocabs/word_to_index.json',
                 index_to_label='./vocabs/index_to_label.json'):
        self.word_to_index = load_json(word_to_index)
        self.index_to_label = load_json(index_to_label)

        graph = tf.Graph()
        with graph.as_default():
            session_conf = tf.ConfigProto(
                allow_soft_placement=config['allow_soft_placement'],
                log_device_placement=config['log_device_placement'])
            self.sess = tf.Session(config=session_conf)
            with self.sess.as_default():
                # Load the saved meta graph and restore variables
                saver = tf.train.import_meta_graph("{}.meta".format(model_path))

                saver.restore(self.sess, model_path)
                # Get the placeholders from the graph by name
                self.input_x = graph.get_operation_by_name("input_x").outputs[0]
                self.dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]
                # Tensors we want to evaluate
                self.predictions = graph.get_operation_by_name("output/predictions").outputs[0]


    def predict(self, list_str):
        input_x = padding(list_str, None, config, self.word_to_index, None)
        feed_dict = {
            self.input_x: input_x,
            self.dropout_keep_prob: 1.0
        }
        predictions = self.sess.run(self.predictions, feed_dict=feed_dict)
        # print(predictions)
        return [self.index_to_label[str(idx)] for idx in predictions]

if __name__ == '__main__':
    prediction = Predict(config)
    start=time.time()
    list2=[]
    result = prediction.predict(['	一场失利造就两大英雄 戈塔特：我会成为太阳领袖新浪体育讯北京时间3月29日消息，据《亚利桑那共和报》报道，周日比赛虽然输给小牛，但我们也看到了太阳可喜的变化。首发阵容中，球队主帅阿尔文-金特里一下换了两名队员，马尔钦-戈塔特和摇摆人贾里德-杜德利被委以重任。而事实上，两人的表现也没有辜负金特里的期望。杜德利在对阵小牛的比赛中，顶替文斯-卡特出任球队先发后卫。那场比赛，他共出战38分钟，得到20分，5个篮板，5次助攻和3次抢断的数据，表现十分抢眼。而相比较杜德利，另外一位晋升首发的戈塔特则更令人眼前一亮。'])
    print(result)
