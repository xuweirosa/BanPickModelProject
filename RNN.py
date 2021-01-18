import tensorflow as tf
import pandas as pd
import numpy as np
import os
import time
import random
import operator
from functools import reduce

def split_input_target(chunk):
    input_text = chunk[:-1]
    target_text = chunk[1:]
    return input_text, target_text

### data processing
def dataProcess(rounds):
    if(rounds==1): ## pick1 (blue)
        data = pd.read_excel('dataset.xlsx', header=1, sheet_name='10.6', usecols='F,G,R:X')
    if(rounds==2): ## pick2 (red)
        data = pd.read_excel('dataset.xlsx', header=1, sheet_name='10.6', usecols='F,G,R:Y')
    if(rounds==3): ## pick3 (red)
        data = pd.read_excel('dataset.xlsx', header=1, sheet_name='10.6', usecols='F,G,R:Z')
    if(rounds==4): ## pick4 (blue)
        data = pd.read_excel('dataset.xlsx', header=1, sheet_name='10.6', usecols='F,G,R:AA')
    if(rounds==5): ## pick5 (blue)
        data = pd.read_excel('dataset.xlsx', header=1, sheet_name='10.6', usecols='F,G,R:AB')
    if(rounds==6): ## pick6 (red)
        data = pd.read_excel('dataset.xlsx', header=1, sheet_name='10.6', usecols='F,G,R:AC')
    if(rounds==7): ## ban7 (red)
        data = pd.read_excel('dataset.xlsx', header=1, sheet_name='10.6', usecols='F,G,R:AD')
    if(rounds==8): ## ban8 (blue)
        data = pd.read_excel('dataset.xlsx', header=1, sheet_name='10.6', usecols='F,G,R:AE')
    if(rounds==9): ## ban9 (red)
        data = pd.read_excel('dataset.xlsx', header=1, sheet_name='10.6', usecols='F,G,R:AF')
    if(rounds==10): ## ban10 (blue)
        data = pd.read_excel('dataset.xlsx', header=1, sheet_name='10.6', usecols='F,G,R:AG')
    if(rounds==11): ## pick7 (red)
        data = pd.read_excel('dataset.xlsx', header=1, sheet_name='10.6', usecols='F,G,R:AH')
    if(rounds==12): ## pick8 (blue)
        data = pd.read_excel('dataset.xlsx', header=1, sheet_name='10.6', usecols='F,G,R:AI')
    if(rounds==13): ## pick9 (blue)
        data = pd.read_excel('dataset.xlsx', header=1, sheet_name='10.6', usecols='F,G,R:AJ')
    if(rounds==14): ## pick10 (red)
        data = pd.read_excel('dataset.xlsx', header=1, sheet_name='10.6', usecols='F,G,R:AK')

    dataArray=np.array(data)
    
    sentenceLength = len(dataArray[0])
    trainSentenceLength = sentenceLength - 1

    dataArray = dataArray.astype('str')
    tensor_a=tf.convert_to_tensor(dataArray)

    ### build the vocabulary
    teamName=['fpx','ig','rng','sn','jdg','lgd','rw','tes','edg','omg','we','blg','lng','vg','dmo','v5','es']
    heroName=['安妮','奥拉夫','加里奥','崔斯特','赵信','厄加特','乐芙兰','弗拉基米尔','费德提克','凯尔','易','阿利斯塔','瑞兹',
            '塞恩','希维尔','索拉卡','提莫','崔丝塔娜','沃里克','努努和威朗普','厄运小姐','艾希','泰达米尔','贾克斯','莫甘娜','基兰',
            '辛吉德','伊芙琳','图奇','卡尔萨斯','科加斯','阿木木','拉莫斯','艾尼维亚','萨科','蒙多医生','娑娜','卡萨丁','艾瑞莉娅',
            '迦娜','普朗克','库奇','卡尔玛','塔里克','维迦','特朗德尔','斯维因','凯特琳','布里茨','墨菲特','卡特琳娜','魔腾',
            '茂凯','雷克顿','嘉文四世','伊莉丝','奥莉安娜','孙悟空','布兰德','李青','薇恩','兰博','卡西奥佩娅','斯卡纳','黑默丁格',
            '内瑟斯','奈德丽','乌迪尔','波比','古拉加斯','潘森','伊泽瑞尔','莫德凯撒','约里克','阿卡丽','凯南','盖伦','蕾欧娜',
            '玛尔扎哈','泰隆','锐雯','克格莫','慎','拉克丝','泽拉斯','希瓦娜','阿狸','格雷福斯','菲兹','沃利贝尔','雷恩加尔',
            '韦鲁斯','诺提勒斯','维克托','瑟庄妮','菲奥娜','吉格斯','璐璐','德莱文','赫卡里姆','卡兹克','德莱厄斯','杰斯','丽桑卓',
            '黛安娜','奎因','辛德拉','奥瑞利安索尔','凯隐','佐伊','婕拉','卡莎','萨勒芬妮','纳尔','扎克','亚索','维克兹',
            '塔莉垭','卡蜜尔','布隆','烬','千珏','金克丝','塔姆','赛娜','卢锡安','劫','克烈','艾克','奇亚娜',
            '蔚','亚托克斯','娜美','阿兹尔','悠米','莎弥拉','锤石','俄洛伊','雷克塞','艾翁','卡莉丝塔','巴德','洛',
            '霞','奥恩','塞拉斯','妮蔻','厄斐琉斯','派克','永恩','瑟提','莉莉娅']

    vocabName=teamName+heroName
    vocab = sorted(set(vocabName))

    dictVocab = {}
    for i in range(len(vocab)):
        dictVocab[vocab[i]]=i

    keyList = []
    for item in dictVocab.keys():
        keyList.append(item)    

    ### translate the str element to int type
    for i in range(len(dataArray)):
        for j in range(len(dataArray[i])):
            if(dataArray[i][j]=='空ban'):
                dataArray[i][j]=random.randint(17,168)
                print(dataArray[i][j])
            else:
                dataArray[i][j]=dictVocab[dataArray[i][j]]
    
    one_dim_dataArray=dataArray.flatten()
    dataArray = one_dim_dataArray.astype('int')
    one_dim_dataArray = one_dim_dataArray.astype('int')

    text_as_int=one_dim_dataArray

    # 设定每个输入句子长度的最大值
    seq_length = trainSentenceLength
    examples_per_epoch = len(text_as_int)//seq_length

    # build training data and target data
    char_dataset = tf.data.Dataset.from_tensor_slices(text_as_int)
    sequences = char_dataset.batch(seq_length+1, drop_remainder=True)
    dataset = sequences.map(split_input_target)

    return dictVocab,dataset,keyList


### use the function to know which position one hero often goes to
def learnPosition(dictVocab):
    dataPos = pd.read_excel('dataset.xlsx', header=1, sheet_name='10.6', usecols='AL:AU') ##pos

    posArray = np.zeros((169,5))
    dataPosArray = np.array(dataPos)
    dataPosArray = dataPosArray.astype('str')

    for i in dataPosArray:
        for j in range(5):
            index = dictVocab[i[j]]
            posArray[index][j]+=1
        for j in range(5):
            index = dictVocab[i[j+5]]
            posArray[index][j]+=1

    return posArray

def build_model(vocab_size, embedding_dim, rnn_units, batch_size):

    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Embedding(vocab_size, embedding_dim,batch_input_shape=[batch_size, None]))
    model.add(tf.keras.layers.GRU(rnn_units,return_sequences=True,stateful=True,recurrent_initializer='glorot_uniform'),)
    model.add(tf.keras.layers.Dense(vocab_size))

    return model

def loss(labels, logits):
    return tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)

def RNN_Model(dataset,dictVocab):
    # 批大小
    BATCH_SIZE = 64
    BUFFER_SIZE = 10000
    dataset = dataset.batch(BATCH_SIZE, drop_remainder=True)
    # 词集的长度
    vocab_size = len(dictVocab)
    # 嵌入的维度
    embedding_dim = 256
    # RNN 的单元数量
    rnn_units = 1024

    model = build_model(vocab_size = len(dictVocab),embedding_dim=embedding_dim,rnn_units=rnn_units,batch_size=BATCH_SIZE)
    model.summary()
    model.compile(optimizer='adam', loss=loss)

    # 检查点保存至的目录
    checkpoint_dir = './training_checkpoints'
    # 检查点的文件名
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")
    checkpoint_callback=tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_prefix,save_weights_only=True)

    EPOCHS=10
    history = model.fit(dataset, epochs=EPOCHS, callbacks=[checkpoint_callback])
    tf.train.latest_checkpoint(checkpoint_dir)

    model = build_model(vocab_size, embedding_dim, rnn_units, batch_size=1)
    model.load_weights(tf.train.latest_checkpoint(checkpoint_dir))
    model.build(tf.TensorShape([1, None]))

    model.summary()

    return model


def generate_text(model, start_string, temperature = 0.3):
    # 要生成的字符个数
    num_generate = 1
    # 将起始字符串转换为数字（向量化）
    input_eval = tf.expand_dims(start_string, 0)
    # 空字符串用于存储结果
    text_generated = []
    # 低温度会生成更可预测的文本
    # 较高温度会生成更令人惊讶的文本
    # 可以通过试验以找到最好的设定
    # 这里批大小为 1
    model.reset_states()
    for i in range(num_generate):
        predictions = model(input_eval)
        # 删除批次的维度
        predictions = tf.squeeze(predictions, 0)
        # 用分类分布预测模型返回的字符
        predictions = predictions / temperature
        predicted_id = tf.random.categorical(predictions, num_samples=1)[-1,0].numpy()
        # 把预测字符和前面的隐藏状态一起传递给模型作为下一个输入
        input_eval = tf.expand_dims([predicted_id], 0)
        text_generated.append(predicted_id)

    return (text_generated)



if __name__ == '__main__':

    dictVocab, dataset, keyList = dataProcess(rounds=1)  

    posArray = learnPosition(dictVocab)

    model = RNN_Model(dataset,dictVocab)

    a = [5,4,116,43,162,50,155,147] ### six bans from naive bayes

    pos_blue = []
    pos_red = []

    for i in range(14):
        a_tensor= tf.convert_to_tensor(a)
        dictVocab, dataset, keyList = dataProcess(rounds=i+1)
        model = RNN_Model(dataset,dictVocab)

        if(i==0 or i==3 or i==4 or i==11 or i==12): ## pick blue
            s = int(generate_text(model, start_string=a, temperature=0.18)[-1])
            while(s in a or np.argmax(posArray[s]) in pos_blue or s<=16):
                s = int(generate_text(model, start_string=a, temperature=0.18)[-1])
                continue
            a.append(s)
            pos_blue.append(np.argmax(posArray[s]))
            print(keyList[s])
            
        
        if(i==1 or i==2 or i==5 or i==10 or i==13): ## pick red
            s = int(generate_text(model, start_string=a, temperature=0.18)[-1])
            while(s in a or np.argmax(posArray[s]) in pos_red or s<=16):
                s = int(generate_text(model, start_string=a, temperature=0.18)[-1])
                continue
            a.append(s)
            pos_red.append(np.argmax(posArray[s]))
            print(keyList[s])
            

        if(i==7 or i==9): ## ban blue
            s = int(generate_text(model, start_string=a, temperature=0.18)[-1])
            while(s in a or np.argmax(posArray[s]) in pos_red or s<=16):
                s = int(generate_text(model, start_string=a, temperature=0.18)[-1])
                continue
            a.append(s)
            

        if(i==6 or i==8): ## ban red
            s = int(generate_text(model, start_string=a, temperature=0.18)[-1])
            while(s in a or np.argmax(posArray[s]) in pos_blue or s<=16):
                s = int(generate_text(model, start_string=a, temperature=0.18)[-1])
                continue
            a.append(s)
        
        if(i==13):
            break
            
    
    for item in a:
        print(keyList[item])


