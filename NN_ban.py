import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras

def dataProcess():
    data = pd.read_excel('dataset.xlsx', header=1, sheet_name='10.6', usecols='F,G,R:W') ### 选择10.6版本数据，读取蓝色方、红色方、第一轮禁用的英雄。

    ## 为了后续构建队伍和英雄的字典而进行的预处理
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

    teamDict={'team':teamName}
    heroDict={'hero':heroName}

    teamPd = pd.DataFrame(teamDict)
    heroPd = pd.DataFrame(heroDict)

    ### 把panda类的数据转化为array类型的数据
    dataArray=np.array(data)

    ###把队伍和英雄转化为one-hot vector
    teamArray=np.array(pd.get_dummies(teamPd))
    heroArray=np.array(pd.get_dummies(heroPd))

    ### 构建词典
    dictTeam={}
    for i in range(17):
        dictTeam[teamName[i]]=teamArray[i]

    dictHero={}
    for i in range(len(heroName)):
        dictHero[heroName[i]]=heroArray[i]


    ### 构建训练时的数据集
    for i in dataArray:
        i[0]=dictTeam[i[0]]
        i[1]=dictTeam[i[1]]
        i[2]=dictHero[i[2]]
        i[3]=dictHero[i[3]]
        i[4]=dictHero[i[4]]
        i[5]=dictHero[i[5]]
        i[6]=dictHero[i[6]]
        i[7]=dictHero[i[7]]

    ### 划分训练集和测试集，并且完成数据的填入
    trainData=dataArray[:100]
    testData=dataArray[100:]

    trainDataX=[]
    trainDataY=[]
    testDataX=[]
    testDataY=[]


    for i in trainData:
        for j in range(6):
            trainDataX.append(np.hstack((i[0],i[1])))
            trainDataY.append(i[2+j])

        
    for i in testData:
        for j in range(6):
            testDataX.append(np.hstack((i[0],i[1])))
            testDataY.append(i[2+j])
        

    trainDataX=np.array(trainDataX)
    trainDataY=np.array(trainDataY)
    testDataX=np.array(testDataX)
    testDataY=np.array(testDataY)

    return trainDataX,trainDataY,testDataX,testDataY,heroArray,heroName,dictTeam


def NNModelTrain(trainDataX,trainDataY,testDataX,testDataY):
    ### 模型结构设置
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Dense(32,activation='relu'))
    model.add(tf.keras.layers.Dense(64,activation='relu'))

    model.add(tf.keras.layers.Dense(256,activation='relu'))
    model.add(tf.keras.layers.Dense(152, activation='softmax'))


    model.compile(optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['accuracy'])

    ### 模型的训练
    model.fit(trainDataX, trainDataY, epochs=20,batch_size=20,validation_data=(testDataX, testDataY))
    model.evaluate(testDataX, testDataY)

    return model


def NNModelPrediction(model,testDataX,heroArray,heroName,one_hot_vector):
    ### 从模型中得到结果
    testDataX[0]=np.array(one_hot_vector)
    predictions = model.predict(testDataX)
    score = tf.nn.softmax(predictions[0])

    print("HEROS TO BAN:")
    scorenew=list(score[:])
    for j in range(6):
        index = np.argmax(scorenew)
        for i in range(152):
            if(heroArray[i][index]==1):
                print(heroName[i])
        scorenew[index]=0
    
    return 


if __name__ == '__main__':

    ### 数据的预处理
    trainDataX,trainDataY,testDataX,testDataY, heroArray, heroName, dictTeam = dataProcess()

    ### 训练模型
    model = NNModelTrain(trainDataX,trainDataY,testDataX,testDataY)

    ### 得到结果
    teamBlue = 'ig'
    teamRed = 'fpx'

    print("===========START==========")
    print("TEAM BLUE:", teamBlue)
    print("TEAM RED:", teamRed)

    print("===========RESULT==========")

    one_hot_Blue = dictTeam[teamBlue]
    one_hot_Red = dictTeam[teamRed]
    
    one_hot_vector = np.hstack((one_hot_Blue,one_hot_Red))

    NNModelPrediction(model,testDataX,heroArray,heroName,one_hot_vector)



