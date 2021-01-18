import pandas as pd
import numpy as np
import tensorflow.keras as keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Activation, Embedding, Masking, Dropout, Conv1D, MaxPooling1D, Reshape
from tensorflow.keras.models import load_model
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
from scipy import interpolate

### 模型一：开局胜率
# 训练参数
predict_step = 1
team_num = 2
cnn_output_dim = 64
kernel_size = 13
pool_size = 2
hidden_size = 256
epochs = 20
batch_size = 10
model_saved_path = "/Users/apple/Desktop/proj/ml/model/"
model_name = 'CNN'
hero_id_max = 152
team_id_max = 17
team_max = team_id_max + hero_id_max * 5

# 数据预处理
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

data = pd.read_excel('dataset.xlsx', header=1, sheet_name='10.6', usecols='F,G,AL:AV')
dataArray=np.array(data)
dataArray0 = np.array(data)

teamArray=np.array(pd.get_dummies(teamPd))
heroArray=np.array(pd.get_dummies(heroPd))

dictTeam={}
for i in range(len(teamName)):
    dictTeam[teamName[i]]=teamArray[i]

dictHero={}
for i in range(len(heroName)):
    dictHero[heroName[i]]=heroArray[i]

for i in dataArray:
    i[0]=dictTeam[i[0]]
    i[1]=dictTeam[i[1]]
    for j in range(2, 12):
        i[j] = dictHero[i[j]]

trainData = dataArray[:100]
testData = dataArray[100:110]
validateData = dataArray[110:]

trainDataX = []
trainDataY = []
testDataX = []
testDataY = []
validateDataX = []
validateDataY = []

for i in trainData:
    tmp1 = i[0]
    for j in range(2, 7):
        tmp1 = np.hstack((tmp1, i[j]))
    tmp2 = i[1]
    for j in range(7, 12):
        tmp2 = np.hstack((tmp2, i[j]))
    trainDataX.append(np.vstack((tmp1, tmp2)))
    trainDataY.append(i[12])

for i in testData:
    tmp1 = i[0]
    for j in range(2, 7):
        tmp1 = np.hstack((tmp1, i[j]))
    tmp2 = i[1]
    for j in range(7, 12):
        tmp2 = np.hstack((tmp2, i[j]))
    testDataX.append(np.vstack((tmp1, tmp2)))
    testDataY.append(i[12])

for i in validateData:
    tmp1 = i[0]
    for j in range(2, 7):
        tmp1 = np.hstack((tmp1, i[j]))
    tmp2 = i[1]
    for j in range(7, 12):
        tmp2 = np.hstack((tmp2, i[j]))
    validateDataX.append(np.vstack((tmp1, tmp2)))
    validateDataY.append(i[12])

train_x = np.array(trainDataX).reshape(len(trainDataX), team_num, team_max)
train_y = np.array(trainDataY).reshape(len(trainDataY), 1)
test_x = np.array(testDataX).reshape(len(testDataX), team_num, team_max)
test_y = np.array(testDataY).reshape(len(testDataY), 1)
validate_x = np.array(validateDataX).reshape(len(validateDataX), team_num, team_max)
validate_y = np.array(validateDataY).reshape(len(validateDataY), 1)

def training():
    # 纯CNN 模型
    model = Sequential()
    model.add(Conv1D(cnn_output_dim,kernel_size,padding='same',activation='relu',input_shape=(team_num,team_max)))
    model.add(MaxPooling1D(pool_size=pool_size,data_format='channels_first'))
    model.add(Reshape((int(team_num*cnn_output_dim/pool_size),), input_shape=(team_num,int(cnn_output_dim/pool_size))))
    model.add(Dropout(0.2))
    model.add(Dense((10),input_shape=(team_num,cnn_output_dim/pool_size)))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))
    model.compile(loss='mse',optimizer='adam',metrics=['accuracy'])

    # 纯LSTM 模型
    #model = Sequential()
    #model.add(LSTM(hidden_size, input_shape=(team_num,team_max), return_sequences=False))
    #model.add(Dropout(0.2))
    #model.add(Dense(10))
    #model.add(Dropout(0.2))
    #model.add(Dense(1))
    #model.add(Activation('sigmoid'))
    #model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])

    # CNN + LSTM 模型
    #model = Sequential()
    #model.add(Conv1D(cnn_output_dim,kernel_size,padding='same', input_shape=(team_num,team_max)))
    #model.add(MaxPooling1D(pool_size=pool_size,data_format='channels_first'))
    #model.add(LSTM(hidden_size, input_shape=(team_num,(cnn_output_dim/pool_size)), return_sequences=False))
    #model.add(Dropout(0.2))
    #model.add(Dense(10))
    #model.add(Dropout(0.2))
    #model.add(Dense(1))
    #model.add(Activation('sigmoid'))
    #model.compile(loss='mse',optimizer='adam',metrics=['accuracy'])

    callbacks = [keras.callbacks.EarlyStopping(monitor='val_loss', patience=2, verbose=0, mode='min'),
                     keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=1, verbose=0, mode='min',
                                                       epsilon=0.0001, cooldown=0, min_lr=0)]
    model.fit(train_x, train_y, batch_size=batch_size, epochs=epochs, shuffle=True,
                         validation_data=(validate_x, validate_y), callbacks=callbacks)
    model.save(model_saved_path + model_name + '.h5')

def testing():
    keras.backend.clear_session()  # 计算图清空，防止越来越慢
    model = load_model(model_saved_path + model_name + '.h5')

    out0 = model.predict(test_x)
    correct_num = 0
    for i in range(len(out0)):
        if out0[i] < 0.5:
            temp_result = 0.0
        else:
            temp_result = 1.0
        if temp_result == test_y[i]:
            correct_num += 1
    print('测试集准确率：', float(correct_num) / len(test_x))

    out1 = model.predict(train_x)
    correct_num = 0
    for i in range(len(out1)):
        if out1[i] < 0.5:
            temp_result = 0.0
        else:
            temp_result = 1.0
        if temp_result == train_y[i]:
            correct_num += 1
    print('训练集准确率：', float(correct_num) / len(train_x))

    out2 = model.predict(validate_x)
    correct_num = 0
    for i in range(len(out2)):
        if out2[i] < 0.5:
            temp_result = 0.0
        else:
            temp_result = 1.0
        if temp_result == validate_y[i]:
            correct_num += 1
    print('验证集准确率：', float(correct_num) / len(validate_x))

    t0 = 0.55
    for i in range(3):
        tag_line = t0 + 0.05 * i
        correct_num = 0
        compare_num = 0
        for i in range(len(out1)):
            if out1[i] > tag_line or 1-out1[i] > tag_line:
                compare_num += 1
                if out1[i] < 0.5:
                    temp_result = 0.0
                else:
                    temp_result = 1.0
                if temp_result == train_y[i]:
                    correct_num += 1
        if compare_num != 0:
            print('训练集,预测胜率在' + str(tag_line) + '以上的准确率：', float(correct_num) / compare_num,
                  ' (' + str(correct_num) + '/' + str(compare_num) + ')')
        else:
            print('训练集,预测胜率在' + str(tag_line) + '以上的准确率：', '0.0',
                  ' (' + str(correct_num) + '/' + str(compare_num) + ')')

    for i in range(len(out1)):
        tag_line = 0.65
        if out1[i] > tag_line:
            tmp1 = [dataArray0[i][0]]
            for j in range(2, 7):
                tmp1.append(dataArray0[i][j])
            tmp2 = [dataArray0[i][1]]
            for j in range(7, 12):
                tmp2.append(dataArray0[i][j])
            print('蓝色方获胜', tmp1, '\n', tmp2, '预测：', out1[i][0], '实际：', train_y[i][0])
        elif out1[i] < 1 - tag_line:
            tmp1 = [dataArray0[i][0]]
            for j in range(2, 7):
                tmp1.append(dataArray0[i][j])
            tmp2 = [dataArray0[i][1]]
            for j in range(7, 12):
                tmp2.append(dataArray0[i][j])
            print('红色方获胜', tmp1, '\n', tmp2, '预测：', out1[i][0], '实际：', train_y[i][0])


# main
#training()
testing()

### 模型二：实时胜率
df = pd.read_excel('dataset.xlsx', header=1, sheet_name='10.6', usecols='AV:BG')
df_model = df[['isWinBlue', 'killsBlue', 'economyBlue', 'bigDragonBlue', 'smallDragonBlue', 'towerBlue',
               'killsRed', 'economyRed', 'bigDragonRed', 'smallDragonRed', 'towerRed']]
x = df_model.drop('isWinBlue', axis = 1)
y = df_model['isWinBlue']
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.1, stratify=y, random_state=0, shuffle=True)

#  参数
parameters= {
    'splitter': ('best', 'random'),
    'criterion': ('gini', 'entropy'),
    'max_depth': [*range(1, 20, 2)],
}

#  建立模型
clf = DecisionTreeClassifier(random_state=0)
GS = GridSearchCV(clf, parameters, cv=10)
GS.fit(X_train, y_train)

GridSearchCV(cv=10, estimator=DecisionTreeClassifier(random_state=0),
             param_grid={'criterion': ('gini', 'entropy'),
                         'max_depth': [1, 3, 5, 7, 9, 11, 13, 15, 17, 19],
                         'splitter': ('best', 'random')})

# 输出最佳得分 
print("best score: ", GS.best_score_)
print("best param: ", GS.best_params_)


# 最佳模型
best_clf = DecisionTreeClassifier(criterion="gini", max_depth=7, splitter="best")
best_clf.fit(X_train,y_train)
print("score:", best_clf.score(X_test,y_test))

# 预测
x = [['ig', '亚托克斯', '奇亚娜', '兰博', '赛娜', '布隆', 'fpx', '奥恩', '嘉文四世', '瑞兹', '厄运小姐', '蕾欧娜'],
['ig', '亚托克斯', '奇亚娜', '奥莉安娜', '韦鲁斯', '布隆', 'fpx', '菲奥娜', '嘉文四世', '莫德凯撒', '厄运小姐', '锤石'],
['ig', '奥恩', '奇亚娜', '奥莉安娜', '韦鲁斯', '布隆', 'fpx', '莫德凯撒', '嘉文四世', '黛安娜', '厄运小姐', '蕾欧娜']]

for i in x:
    i[0] = dictTeam[i[0]]
    i[6] = dictTeam[i[6]]
    for j in range(1, 6):
        i[j] = dictHero[i[j]]
    for j in range(7, 12):
        i[j] = dictHero[i[j]]

test = []
for i in x:
    tmp1 = i[0]
    for j in range(1, 6):
        tmp1 = np.hstack((tmp1, i[j]))
    tmp2 = i[6]
    for j in range(7, 12):
        tmp2 = np.hstack((tmp2, i[j]))
    test.append(np.vstack((tmp1, tmp2)))

test = np.array(test).reshape(len(test), team_num, team_max)

keras.backend.clear_session()  # 计算图清空，防止越来越慢
model = load_model(model_saved_path + model_name + '.h5')

begin = model.predict(test)

df = pd.read_excel('dataset.xlsx', header=1, sheet_name='Sheet3', usecols='A:K')
X_test = df.drop('time', axis=1)
t = df['time']

out = best_clf.predict_proba(X_test)

blue1 = [begin[0][0]*100]
blue2 = [begin[1][0]*100]
blue3 = [begin[2][0]*100]
red1 = [(1-begin[0][0])*100]
red2 = [(1-begin[1][0])*100]
red3 = [(1-begin[2][0])*100]
t1 = []
t2 = []
t3 = []
t1.append(0)
t2.append(0)
t3.append(0)
for i in range(15):
    t1.append(t[i])
    p = 1/14
    tmp1 = out[i][0]*(i*p) + begin[0][0]*(1-i*p)
    tmp2 = out[i][1]*(i*p) + (1-begin[0][0])*(1-i*p)
    s = tmp1 + tmp2
    tmp1 = tmp1 / s * 100
    tmp2 = tmp2 / s * 100
    blue1.append(tmp1)
    red1.append(tmp2)


for i in range(15, 27):
    t2.append(t[i])
    p = 1 / 11
    j = i-15
    tmp1 = out[i][0] * (j * p) + begin[1][0] * (1 - j * p)
    tmp2 = out[i][1] * (j * p) + (1 - begin[1][0]) * (1 - j * p)
    s = tmp1 + tmp2
    tmp1 = tmp1 / s * 100
    tmp2 = tmp2 / s * 100
    blue2.append(tmp1)
    red2.append(tmp2)

for i in range(27, 36):
    t3.append(t[i])
    p = 1 / 8
    j = i -27
    tmp1 = out[i][0] * (j * p) + begin[2][0] * (1 - j * p)
    tmp2 = out[i][1] * (j * p) + (1 - begin[2][0]) * (1 - j * p)
    s = tmp1 + tmp2
    tmp1 = tmp1 / s * 100
    tmp2 = tmp2 / s * 100
    blue3.append(tmp1)
    red3.append(tmp2)

# 直方图
xnew1 = np.arange(0, t1[len(t1)-1], 0.1)
func = interpolate.interp1d(t1, blue1, kind='cubic')
ynew1 = func(xnew1)
func = interpolate.interp1d(t1, red1, kind='cubic')
ynew2 = func(xnew1)

xnew2 = np.arange(0, t2[len(t2)-1], 0.1)
func = interpolate.interp1d(t2, blue2, kind='cubic')
ynew21 = func(xnew2)
func = interpolate.interp1d(t2, red2, kind='cubic')
ynew22 = func(xnew2)

xnew3 = np.arange(0, t3[len(t3)-1], 0.1)
func = interpolate.interp1d(t3, blue3, kind='cubic')
ynew31 = func(xnew3)
func = interpolate.interp1d(t3, red3, kind='cubic')
ynew32 = func(xnew3)

plt.subplot(3, 1, 1)
plt.plot(xnew1, ynew1, label="ig")
plt.plot(xnew1, ynew2, color='red', linestyle='-.', label="fpx")
plt.title("first game")
plt.xlabel("t")
plt.tick_params(axis='both')
plt.xlim(xmin=0)
plt.ylim([0, 100])
plt.legend()

plt.subplot(3, 1, 2)
plt.plot(xnew2, ynew21, label="ig")
plt.plot(xnew2, ynew22, color='red', linestyle='-.', label="fpx")
plt.title("second game")
plt.xlabel("t")
plt.tick_params(axis='both')
plt.xlim(xmin=0)
plt.ylim([0, 100])
plt.legend()


plt.subplot(3, 1, 3)
plt.plot(xnew3, ynew31, label="ig")
plt.plot(xnew3, ynew32, color='red', linestyle='-.', label="fpx")
plt.title("third game")
plt.xlabel("t")
plt.tick_params(axis='both')
plt.xlim(xmin=0)
plt.ylim([0, 100])
plt.legend()
plt.tight_layout()
plt.show()