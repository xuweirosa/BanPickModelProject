import pandas as pd
import numpy as np

def dataProcess():
    data = pd.read_excel('dataset.xlsx', header=1, sheet_name='10.6', usecols='F,G,R:W')
    dataArray=np.array(data)

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

    ### 构建字典【名称作为key，整数作为value】
    dictVocab={}
    for i in range(len(heroName)):
        dictVocab[heroName[i]]=i

    return dataArray,heroName,teamName


def NaiveBayes(dataArray,heroName,teamName):
    # totalMatch=len(dataArray)# 数据集中比赛的总场次

    ban_times={} # 某个英雄被ban的次数，以字典存储
    for i in range(len(heroName)):
        ban_times[heroName[i]]=1
    for i in heroName:
        for j in dataArray:
            if(i in j):
                ban_times[i]+=1

    ban_blue_times={} # 某个英雄在蓝色方主动ban的次数
    for i in range(len(heroName)):
        ban_blue_times[heroName[i]]=1
    for i in heroName:
        for j in dataArray:
            if(i==j[2]or i==j[4] or i==j[6]): # 在蓝色方被主动ban掉
                ban_blue_times[i]+=1
            


    ban_red_times={} # 某个英雄在红色方主动ban的次数
    for i in range(len(heroName)):
        ban_red_times[heroName[i]]=1
    for i in heroName:
        for j in dataArray:
            if(i==j[3]or i==j[5] or i==j[7]): # 在蓝色方被主动ban掉
                ban_red_times[i]+=1
            


    ban_positive_team_times=[] #储存所有队伍的英雄主动ban次数，每个元素都是字典，存储着某个队伍的英雄主动ban情况。
    ban_negative_team_times=[] #储存所有队伍的英雄被动ban次数，每个元素都是字典，存储着某个队伍的英雄被ban情况。

    for i in range(len(teamName)):
        dict_1 = {} #positive
        dict_2 = {} #negative
        for s in range(len(heroName)):
            dict_1[heroName[s]]=1
            dict_2[heroName[s]]=1
            
        for j in heroName:
            for k in dataArray:
                
                if(teamName[i]==k[0]): ##蓝色方，主动ban位为0，2，4，被动ban位为1，3，5
                    if(j==k[2]or j==k[4] or j==k[6]): #positive
                        dict_1[j]+=1
                    if(j==k[3] or j==k[5] or j==k[7]):#negative
                        dict_2[j]+=1
                
                if(teamName[i]==k[1]): ##红色方，主动ban维为1，3，5，被动ban位为0，2，4
                    if(j==k[3]or j==k[5] or j==k[7]): # positive
                        dict_1[j]+=1
                    if(j==k[2] or j==k[4] or j==k[6]): #negative
                        dict_2[j]+=1
                

        ban_positive_team_times.append(dict_1)
        ban_negative_team_times.append(dict_2)

    return ban_negative_team_times,ban_positive_team_times,ban_red_times,ban_blue_times


def makePrediction(teamBlue,teamRed,standRed,already_ban):

    if(standRed==1): ## We make prediction for teamRed
        posBanRed=[]
        ## calculate the probability using Naive Bayes
        for i in heroName:
            pos_we_positive_ban=ban_positive_team_times[0][i]
            pos_they_negative_ban=ban_negative_team_times[1][i]
            pos_red_bans=ban_red_times[i]
            pos_final = pos_we_positive_ban * pos_they_negative_ban * pos_red_bans
            posBanRed.append(pos_final)

        
        ## choose the maximum probability and make predictions
        while(True):
            index = np.argmax(posBanRed)
            if(heroName[index] in already_ban):
                posBanRed[index]=0
                continue
            else:
                already_ban.append(heroName[index])
                print(teamRed," ban: ",heroName[index])
                break

    
    else: ## We make prediction for teamBlue
        posBanBlue=[]

        ## calculate the probability for teamBlue
        for i in heroName:
            pos_we_positive_ban=ban_positive_team_times[1][i]
            pos_they_negative_ban=ban_negative_team_times[0][i]
            pos_blue_bans=ban_blue_times[i]
            pos_final = pos_we_positive_ban * pos_they_negative_ban * pos_blue_bans
            posBanBlue.append(pos_final)
            
        while(True):
            index = np.argmax(posBanBlue)
            if(heroName[index] in already_ban):
                posBanBlue[index]=0
                continue
            else:
                already_ban.append(heroName[index])
                print(teamBlue," ban: ",heroName[index])

                break

    return already_ban



if __name__ == '__main__':

    dataArray,heroName,teamName = dataProcess()
    ban_negative_team_times,ban_positive_team_times,ban_red_times,ban_blue_times = NaiveBayes(dataArray,heroName,teamName)

    teamBlue = 'ig'
    teamRed = 'fpx'

    already_ban = []

    ###makePrediction(teamBlue,teamRed,standRed=0,already_ban=already_ban)

    print("===========START==========")
    print("TEAM BLUE:", teamBlue)
    print("TEAM RED:", teamRed)

    print("===========RESULT==========")


    for times in range(6):
        ## make prediction for teamBlue
        if(times%2==0):
            already_ban = makePrediction(teamBlue,teamRed,standRed=0,already_ban=already_ban)
        ## make predction for teamRed
        else:
            already_ban = makePrediction(teamBlue,teamRed,standRed=1,already_ban=already_ban)



