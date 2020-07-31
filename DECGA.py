#遗传算法（Genetic Algorithm）实现
import math
import numpy as np
import pandas as pd
import random
import copy
from matplotlib import pyplot as plt
#全局变量
pop_size = 1000#种群个体数
chromosome_len = 17#染色体长度
pop = []#种群
x = []#自变量
fit_value = []#适度值
pc = 0.7#交叉概率
pm = 0.02#变异概率
M = 10#精英库个体数
u = 0.6#差异度界线
def init():
    #初始化种群
    pop = np.random.randint(0,2,[pop_size,chromosome_len])#生成种群
    return pop
def get_chomo_str(pop,size):
    #获得字符串形式的二进制编码
    res = []
    for i in range(size):
        chromosome_str = [str(i) for i in pop[i]]#把整数列表转换成字符串
        chromosome_str = "".join(chromosome_str)#拼接成字符串
        res.append(chromosome_str)
    return res
def decodechrom(chromosome,lower_bound,upper_bound):
    #解码染色体
    chromosome_str = [str(i) for i in chromosome]#把整数列表转换成字符串
    chromosome_str = "".join(chromosome_str)#拼接成字符串
    dec = int(chromosome_str,2)#转换成十进制
    return lower_bound + dec/(2**len(chromosome)-1)*(upper_bound-lower_bound)
def decode(pop,size):
    #解码种群
    x = []
    for i in range(size):
        x.append(decodechrom(pop[i],0,9))
    return x
def eval(x):
    #计算适度值
    return x + 10*math.sin(5*x) + 7*math.cos(4*x)
def get_fit_value(x,size):
    #获得种群适度值
    fit_value = []
    for i in range(size):
        fit_value.append(eval(x[i]))
    return fit_value
def get_child_point(father,mother,copint,type):
    #单点交叉
    father = np.array(father)
    mother = np.array(mother)
    if type == "son":
        father[copint:] = 0
        mother[:copint] = 0
    elif type == "daughter":
        father[:copint] = 0
        mother[copint:] = 0
    return father + mother
def reverse(list):
    return [ele for ele in reversed(list)]
def get_child_reverse(father,mother,copint,type):
    #翻转交叉
    #123ABC
    #456DEF
    #子代
    #123FED
    #456CBA
    father = np.array(father)
    mother = np.array(mother)
    if type == "son":
        father[copint:] = 0
        mother[:copint] = 0
        mother[copint:] = reverse(mother[copint:])
    elif type == "daughter":
        father[:copint] = 0
        mother[copint:] = 0
        father[copint:] = reverse(father[copint:])
    return father + mother
def chom_strTolist(chrom):
    '''
    将字符串形式的二进制编码转换成列表
    '''
    l = [int(char) for char in chrom]
    return l
def cross(elite,team,type='A'):
    '''
    交叉\n
    elite:精英个体\n
    team:进化团队\n
    type:进化团队种类，A代表以eliteA为中心的进化，B代表以eliteB为中心的进化
    '''
    #交叉
    offspring = []#子代
    if type == 'A':
        for i in team:
            if np.random.uniform(0,1)<=pc:
                father = chom_strTolist(i)
                mother = chom_strTolist(elite)
                if i in POPe['chromosome'].to_list():
                    D = diff(father,mother)
                    if D <= u:
                        #两个体间差异较小，通过翻转交叉产生新个体
                        copint = np.random.randint(0,int(chromosome_len/2))#交换基因的位置
                        son = get_child_reverse(father,mother,copint,type="son")
                        daughter = get_child_reverse(father,mother,copint,type="daughter")
                    else:
                        #两个体间差异大，直接通过单点交叉产生新个体
                        copint = np.random.randint(0,int(chromosome_len/2))#交换基因的位置
                        son = get_child_point(father,mother,copint,type="son")
                        daughter = get_child_point(father,mother,copint,type="daughter")
                else:
                    #单点交叉
                    copint = np.random.randint(0,int(chromosome_len/2))#交换基因的位置
                    son = get_child_point(father,mother,copint,type="son")
                    daughter = get_child_point(father,mother,copint,type="daughter")
            else:
                son = chom_strTolist(i)
                daughter = chom_strTolist(elite)
            offspring.append(son)
            offspring.append(daughter)
        return offspring
    elif type == 'B':
        for i in team:
            if np.random.uniform(0,1)<=pc:
                father = chom_strTolist(i)
                mother = chom_strTolist(elite.item())
                copint = np.random.randint(0,int(chromosome_len/2))#交换基因的位置
                son = get_child_point(father,mother,copint,type="son")
                daughter = get_child_point(father,mother,copint,type="daughter")
            else:
                son = chom_strTolist(i)
                daughter = chom_strTolist(elite.item())
            offspring.append(son)
            offspring.append(daughter)
        return offspring
def mutate(pop):
    #变异
    for i in range(len(pop)):
        if np.random.uniform(0,1) <= pm:
            #变异
            position = np.random.randint(0,len(pop[i]))
            if pop[i][position] == 1:
                pop[i][position] = 0
            else:
                pop[i][position] = 1
    return pop
def diff(i,j):
    #计算个体i和j之间的差异度
    sum = 0
    for k in range(chromosome_len):
        sum = sum + abs(i[k]-j[k])
    sum = sum / chromosome_len
    return sum
def cum_p(tab):
    #计算累积概率
    t = 0
    cum = []
    for i in range(tab.shape[0]):
        t = t + tab.loc[i,"p"]
        cum.append(t)
    return cum
def select(tab,number):
    #选择【轮盘赌选择】
    fit_value_sum = tab['fit_value'].sum()
    tab['p'] = tab['fit_value']/fit_value_sum#个体选择概率
    cum = cum_p(tab)#获得累积概率
    # for index,row in tab.iterrows():
    #     rand = random.random()#[0,1)之间的随机浮点数
    #     if row['cum_p'] < rand:
    #         tab = tab.drop(index)
    count = 0
    new_pop = pd.DataFrame(columns=["chomosome","x","fit_value"])
    for i in range(len(cum)):
        rand = np.random.uniform(0,1)
        for j in range(len(cum)):
            if j == 0:
                if rand > 0 and cum[j] >= rand:
                    new_pop = new_pop.append(tab.iloc[j,:],ignore_index=True)
                    count += 1
            else:
                if cum[j-1] < rand and cum[j] > rand:
                    new_pop = new_pop.append(tab.iloc[j,:],ignore_index=True)
                    count += 1
            if count == number:
                break
    # tab = tab.reset_index(drop=True)
    # tab = tab.drop(['p','cum_p'],axis=1)
    return new_pop
def get_teamB_selection(pop_all,elite,number):
    '''
    根据个体差异度评价函数选择teamB
    pop_tab:种群dataframe
    elite:eliteB
    number:选择的个体数
    '''
    pop_tmp = copy.copy(pop_all)
    eliteB = chom_strTolist(elite['chromosome'].item())
    #print("eliteB",elite['chromosome'].item())
    for index,row in pop_tmp.iterrows():
        pop_tmp.iloc[index,2] = row['fit_value']*diff(eliteB,chom_strTolist(row['chromosome']))
    return select(pop_tmp,number)
pop = init()#初始化种群
x = decode(pop,pop_size)
fit_value = get_fit_value(x,pop_size)#获得适度值
pop_tab = pd.DataFrame({"chromosome":get_chomo_str(pop,pop_size),"x":x,"fit_value":fit_value})#构造表
pop_tab = pop_tab.drop(pop_tab[pop_tab['fit_value']<0].index)
pop_tab = pop_tab.sort_values(["fit_value"],ascending=False).reset_index(drop=True)#根据适度值进行排序
elite = pop_tab.iloc[:M,:]#精英库
t = 0#代
maxGen = 20#最大代数
max = []#记录每一代中的最优解
while(t<maxGen):
    eliteA = elite.iloc[0,:]#最优个体
    aver = pop_tab['fit_value'].mean()
    POPe = pop_tab[pop_tab['fit_value']>=aver]#精英种群population_elite
    POPc = pop_tab[pop_tab['fit_value']<aver]#普通种群population_common
    eliteB = elite[elite['chromosome']!=eliteA['chromosome']].sample(n=1)#随机选择一个与eliteA相异的个体作为eliteB
    teamA = pop_tab.sample(n=int(pop_size/4))#选择n/4个个体组成交配teamA的个体
    A = cross(eliteA['chromosome'],teamA['chromosome'].tolist(),type='A')#交叉产生A(t+1)
    r0 = 0.5#初始规模参数
    d = 0.1#调整步长
    if t>=0 and t<maxGen/4:
        lamda = 0
    elif t>=maxGen/4 and t<maxGen/2:
        lamda = 1
    elif t>=maxGen/2 and t<3*maxGen/4:
        lamda = 2
    elif t>=3*maxGen/4 and t<=maxGen:
        lamda = 3
    r = r0 + lamda*d#计算出规模参数
    teamB_size = int(r*pop_size/4)#teamB个体数
    teamB_rand = np.random.randint(0,2,[teamB_size,chromosome_len])#随机生成teamB
    teamB_rand = get_chomo_str(teamB_rand,teamB_size)#将生成个体的二进制编码转换成字符串
    teamB_sel = get_teamB_selection(pop_tab,eliteB,int((1-r)*pop_size/4))#进行选择，选择个数为(1-r)*pop_size/4)
    teamB = teamB_sel['chromosome'].tolist()
    teamB.extend(teamB_rand)#合并随机生成的和选择的个体为teamB
    #print(teamB)
    B = cross(eliteB['chromosome'],teamB,type='B')#交叉产生B(t+1)
    #分别进行变异
    A = mutate(A)#A(t+1)
    B = mutate(B)#B(t+1)
    #合并两个种群
    temp = A + B
    #print(len(temp))
    x = decode(temp,len(temp))#解码子代
    fit_value = get_fit_value(x,len(temp))#计算子代的适度值
    temp_pop_tab = pd.DataFrame({"chromosome":get_chomo_str(temp,len(temp)),"x":x,"fit_value":fit_value})#构造表
    for i,e in elite.iterrows():
        if e['chromosome'] not in temp_pop_tab['chromosome']:#子代中不存在该精英个体
            #用该精英个体替换该子代中最差个体
            index = temp_pop_tab[temp_pop_tab['fit_value']==temp_pop_tab['fit_value'].min()].index#获得最差个体的索引
            temp_pop_tab.loc[index,'chromosome'] = e['chromosome']
            temp_pop_tab.loc[index,'x'] = e['x']
            temp_pop_tab.loc[index,'fit_value'] = e['fit_value']
    pop_tab = temp_pop_tab#得到P(t+1)代
    pop_tab = pop_tab.sort_values(["fit_value"],ascending=False).reset_index(drop=True)#根据适度值进行排序
    if pop_tab.loc[0,'fit_value'] > elite.loc[0,'fit_value']:
        #新种群最优个体大于精英库中的最优个体,替换最差个体，得到精英库Best(t+1)
        elite.loc[9,'chromosome'] = pop_tab.loc[0,'chromosome']
        elite.loc[9,'x'] = pop_tab.loc[0,'x']
        elite.loc[9,'fit_value'] = pop_tab.loc[0,'fit_value']
    print("第%d代\n最优个体%s\nx=%lf,\nfit_value=%lf"%(t+1,pop_tab.loc[0,"chromosome"],pop_tab.loc[0,"x"],pop_tab.loc[0,"fit_value"]))
    max.append(pop_tab.loc[0,"fit_value"])
    t += 1#下一代
plt.plot(range(1,21),max)
plt.xticks(np.arange(1,21,1))
plt.show()