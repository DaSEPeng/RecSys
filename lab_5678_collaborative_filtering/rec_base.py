"""

通用函数

"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import math
from tqdm import tqdm
import pickle   # pickle存储大规模矩阵的时候会有错误，可以换用joblib

def save_obj(obj, name):
    """
    保存对象，主要是用来存储一些计算时间较长的数据，方便下次使用时直接读取
    :param obj: 对象
    :param name: 路径名
    :return: 无
    """
    with open('obj/'+ name + '.pkl', 'wb') as f:
        pickle.dump(obj, f,pickle.HIGHEST_PROTOCOL)

def load_obj(name):
    """
    从路径名对应的位置加载对象
    :param name: 路径名
    :return: 无
    """
    with open('obj/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)

def loadDataAndSplit(data_path,test_size):
    """
    加载数据集并划分训练集测试集
    返回DataFrame, 第一列是user_id, 第二列是movie_id,第三列为用户对电影的评分
    :param data_path: 数据集路径
    :param test_size: 测试集比例
    :return:
    """
    ## 读取数据
    data_pd = pd.read_table(data_path,sep=',', header=None)
    data_pd = data_pd[0].str.split("::",expand=True)
    data_pd.columns = ['user_id','movie_id','ratings','timestamp']

    ## 转换数据类型
    data_pd['user_id'] = data_pd['user_id'].astype("int")
    data_pd['movie_id'] = data_pd['movie_id'].astype("int")
    data_pd['ratings'] = data_pd['ratings'].astype("int")  # 原书中并没有用这里的评分数据

    ## 分割训练集测试集
    tmp_train,tmp_test = train_test_split(data_pd.iloc[:,:3],test_size=test_size,random_state=233)
    print ("训练集大小：", tmp_train.shape[0])
    print ("测试集大小：", tmp_test.shape[0])
    return tmp_train, tmp_test

def loadMoviesName(data_path):
    """
    读取电影信息表，主要用来得到为user_id推荐的电影对应的电影名
    :param data_path: 电影列表路径
    :return: pandas的series，索引为电影id，对应的为电影名
    """
    data_pd = pd.read_table(data_path,header=None,encoding='latin-1')
    data_pd = data_pd[0].str.split("::",expand=True)
    data_pd.columns = ['movie_id','title','genres']
    data_pd['movie_id'] = data_pd['movie_id'].astype("int")
    data_pd = data_pd.iloc[:,:2].set_index("movie_id")['title']
    return data_pd

def movieUsers(tmp_pd):
    """
    这里是 电影用户索引表，使用了pandas的set_index函数来hash
    没有使用python3的dict
    :param tmp_pd: 原dataframe
    :return: 加了索引，然后对user_id列转换成了series
    """
    return tmp_pd.iloc[:,:2].set_index('movie_id')['user_id']

def userMovies(tmp_pd):
    """
    用户到电影的索引，同上面函数一样使用了set_index代替构建Dict
    :param tmp_pd:
    :return:
    """
    return tmp_pd.iloc[:,:2].set_index('user_id')['movie_id']

def movieUsersNum(tmp_mu,tmp_path):
    """
    计算每个电影被多少用户看过
    :param tmp_mu: 电影到用户的索引表
    :return: 返回一个字典
    """
    movie_user_num_dict = {}
    try:
        movie_user_num_dict = load_obj(tmp_path)
    except:
        print ("\n计算每个电影被多少个用户看过：")
        for m in tqdm(set(tmp_mu.index.values)):
            movie_user_num_dict[m] = len(result2list(tmp_mu.loc[m]))
        save_obj(movie_user_num_dict,tmp_path)
        print (tmp_path + " Saved!")
    return movie_user_num_dict

def userMoviesNum(tmp_um,tmp_path):
    """
    计算每个用户看的电影数目
    :param tmp_um: 用户到电影的索引表
    :return: 一个字典
    """
    user_movie_num_dict = {}
    try:
        user_movie_num_dict = load_obj(tmp_path)
    except:
        print ("\n计算每个用户看过多少个电影：")
        for u in tqdm(set(tmp_um.index.values)):
            user_movie_num_dict[u] = len(result2list(tmp_um.loc[u]))
        save_obj(user_movie_num_dict,tmp_path)
        print (user_movie_num_dict + " Saved!")
    return user_movie_num_dict

def print_samples(tmp_series,num =2):
    """
    打印一些索引样例
    :param tmp_series: 设置好index的series
    :param num: 要打印的样例数
    :return:
    """
    num = num
    for index in set(tmp_series.index.values):
        if num >0:
            print ("索引值：", index)
            print ("对应的列表：")
            print (tmp_series.loc[index].tolist())
            num -= 1
        else:
            break
    pass

def result2list(tmp_value):
    """
    辅助函数，将series按照索引返回的单个值或者pd.series转换成list
    :param tmp_value:
    :return:
    """
    if (type(tmp_value) != pd.core.series.Series):
        tmp_value = np.array([tmp_value])
    else:
        tmp_value = tmp_value.values
    return tmp_value

def Similarity(type,tmp_train_pd,need_norm=False):
    """
    构建用户或者电影相似度矩阵，利用倒排索引进行加速，采用改进版本的相似度计算方式
    返回排序后的用户相似度矩阵（注：并不是严格意义上的矩阵）
    :param type: 是基于用户的还是基于电影的CF
    :param tmp_train_pd: 输入的DataFrame
    :param num_dict: 训练集中 每个用户所评论的电影数 或者 每个电影所观看的用户数
    :param need_norm: 是否需要归一化
    :return: 相似度矩阵
    """
    try:
        return load_obj("sim-matrix-"+type)
    except:
        pass

    tmp_inverted_index = None
    if type == "user_cf":
        ## 构建电影到用户的倒排索引
        tmp_inverted_index = movieUsers(tmp_train_pd)
        print ("\n训练集电影到用户索引举例：")
        print_samples(tmp_inverted_index)
    elif type == "item_cf":
        ## 构建用户到电影的倒排索引
        tmp_inverted_index = userMovies(tmp_train_pd)
        print ("\n训练集用户到电影索引举例：")
        print_samples(tmp_inverted_index)

    ## 构建共现矩阵
    print("\n构建共现矩阵： ")
    C = {}
    for index in tqdm(set(tmp_inverted_index.index.values)):
        contents = tmp_inverted_index.loc[index]
        contents = result2list(contents)
        tmp_contents_num = len(contents)

        for tmp_j in contents:
            if tmp_j not in C:
                C[tmp_j] = {}
            for tmp_k in contents:
                if tmp_j == tmp_k:
                    continue
                if tmp_k not in C[tmp_j]:
                    C[tmp_j][tmp_k] = 0
                C[tmp_j][tmp_k] += 1/math.log(1+tmp_contents_num)   # 这里其实是两倍的存储量，但是可以节约后面的时间

    if type == "user_cf":
        num_dict = userMoviesNum(userMovies(tmp_train_pd),"user_movies_num")
    elif type == "item_cf":
        num_dict = movieUsersNum(movieUsers(tmp_train_pd),"movie_users_num")

    print ("\n得到最终的相似度矩阵：")
    print_num = 0
    norm_weight = {}
    for u in tqdm(C):
        if u not in norm_weight:
            norm_weight[u] = -1
        for v in C[u]:
            N_u = num_dict[u]
            N_v = num_dict[v]
            if print_num <2:
                print ("输出样例：")
                print ("C[u][v]: ",C[u][v])
                print ("movies num or users num: ",N_u)
                print ("movies num or users num: ",N_v)
            C[u][v] /= math.sqrt(N_u * N_v)
            if print_num <2:
                print ("sim_matrix: ",C[u][v])
                print_num+=1
            if C[u][v] > norm_weight[u]:
                norm_weight[u] = C[u][v]

    print_num = 0
    if need_norm == True:
        print ("归一化相似度权重：")
        for u in tqdm(C):
            for v in C[u]:
                C[u][v]/=norm_weight[u]
                if print_num <2:
                    print("norm_sim_matrix: ",C[u][v])
                    print_num +=1

    ## 对相似度矩阵进行排序
    # 格式举例：3726: [(1465, 0.6213349345596119), (3686, 0.6213349345596119), (198, 0.6213349345596119)]
    # 如果出现 "3726：[]" 这种情况，说明某个电影只被这一个用户看过，这一个用户也只看过这一个电影
    print ("\n对相似度矩阵进行排序：")
    sorted_sim = {k: list(sorted(v.items(), \
                               key=lambda x: x[1], reverse=True)) \
                       for k, v in tqdm(C.items())}

    print ("\n保存相似度矩阵：")
    save_obj(sorted_sim,"sim-matrix-"+type)
    return sorted_sim