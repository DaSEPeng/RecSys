"""

一些通用函数，如加载数据集、格式转换等

"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

def loadMovies(tmp_path):
    """
    加载电影数据集
    :param tmp_path:
    :return:
    """
    data_pd = pd.read_table(tmp_path,header=None,encoding='latin-1')
    data_pd = data_pd[0].str.split("::",expand=True)
    data_pd.columns = ['movie_id','title','genres']
    data_pd['movie_id'] = data_pd['movie_id'].astype("int")
    return data_pd

def loadUsers(tmp_path):
    """
    加载用户数据集
    :param tmp_path:
    :return:
    """
    data_pd = pd.read_table(tmp_path,header=None,encoding='latin-1')
    data_pd = data_pd[0].str.split("::",expand=True)
    data_pd.columns = ['user_id','gender','age','occupation','zipcode']
    data_pd['user_id'] = data_pd['user_id'].astype("int")
    return data_pd

def  loadRatings(tmp_path):
    """
    加载评分数据
    :param tmp_path:
    :return:
    """
    ## 读取数据
    data_pd = pd.read_table(tmp_path,sep=',', header=None)
    data_pd = data_pd[0].str.split("::",expand=True)
    data_pd.columns = ['user_id','movie_id','ratings','timestamp']

    ## 转换数据类型
    data_pd['user_id'] = data_pd['user_id'].astype("int")
    data_pd['movie_id'] = data_pd['movie_id'].astype("int")
    data_pd['ratings'] = data_pd['ratings'].astype("int")  # 原书中并没有用这里的评分数据
    return data_pd

def loadDataAndSplit(data_path,test_size=0.0):
    """
    加载评分数据集并划分训练集测试集
    返回DataFrame, 第一列是user_id, 第二列是movie_id,第三列为用户对电影的评分
    :param data_path: 数据集路径
    :param test_size: 测试集比例
    :return: 训练集和测试集
    """
    ## 加载评分数据集
    data_pd = loadRatings(data_path)

    ## 分割训练集测试集
    tmp_train,tmp_test = train_test_split(data_pd.iloc[:,:3],test_size=test_size,random_state=233)
    print ("训练集大小：", tmp_train.shape[0])
    print ("测试集大小：", tmp_test.shape[0])
    return tmp_train, tmp_test

def div0( a, b ):
    """
    修正后的numpy除法，主要解决除0的问题
    :param a:
    :param b:
    :return:
    """
    """ ignore / 0, div0( [-1, 0, 1], 0 ) -> [0, 0, 0] """
    with np.errstate(divide='ignore', invalid='ignore'):
        c = np.true_divide( a, b )
        c[ ~ np.isfinite( c )] = 0  # -inf inf NaN
    return c

def userMovies(tmp_pd):
    """
    用户到电影的索引，使用了set_index代替构建Dict
    :param tmp_pd:
    :return:
    """
    return tmp_pd.iloc[:,:2].set_index('user_id')['movie_id']

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

def get_users(tmp_pd):
    """
    从dataframe中提取出用户的id集合
    :param tmp_pd:
    :return:
    """
    data_pd_indexed = tmp_pd.set_index('user_id')
    return set(data_pd_indexed.index)