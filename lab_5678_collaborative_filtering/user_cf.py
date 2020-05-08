"""

======================================
推荐系统实验二：基于用户的协同过滤

@Author: Peng Li
@Date: 2020/05/07
======================================

数据集介绍：

## 评分数据集
All ratings are contained in the file "ratings.dat" and are in the
following format:

UserID::MovieID::Rating::Timestamp

- UserIDs range between 1 and 6040
- MovieIDs range between 1 and 3952
- Ratings are made on a 5-star scale (whole-star ratings only)
- Timestamp is represented in seconds since the epoch as returned by time(2)
- Each user has at least 20 ratings

## 电影名称数据集
Movie information is in the file "movies.dat" and is in the following
format:

MovieID::Title::Genres

- Titles are identical to titles provided by the IMDB (including
year of release)
- Some MovieIDs do not correspond to a movie due to accidental duplicate
entries and/or test entries
- Movies are mostly entered by hand, so errors and inconsistencies may exist

======================================

参考：
- 《推荐系统实践》2.4节
- https://blog.csdn.net/qq_34105362/article/details/84345044
- https://blog.csdn.net/houyanhua1/article/details/87874397

=======================================

现存问题
- 冷启动

=======================================
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import math
from tqdm import tqdm
import pickle   # pickle存储大规模矩阵的时候会有错误，可以换用joblib

from metric import Metric
from rec_base import *


class Recommend2User:
    def __init__(self,type,train_pd,movies_name):
        self.type = type  # "user_cf"  "item_cf"
        self.trian_pd = train_pd
        self.movies_name = movies_name
        self.user_movies = userMovies(train_pd)
        self.movie_users = movieUsers(train_pd)
        self.ratings = train_pd.set_index(['user_id','movie_id'])['ratings']
        if self.type == "user_cf":
            need_norm = False
        elif self.type == "item_cf":
            need_norm = True
        self.sim_matrix = Similarity(type=self.type, tmp_train_pd=train_pd, need_norm=need_norm)

    def pred(self,user_id,N,K):
        ## 异常处理
        if user_id not in self.user_movies or user_id not in self.sim_matrix:
            # print("该用户没有出现在训练集中")
            return []
        elif len(self.sim_matrix[user_id]) == 0:
            # print("该用户在训练集中没有相似用户")
            return []

        ## 该用户看过的电影，就不推荐了,格式为np的array
        user_seen = result2list(self.user_movies.loc[user_id])
        ## topK个相似用户
        topK_sim_users = self.sim_matrix[user_id][:K]

        ## 得到推荐的电影及推荐概率
        rec_movies = {}
        for u, similarity in topK_sim_users:
            # 遍历每个相似用户看过的电影
            u_movies = result2list(self.user_movies.loc[u])
            for movie in u_movies:
                # 要去掉用户见过的
                if movie not in user_seen:
                    if movie not in rec_movies:
                        rec_movies[movie] = 0
                    rec_movies[movie] += (similarity * self.ratings.loc[u, movie])
        rec_movies = list(sorted(rec_movies.items(), key=lambda x: x[1], reverse=True))[:N]
        rec_list = [item[0] for item in rec_movies]

        ## 打印推荐结果样例
        # print("推荐结果：")
        # print("需要推荐电影的用户id： ", user_id)
        # print("推荐的电影： ")
        # for movie_id in rec_list:
        #     print("电影id： ", movie_id, "电影名称：", self.movies_name.loc[int(movie_id)])
        return rec_list

if __name__ == '__main__':
    print ("="*30)
    print ("User-Based-Collaborative-Filtering")
    print("=" *30)
    ## 加载数据集
    test_size = 0.02
    ratings_data_path = "movielens/ratings.dat"
    movies_data_path = "movielens/movies.dat"
    sim_matrix_path = "sim_matrix_with_testsize_" + str(test_size)
    user_movie_num_path = "user_movie_num_with_testsize_" + str(test_size)

    training_data,test_data = loadDataAndSplit(ratings_data_path,test_size=test_size)# 后面测试数据集大小要调整
    movies_name = loadMoviesName(movies_data_path)
    # print ("\n训练集例子：")
    # print (training_data[:3])
    # print ("\n测试集例子：")
    # print (test_data[:3])

    ## 为某一用户推荐电影
    # K表示相似用户数取值
    # N表示推荐电影数取值
    rec2user = Recommend2User(type="user_cf",train_pd=training_data,movies_name=movies_name)
    # rec2user.pred(695,N=10,K=5)

    ## 构建用户到电影的索引
    tmp_train_UM = userMovies(training_data)
    tmp_test_UM = userMovies(test_data)

    k_list = [5,10,20,50]
    for k in k_list:
        print ("评测模型：")
        print ("N=10"," K=",k)
        tmp_metric = Metric(type='user_cf',train_UM=tmp_train_UM,test_UM=tmp_test_UM,Recommender=rec2user,N=10,K=k)
        tmp_metric.eval()

