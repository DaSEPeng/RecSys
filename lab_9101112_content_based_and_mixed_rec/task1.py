"""
=====================================

推荐系统实验：基于内容的推荐系统
@Author: 李鹏
@Date: 2020/05/25

=====================================

实验流程：
    - 构建物品画像
    - 构建用户画像
    - 计算物品用户相似度并得到推荐结果
    - 计算准确率、召回率、覆盖率和新颖性


注：
    - 注意训练集测试集大小
    - 直接矩阵乘法求相似度计算复杂度还是比较高，或许在矩阵运算角度还能继续优化
    - 运行的时候注意修改metric.py文件中推荐结果和评价指标的保存路径
"""

from rec_base import loadMovies,loadDataAndSplit,userMovies,div0,result2list
import numpy as np
from metric import Metric
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer


def buildItemPortrait(data_pd,max_movie_id):
    """
    构建物品画像，基于tf-idf
    注意电影id并不是顺序相连的，但是这里默认电影id是升序的
    :param data_pd: 物品数据集
    :return:
    """
    ## tf-idf编码电影类型
    data_pd['genres'] = data_pd['genres'].str.replace('|'," ")
    data_pd['genres'] = data_pd['genres'].str.replace('-','')  # 这里是为了适配sklearn的tfidf函数，也可以修改正则表达式
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(data_pd['genres']).toarray()
    print ("电影编码特征名称：")
    print(vectorizer.get_feature_names())
    idx_set = set(data_pd.loc[:,'movie_id'].values.tolist())
    dummy_ids = []    # 哑变量，为了补全电影id来构造矩阵，后面会去掉
    for i in range(max_movie_id+1):      # 前面有补0行
        if i not in idx_set:
            dummy_ids.append(i)
            X = np.insert(X,i,0,axis=0)  # 第i行表示第i个电影的编码
    print ("电影编码形状：")
    print(X.shape)
    return X,dummy_ids # [电影数+1，编码维度]

def buildUserPortrait(data_pd,item_portrait,max_user_id,max_movie_id):
    """
    构建用户画像
    :param data_pd: 训练集
    :param item_portrait: 物品画像
    :param max_user_id: 最大的用户id
    :param max_movie_id: 最大的电影id
    :return: 用户画像（用户-->embedding）
    """
    data_pd_indexed = data_pd.set_index('user_id')
    indices = set(data_pd_indexed.index)

    user_movies = np.zeros((max_user_id+1,max_movie_id+1))
    for idx in tqdm(indices): # 用户id
        tmp_pd = data_pd_indexed.loc[idx]
        if len(tmp_pd.shape)==1:
            movies_id_list = [tmp_pd['movie_id']]
            ratings_list = [tmp_pd['ratings']]
        else:
            movies_id_list = tmp_pd.movie_id.values
            ratings_list = tmp_pd.ratings.values
        tmp_np = np.zeros(max_movie_id+1)
        tmp_np[movies_id_list] = 1              # 这里是直接求和
        # tmp_np[movies_id_list] = ratings_list # 这里是加权
        user_movies[idx] = tmp_np
    # print ("用户-电影形状：")
    # print (user_movies.shape)
    user_portrait = user_movies.dot(item_portrait)
    print ("用户画像形状：")
    print (user_portrait.shape)
    return user_portrait  # [用户数+1，编码维度]

class Recommend2User:
    def __init__(self, training_data,user_portrait,item_portrait,dummy_movie_ids):
        data_pd_indexed = training_data.set_index('user_id')
        self.users = set(data_pd_indexed.index)
        self.user_movies = userMovies(training_data)
        self.user_portrait = user_portrait  # [用户数+1，编码维度]
        self.item_portrait = item_portrait  # [物品数+1，编码维度]
        self.dummy_movie_ids = set(dummy_movie_ids)
        self.sim = self.sim_matrix()   # [用户数+1，物品数+1]
        self.sim_idx = np.argsort(-self.sim,axis=1)

    def sim_matrix(self):
        """
        用户到物品的相似度矩阵
        :return:
        """
        user_num_add1 = self.user_portrait.shape[0]
        item_num_add1 = self.item_portrait.shape[0]
        sim = self.user_portrait.dot(self.item_portrait.T) # [用户数+1，物品数+1]
        user_norm = np.sqrt(np.sum(self.user_portrait*self.user_portrait,axis=1))
        item_norm = np.sqrt(np.sum(self.item_portrait*self.item_portrait,axis=1))

        sim = div0(sim,(user_norm.reshape((user_num_add1,1)).\
                        dot(item_norm.reshape((item_num_add1,1)).T))) # 调整后的除法
        return sim

    def pred(self,user_id,K):
        """
        为user_id预测K个电影
        :param user_id:
        :param K:
        :return:
        """
        ## 异常处理
        if user_id not in self.users:
            # print("该用户没有出现在训练集中")
            return []
        rec_list = []
        tmp_sim_sorted_idx = self.sim_idx[user_id]  # 对应的电影相似度索引
        user_seen = set(result2list(self.user_movies.loc[user_id])) # 注意要把看过的去掉

        # 根据相似度进行推荐，注意去掉之前加上的哑变量
        num = 0
        for i in range(len(tmp_sim_sorted_idx)):
            tmp_rec = tmp_sim_sorted_idx[i]
            if tmp_rec not in self.dummy_movie_ids and tmp_rec not in user_seen:
                rec_list.append(tmp_sim_sorted_idx[i])
                num+=1
            else:
                pass
            if num==K:
                break
        assert len(rec_list)==K
        return rec_list

if __name__ == '__main__':
    base_path = 'movielens/'
    movies_path = base_path + 'movies.dat'
    ratings_path = base_path + 'ratings.dat'

    max_movie_id = 3952
    max_user_id = 6040

    movies_pd = loadMovies(movies_path)
    movies_portrait,dummy_movie_ids = buildItemPortrait(movies_pd,max_movie_id) # np.array 第i列表示第i个电影的编码

    training_data,test_data = loadDataAndSplit(ratings_path,test_size=0.5)
    user_portrait = buildUserPortrait(training_data,movies_portrait,max_user_id,max_movie_id)  # [用户数，编码维度]

    recommender = Recommend2User(training_data,user_portrait,movies_portrait,dummy_movie_ids)

    print ("评测模型：")
    K = [10]
    for k in K:
        print ("K = ",k)
        tmp_metric = Metric(type='content_rec',training_data = training_data,\
                            test_data=test_data,Recommender=recommender,K=k)
        tmp_metric.eval()

