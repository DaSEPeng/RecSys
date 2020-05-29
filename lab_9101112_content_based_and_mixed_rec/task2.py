"""
=====================================

推荐系统实验：推荐系统的融合排序
@Author: 李鹏
@Date: 2020/05/28

=====================================

实验流程：
    - 召回：根据基于用户的CF、基于内容的推荐得到每个测试用户的召回列表
    - 排序：将用户和电影的属性进行编码，训练LR模型，以用户是否看过电影为标签，得到的LR模型能够基于
           用户和电影预测出其观看的概率，基于此模型对召回的列表重新排序，得到修正后的结果

注：
    - 运行的时候注意修改metric.py文件中推荐结果和评价指标的保存路径
"""

import numpy as np
import pandas as pd
from rec_base import loadMovies,loadUsers,loadRatings,loadDataAndSplit
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from metric import Metric

def recall(result1_path,result2_path):
    """
    对基于推荐的结果进行聚合，进行召回，注意这里为了维护数组形状并没有去重，
    去重操作在最后推荐的时候进行
    :param result1_path:
    :param result2_path:
    :return:
    """
    result1 = pd.read_csv(result1_path,header=None,index_col=0) # [user_num,10]
    result2 = pd.read_csv(result2_path,header=None,index_col=0) # [user_num,10]
    result1_np = result1.values
    result2_np = result2.values
    result = np.concatenate((result1_np,result2_np),axis=1) # [user_num,20]
    return result

def movies_embed(tmp_pd,max_movie_id):
    """
    根据电影的类型对电影进行向量化表示，过程中为了后面矩阵计算加入了哑变量
    注意：该混合模型由于对召回的数据进行重排，所以这里最后推荐的时候并不需要哑变量
    :param tmp_pd: 电影数据集
    :param max_movie_id:
    :return:
    """
    ## tf-idf编码电影类型
    tmp_pd['genres'] = tmp_pd['genres'].str.replace('|'," ")
    tmp_pd['genres'] = tmp_pd['genres'].str.replace('-','')  # 这里是为了适配sklearn的tfidf函数，也可以修改正则表达式
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(tmp_pd['genres']).toarray()
    print ("电影编码特征名称：")
    print(vectorizer.get_feature_names())
    idx_set = set(tmp_pd.loc[:,'movie_id'].values.tolist())
    dummy_ids = []    # 哑变量，为了补全电影id来构造矩阵，后面会去掉
    for i in range(max_movie_id+1):      # 前面有补0行
        if i not in idx_set:
            dummy_ids.append(i)
            X = np.insert(X,i,0,axis=0)  # 第i行表示第i个电影的编码
    print ("电影编码形状：")
    print(X.shape)
    return X,dummy_ids # [电影数+1，编码维度]

def users_embed(tmp_pd,max_user_id):
    """
    根据用户的属性对用户进行编码
    dataframe列名：['user_id','gender','age','occupation','zipcode']
    :param tmp_pd:用户数据集
    :param max_user_id: 这里先假设所有的id都顺序相连，所以实际并没有加哑变量
    :return:
    """
    onehot_enc = OneHotEncoder(sparse=False)
    gender_embed = onehot_enc.fit_transform(tmp_pd['gender'].values.reshape((-1,1)))  # [user_num,2]
    age_embed = onehot_enc.fit_transform(tmp_pd['age'].values.reshape((-1,1))) # [user_num,7]
    occupation_embed = onehot_enc.fit_transform(tmp_pd['occupation'].values.reshape((-1,1))) # [user_num,21]
    # 这里对zip-code进行了约简，不然维度太高，太稀疏，模型效果不好
    # 这里只保留了州（第一个数字）
    # ref: https://wenwen.sogou.com/z/q795434127.htm?ch=fromnewwenwen.pc
    tmp_pd['zipcode'] = tmp_pd['zipcode'].str[:1]
    zip_embed = onehot_enc.fit_transform(tmp_pd['zipcode'].values.reshape((-1,1))) # [user_num,3439] 需要考虑要不要去掉
    result = np.concatenate((gender_embed,age_embed,occupation_embed,zip_embed),axis=1)
    result = np.insert(result,0,0,axis=0) # 前面加一个0行
    print ("用户编码形状：")
    print (result.shape)
    return result # [user_num+1,embed_dim]

def user_movie_embed(tmp_np,user_emb,movie_emb):
    """
    对用户--电影对进行编码
    :param tmp_np: [data_size,2]
    :param user_emb: [user_num+1,emb_dim]
    :param movie_emb: [movie_emb+1,emb_dim]
    :return:
    """
    tmp_np = tmp_np.T # [2,data_size] 第一行为用户id，第二行为电影id
    ## 下面直接lookup就好了
    tmp_users_emb = user_emb[tmp_np[0]]  # [data_size,emb_dim]
    tmp_movie_emb = movie_emb[tmp_np[1]] # [data_size,emb_dim]
    return np.concatenate((tmp_users_emb,tmp_movie_emb),axis=1)

def negtive_sampling(tmp_pd,neg_rate,max_movie_id,max_user_id):
    """
    对用户没有看过的电影进行采样，这里先构造了一个2维的矩阵，用1表示用户看过的点
    利用np.where函数得到没有看过的点的坐标，然后利用randint()函数得到要采样数目的坐标
    :param tmp_pd: rating数据集
    :param neg_rate: 负样本占全体数据集的比例
    :param max_movie_id: 最大的电影id
    :param max_user_id: 最大的用户id
    :return:
    """
    ## 负采样，得到正样本和负样本
    mask = np.zeros((max_user_id+1,max_movie_id+1)) # 标记哪些点已经被选过了
    seen_points = tmp_pd.iloc[:,:2].values.T
    pos_result = seen_points.T
    pos_num = len(pos_result)
    mask[seen_points[0],seen_points[1]] = 1
    neg_num = int(neg_rate/(1-neg_rate) * pos_num)
    neg_points = np.array(np.where(mask==0))  # [2,neg_points_num]
    neg_points_num = len(neg_points.T)
    np.random.seed(1234)
    neg_idx = np.random.randint(0,neg_points_num,neg_num)
    x_idx = neg_points[0][neg_idx]
    y_idx = neg_points[1][neg_idx]
    neg_result = np.array([x_idx,y_idx]).T
    ## 打印形状
    print("正样本形状：")
    print(pos_result.shape)
    print("负样本形状:")
    print(neg_result.shape)
    return pos_result,neg_result

class Recommend2User():
    def __init__(self,recalled_np,users_vec,movies_vec,lr_clf):
        self.recall_list = recalled_np
        self.users_vec = users_vec
        self.movies_vec = movies_vec
        self.lr = lr_clf
        self.rec_dict = self.resorted()

    def resorted(self):
        """
        基于逻辑回归对原始召回列表进行重排
        :return:
        """
        tmp_dict = {}
        for i in range(self.recall_list.shape[0]):
            rec_np = np.array(list(set(self.recall_list[i])))
            tmp_user_emb = self.users_vec[i]
            tmp_movies_emb = self.movies_vec[rec_np] # [movie_num.emb]
            scale = np.ones([tmp_movies_emb.shape[0],len(tmp_user_emb)]) # 用于广播
            tmp_user_emb = tmp_user_emb * scale
            concatted = np.concatenate((tmp_user_emb,tmp_movies_emb),axis=1) # [movies_num,emb_dim]
            prob = self.lr.predict_proba(concatted).T[1]  # 得到观看的概率
            prob_sorted_idx = np.argsort(-prob)
            sorted_movies = rec_np[prob_sorted_idx]
            tmp_dict[i+1] = sorted_movies # 注意这里的从坐标到用户id的转换
        print ("词典大小：", len(tmp_dict))
        return tmp_dict

    def pred(self, user_id, K):
        return self.rec_dict[user_id][:K]

if __name__ == '__main__':
    ## 数据集路径
    base_path = 'movielens/'
    movies_path = base_path + 'movies.dat'
    users_path = base_path + 'users.dat'
    ratings_path = base_path + 'ratings.dat'

    ## 对基于用户推荐和基于内容推荐的结果进行召回
    content_rec_path = 'task1_0.csv'
    user_rec_path = 'user_cf_rec_list.csv'
    recalled_np = recall(content_rec_path,user_rec_path)  # [user_num,20]

    ## 对电影和用户进行embedding
    max_movie_id = 3952
    max_user_id = 6040
    movies = loadMovies(movies_path)
    users = loadUsers(users_path)
    movies_vec,dummy_movie_list = movies_embed(movies,max_movie_id)
    users_vec = users_embed(users,max_user_id)

    ## 对评分数据集进行处理，构建LR模型要用的数据集
    ratings = loadRatings(ratings_path)
    pos_simples,neg_simples = negtive_sampling(ratings,neg_rate = 0.5,\
                                     max_movie_id=max_movie_id,max_user_id=max_user_id)  # 对数据集进行负采样
    pos_labels = [1]*pos_simples.shape[0]
    neg_labels = [0]*neg_simples.shape[0]

    all_data = np.concatenate((pos_simples,neg_simples),axis=0)
    all_data = user_movie_embed(all_data,users_vec,movies_vec)
    all_labels = np.array(pos_labels + neg_labels)
    print ("全体数据集编码后的形状")
    print (all_data.shape)

    X_train, X_test, y_train, y_test = train_test_split(all_data,all_labels,test_size=0.2,random_state=233)
    print ("训练集大小：", X_train.shape)
    print ("测试集大小：", X_test.shape)

    ## 构建LR模型
    print ("构建LR模型......")
    lr_clf = LogisticRegression(solver='lbfgs',max_iter=300)
    lr_clf.fit(X_train,y_train)
    test_pred = lr_clf.predict(X_test) # 虽然名字叫测试集，这里也主要是调参用，保证模型效果不是太差
    conf_mat = confusion_matrix(y_test,test_pred)
    print ("预测混淆矩阵：\n",conf_mat)
    print ("对应的比例：\n",conf_mat/X_test.shape[0])  # 重点看实际为1，预测为1的比例

    ## 利用调参后的LR模型对召回的结果进行重新排序
    print ("推荐.....")
    recommender = Recommend2User(recalled_np,users_vec,movies_vec,lr_clf)
    # print (recommender.pred(5,3))

    ## 评测模型
    print ("评测模型：")
    training_data,test_data = loadDataAndSplit(ratings_path,test_size=0.5)  # 重新加载原始的数据集（这里写得并不优雅）
    K = [10]
    for k in K:
        print ("K = ",k)
        tmp_metric = Metric(type='content_rec',training_data = training_data,\
                            test_data=test_data,Recommender=recommender,K=k)
        tmp_metric.eval()


