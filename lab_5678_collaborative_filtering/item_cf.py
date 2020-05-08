from rec_base import *
from metric import Metric

class Recommend2User:
    def __init__(self, type, train_pd, movies_name):
        self.type = type  # "user_cf"  "item_cf"
        self.trian_pd = train_pd
        self.movies_name = movies_name
        self.user_movies = userMovies(train_pd)
        self.movie_users = movieUsers(train_pd)
        self.ratings = train_pd.set_index(['user_id', 'movie_id'])['ratings']
        if self.type == "user_cf":
            need_norm = False
        elif self.type == "item_cf":
            need_norm = True
        self.sim_matrix = Similarity(type=self.type, tmp_train_pd=train_pd, need_norm=need_norm)

    def pred(self,user_id,N,K):
        ## 异常处理
        if user_id not in self.user_movies :
            # print("该用户没有出现在训练集中")
            return []
        user_seen = result2list(self.user_movies.loc[user_id])
        rec_movies = {}
        for movie in user_seen:
            sim_movies = self.sim_matrix[movie][:K]
            for item in sim_movies:
                m,sim = item
                if m not in user_seen:
                    if m not in rec_movies :
                        rec_movies[m] = 0
                    rec_movies[m] += sim * self.ratings.loc[user_id, movie]
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
    print ("Item-Based-Collaborative-Filtering")
    print("=" *30)

    test_size = 0.02
    ratings_data_path = "movielens/ratings.dat"
    movies_data_path = "movielens/movies.dat"

    training_data,test_data = loadDataAndSplit(ratings_data_path,test_size=test_size)# 后面测试数据集大小要调整
    movies_name = loadMoviesName(movies_data_path)

    rec2user = Recommend2User(type= "item_cf",train_pd=training_data,movies_name=movies_name)
    # rec2user.pred(user_id=922,N=10,K=5)

    train_UM = userMovies(training_data)
    test_UM = userMovies(test_data)

    print ("评测模型：")
    K = [10,20,50]
    for k in K:
        tmp_metric = Metric(type='item_cf',train_UM=train_UM,test_UM=test_UM,Recommender=rec2user,N=10,K=k)
        tmp_metric.eval()



