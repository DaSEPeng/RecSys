"""

评测类

注：
实验结果和评测指标的保存路径可能需要调整，这里在metric类中写死的实现方式不好，需要后面调整

"""

import math
from rec_base import userMovies,result2list,get_users
from tqdm import tqdm
import pandas as pd

def dict2csv(tmp_dict,saved_path,header=True):
    tmp_pd = pd.DataFrame.from_dict(tmp_dict, orient='index')
    tmp_pd.to_csv(saved_path,header=header)

class Metric():
    def __init__(self,type,training_data,test_data,Recommender,K):
        self.type = type
        self.training_data = training_data
        self.train_users = get_users(self.training_data)
        self.train_um = userMovies(self.training_data)
        self.test_data = test_data
        self.test_users = get_users(self.test_data)
        self.test_um = userMovies(self.test_data)

        self.Recommender = Recommender
        self.K = K
        self.recs = self.getRec()
        dict2csv(self.recs,saved_path="task1_0.csv",header=False)

    # 为test中的每个用户进行推荐
    def getRec(self):
        recs = {}
        for user in tqdm(self.test_users):
            rank = self.Recommender.pred(user,self.K)
            recs[user] = rank
        return recs

    # 定义精确率指标计算方式
    def precision(self):
        all, hit = 0, 0
        for user in self.test_users:
            test_items = set(result2list(self.test_um.loc[user]))
            rank = self.recs[user]
            for item in rank:
                if item in test_items:
                    hit += 1
            all += len(rank)
        return round(hit / all * 100, 2)

    # 定义召回率指标计算方式
    def recall(self):
        all, hit = 0, 0
        for user in self.test_users:
            test_items = set(result2list(self.test_um.loc[user]))
            rank = self.recs[user]
            for item in rank:
                if item in test_items:
                    hit += 1
            all += len(test_items)
        return round(hit / all * 100, 2)


    # 定义覆盖率指标计算方式
    def coverage(self):
        all_item, recom_item = set(), set()
        for user in self.test_users:
            if user not in self.train_um.index.values:
                continue
            for item in result2list(self.train_um.loc[user]):
                all_item.add(item)
            rank = self.recs[user]
            for item in rank:
                recom_item.add(item)
        return round(len(recom_item) / len(all_item) * 100, 2)

    # 定义新颖度指标计算方式
    def popularity(self):
        # 计算物品的流行度
        item_pop = {}
        for user in self.train_users:
            for item in result2list(self.train_um.loc[user]):
                if item not in item_pop:
                    item_pop[item] = 0
                item_pop[item] += 1

        num, pop = 0, 0
        for user in self.test_users:
            rank = self.recs[user]
            for item in rank:
                # 取对数，防止因长尾问题带来的被流行物品所主导
                if item not in item_pop:
                    continue
                pop += math.log(1 + item_pop[item])
                num += 1
        return round(pop / num, 6)

    def eval(self):
        metric = {'Precision': self.precision(),
                  'Recall': self.recall(),
                  'Coverage': self.coverage(),
                  'Popularity': self.popularity()}
        print('Metric:', metric)
        dict2csv(metric, saved_path="task1_1.csv",header=False)
        return metric