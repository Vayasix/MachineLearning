#encoding:utf-8
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from collections import defaultdict

class TrainDataSet():
    def __init__(self, data):
        data = np.array(data)

        self.labels = data[:,0]
        self.data_set = data[:,1:]

    def __repr__(self):
        ret  = repr(self.labels) + "\n"
        ret += repr(self.data_set)
        return ret

    def get_data_num(self):
        return self.labels.size

    def get_labels(self, *args):
        if args is None:
            return self.labels
        else:
            return self.labels[args[0]]
    def get_data_set(self):
        return self.data_set

    def get_data_set_partial(self, *args):
        if args is None:
            return self.data_set
        else:
            return self.data_set[args[0]]
    def get_label(self, i):
        return self.labels[i]
    def get_data(self, i):
        return self.data_set[i,:]
    def get_data(self,i, j):
        return self.data_set[i][j]


size = 28
master_data= np.loadtxt('train.csv',delimiter=',',skiprows=1)
test_data= np.loadtxt('test.csv',delimiter=',',skiprows=1)

train_data_set = TrainDataSet(master_data)

def get_list_sorted_by_val(k_result, k_dist):
    result_dict = defaultdict(int)
    distance_dict = defaultdict(float)

    # 数字ラベルごとに集計
    for i in k_result:
        result_dict[i] += 1

    # 数字ラベルごとに距離の合計を集計
    for i in range(len(k_dist)):
        distance_dict[k_result[i]] += k_dist[i]

    # 辞書型からリストに変換（ソートするため）
    result_list = []
    order = 0
    for key, val in result_dict.items():
        order += 1
        result_list.append([key, val, distance_dict[key]])

    # ndarray型に変換
    result_list = np.array(result_list) 

    return result_list


k = 5
predicted_list = []    # 数字ラベルの予測値
k_result_list  = []    # k個の近傍リスト
k_distances_list = []  # k個の数字と識別対象データとの距離リスト

# execute k-nearest neighbor method
for i in range(len(test_data)):

    # 識別対象データと教師データの差分をとる
    diff_data = np.tile(test_data[i], (train_data_set.get_data_num(),1)) - train_data_set.get_data_set()

    sq_data   = diff_data ** 2       # 各要素を2乗して符号を消す
    sum_data  = sq_data.sum(axis=1)  # それぞれのベクトル要素を足し合わせる
    distances = sum_data ** 0.5      # ルートをとって距離とする
    ind = distances.argsort()        # 距離の短い順にソートしてその添え字を取り出す
    k_result = train_data_set.get_labels(ind[0:k]) # 近いものからk個取り出す
    k_dist   = distances[ind[0:k]]   # 距離情報もk個取り出す

    k_distances_list.append(k_dist)
    k_result_list.append(k_result)

    # k個のデータから数字ラベルで集約した、(数字ラベル, 個数, 距離)のリストを生成
    result_list = get_list_sorted_by_val(k_result, k_dist)
    candidate = result_list[result_list[:,1].argsort()[::-1]]

    counter = 0
    min = 0
    label_top = 0

    # もっとも数の多い数字ラベルが複数あったらその中で合計距離の小さい方を選択
    result_dict = {}
    for d in candidate:
        if d[1] in result_dict:
            result_dict[d[1]] += [(d[0], d[2])]
        else:
            result_dict[d[1]] =  [(d[0], d[2])]

    for d in result_dict[np.max(result_dict.keys())]:
        if counter == 0:
            label_top = d[0]
            min = d[1]
        else:
            if d[1] < min:
                label_top = d[0]
                min = d[1]
        counter += 1

    # 結果をリストに詰める
    predicted_list.append(label_top)

# disp calc result
print "[Predicted Data List]"
for i in range(len(predicted_list)):
    print ("%d" % i) + "\t" + str(predicted_list[i])

print "[Detail Predicted Data List]"
print "index k units of neighbors, distances for every k units"
for i in range(len(k_result_list)):
    print ("%d" % i) + "\t" + str(k_result_list[i]) + "\t" + str(k_distances_list[i])
