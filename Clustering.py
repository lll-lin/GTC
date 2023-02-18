import sys
import numpy as np
import json
import os
import math

np.set_printoptions(formatter={'all': lambda x: str(x)}, threshold=10000)

class NumpyArrayEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

def eucliDist(A, B):
    return np.sqrt(np.sum((A - B)**2))
    # return math.sqrt(sum([(a - b) ** 2 for (a, b) in zip(A, B)]))

def mkdir(path):
    folder = os.path.exists(path)
    if not folder:
        os.makedirs(path)

class GraphConvolution():
    def __init__(self,A,X):
        self.A = A
        self.X = X

        I = np.matrix(np.eye(self.A.shape[0]))
        self.A_hat = self.A + I

        # self.D = np.array(np.sum(self.A, axis=0))[0]
        self.D = self.A.sum(1)
        self.D = np.matrix(np.diag(self.D))

        self.D_hat = np.array(np.sum(self.A_hat, axis=0))[0]
        self.D_hat = np.matrix(np.diag(self.D_hat))

        self.Q = self.D_hat ** -1 * self.A_hat * X
        self.QZ = (abs(self.Q) + self.Q) / 2


def GCN(grp_num):
    for i in range(grp_num):
        with open(f"./{filename}/{i}.json", 'r') as f:
            data_read = json.load(f)
            A = np.asarray(data_read['array'])

        X = np.zeros((len(A), len(A)))
        for j in range(len(A)):
            X[j][len(A) - j - 1] = 1

        gcn1 = GraphConvolution(A, X)
        X1 = gcn1.QZ
        gcn2 = GraphConvolution(A, X1)
        X2 = gcn2.QZ
        gcn3 = GraphConvolution(A, X2)
        X3 = gcn3.QZ
        mkdir(f"./{filename}_gcn")
        np.savez(f"./{filename}_gcn/file{i}", X, X1, X2, X3)
    return Embedding(grp_num)

def Embedding(grp_num):
    similarity = np.zeros((grp_num, grp_num))
    similarity = similarity - 1

    for i in range(grp_num):
        F = np.load(f"./{filename}_gcn/file{i}.npz")
        A1 = F['arr_2']
        for j in range(i + 1, grp_num):
            sum_1 = 0
            Q = np.load(f"./{filename}_gcn/file{j}.npz")
            A2 = Q['arr_2']
            for k in range(len(A1)):
                X = np.array(A1[k])
                Y = np.array(A2[k])
                sum_1 = sum_1 + eucliDist(X, Y)
            avg = sum_1 / len(A1)
            avg = round(avg, 4)
            similarity[i][j] = avg
    onset = 0
    return Normalization(similarity, onset),A1

def Normalization(similarity, onset):
    a = list(filter(lambda x: x >= 0, similarity.flatten()))
    sim_max = max(a)
    sim_min = min(a)
    for i in range(onset, len(similarity)):
        for j in range(i + 1, len(similarity)):
            similarity[i][j] = round((similarity[i][j] - sim_min) / (sim_max - sim_min), 4)
    return similarity

def Combined(similarity):
    grp_sum = []
    k = 0
    while len(grp_sum) < len(similarity):
        a = list(filter(lambda x: x >= 0, similarity.flatten()))
        d = np.flatnonzero(similarity == np.min(a))
        X = d // len(similarity)
        Y = d % len(similarity[0])

        listsim = []
        for i in range(len(X)):
            listsim.append([X[i], Y[i]])
            similarity[X[i], Y[i]] = -1

        for i in range(len(listsim)):
            if listsim[i][0] not in grp_sum and listsim[i][1] not in grp_sum:
                k = k + 1
                class_list = globals()
                class_list['class_' + str(k)] = []
                class_list['class_' + str(k)].append(listsim[i][0])
                class_list['class_' + str(k)].append(listsim[i][1])

            if listsim[i][0] in grp_sum and listsim[i][1] not in grp_sum:
                for j in range(1, k + 1):
                    if listsim[i][0] in class_list['class_' + str(j)]:
                        num = j
                        class_list['class_' + str(num)].append(listsim[i][1])
                        break

            if listsim[i][0] not in grp_sum and listsim[i][1] in grp_sum:
                for j in range(1, k + 1):
                    if listsim[i][1] in class_list['class_' + str(j)]:
                        num = j
                        class_list['class_' + str(j)].append(listsim[i][0])
                        break

            grp_sum.append(listsim[i][0])
            grp_sum.append(listsim[i][1])
            grp_sum = list(set(grp_sum))
    return Incise(k, class_list)

def Incise(k, class_list):
    if k == final_class_num:
        sys.exit()

    r = k
    for i in range(1, k + 1):
        if len(class_list['class_' + str(i)]) > 5:
            r = r + 1
            #class_list = locals()
            class_list['class_' + str(r)] = []
            class_list['class_' + str(r)].extend(class_list['class_' + str(i)][4::])
            class_list['class_' + str(i)] = class_list['class_' + str(i)][:4]
    k = r
    return k, class_list

def ClassGCN(k, A1, class_list):
    for i in range(1, k + 1):
        grp_class = np.zeros((len(A1), len(A1)))
        for j in range(len(class_list['class_' + str(i)])):
            grp_num = class_list['class_' + str(i)][j]
            with open(f"./{filename}/{grp_num}.json", 'r') as f:
                data_read = json.load(f)
                grp = np.asarray(data_read['array'])
            grp_class = grp_class + grp
        for j in range(len(grp_class)):
            for q in range(len(grp_class)):
                if grp_class[j][q] != 0:
                    grp_class[j][q] = grp_class[j][q] / len(class_list['class_' + str(i)])
        mkdir(f"./{filename}_class")
        np.savez(f"./{filename}_class/class{i}", grp_class)

    for i in range(1, k + 1):
        F = np.load(f"./{filename}_class/class{i}.npz")
        A = F['arr_0']

        X = np.zeros((len(A), len(A)))
        for j in range(len(A)):
            X[j][len(A) - j - 1] = 1

        gcn1 = GraphConvolution(A, X)
        X1 = gcn1.QZ
        gcn2 = GraphConvolution(A, X1)
        X2 = gcn2.QZ
        gcn3 = GraphConvolution(A, X2)
        X3 = gcn3.QZ

        mkdir(f"./{filename}_class_gcn")
        np.savez(f"./{filename}_class_gcn/file{i}", X, X1, X2, X3)
    return ClassEmbedding(k)

def ClassEmbedding(k):
    similarity_2 = np.zeros((k + 1, k + 1))
    similarity_2 = similarity_2 - 1
    for i in range(1, k + 1):
        F = np.load(f"./{filename}_class_gcn/file{i}.npz")
        A1 = F['arr_2']
        for j in range(i + 1, k + 1):
            sum_1 = 0
            Q = np.load(f"./{filename}_class_gcn/file{j}.npz")
            A2 = Q['arr_2']
            for q in range(len(A1)):
                X = np.array(A1[q])
                Y = np.array(A2[q])
                sum_1 = sum_1 + eucliDist(X, Y)
            avg = sum_1 / len(A1)
            avg = round(avg, 4)
            similarity_2[i][j] = avg
    onset = 1
    return Normalization(similarity_2, onset)

def ClassCluster(k, similarity_2, class_list):
    class_num = k
    clear_list = []
    class_id = {}

    while class_num > final_class_num:
        a = list(filter(lambda x: x >= 0, similarity_2.flatten()))
        d = np.flatnonzero(similarity_2 == np.min(a))
        X = d // len(similarity_2)
        Y = d % len(similarity_2[0])
        listsim = []
        for i in range(len(X)):
            listsim.append([X[i], Y[i]])
            similarity_2[X[i], Y[i]] = -1

        for i in range(len(listsim)):
            if listsim[i][0] in clear_list or listsim[i][1] in clear_list:
                break
            class_list['class_' + str(listsim[i][0])].extend(class_list['class_' + str(listsim[i][1])])
            class_list['class_' + str(listsim[i][1])].clear()
            clear_list.append(listsim[i][1])
            class_num = class_num - 1
        '''
        # the second method 
        for i in range(len(listsim)):
            if listsim[i][0] in clear_list and listsim[i][1] in clear_list:
                continue
            else:
                if listsim[i][0] in clear_list:
                    t = class_id[listsim[i][0]]
                    while t in clear_list:
                        t = class_id[t]
                    class_list['class_' + str(t)].extend(class_list['class_' + str(listsim[i][1])])
                    class_list['class_' + str(listsim[i][1])].clear()
                    clear_list.append(listsim[i][1])
                    class_num = class_num - 1
                    class_id[listsim[i][1]] = t
                    continue
                if listsim[i][1] in clear_list:
                    continue
                else:
                    class_list['class_' + str(listsim[i][0])].extend(class_list['class_' + str(listsim[i][1])])
                    class_list['class_' + str(listsim[i][1])].clear()
                    clear_list.append(listsim[i][1])
                    class_num = class_num - 1
                    class_id[listsim[i][1]] = listsim[i][0]
        '''
        if class_num == final_class_num + 2:
            for i in range(1, k + 1):
                if len(class_list['class_' + str(i)]) == 2:
                    lie_min = np.argmin(similarity_2, axis=0)
                    while similarity_2[lie_min[i]][i] == -1 or len(class_list['class_' + str(lie_min[i])]) == 0:
                        similarity_2[lie_min[i]][i] = 99
                        lie_min = np.argmin(similarity_2, axis=0)
                    class_list['class_' + str(lie_min[i])].extend(class_list['class_' + str(i)])
                    class_list['class_' + str(i)].clear()
                    class_num = class_num - 1
    return Formatting(k, class_list)

def Formatting(k, class_list):
    for i in range(1, final_class_num + 1):
        class_list2 = locals()
        class_list2['class2_' + str(i)] = []
    j = 1
    for i in range(1, k + 1):
        if len(class_list['class_' + str(i)]) != 0:
            class_list2['class2_' + str(j)].extend(class_list['class_' + str(i)])
            j = j + 1

    for i in range(1, final_class_num + 1):
        print("class", i, class_list2['class2_' + str(i)])

def main():
    usage = """\
    usage:
           Clustering.py [-g value] [-n value] [-f value]
    options:
            -g grp_num -- the number of graphs, integer, default value is 300
            -n final_class_num -- the final number of target classes, integer, default value is 4
            -f filename -- the file that holds the graph's adjacency matrix, default BPI_Challenge_2012
    """
    import getopt, sys

    try:
        opts, args = getopt.getopt(sys.argv[1:], "g:n:f:")
        if len(opts) == 0:
            print(usage)
            return


        grp_num = 300
        global final_class_num
        final_class_num = 4
        global filename
        filename = "BPI_Challenge_2012"

        for opt, value in opts:
            if opt == '-g':
                grp_num = int(value)
            elif opt == '-n':
                final_class_num = int(value)
            elif opt == '-f':
                filename = value

        print("--------------------------------------------------------------")
        print(" graph_num: ", grp_num)
        print(" final_class_num: ", final_class_num)
        print(" Log: ", filename)
        print("--------------------------------------------------------------")

        similarity, A1 = GCN(grp_num)
        k, class_list = Combined(similarity)
        similarity_2 = ClassGCN(k, A1, class_list)
        ClassCluster(k, similarity_2, class_list)

    except getopt.GetoptError:
        print(usage)
    except SyntaxError as error:
        print(error)
        print(usage)
    return 0


if __name__ == '__main__':
    main()