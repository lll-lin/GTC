
#from lxml import etree             python 3.5版本之前的、引入etree模块

from lxml import html
etree = html.etree

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cluster import KMeans
from time import time
from sklearn import metrics
import pandas as pd
import numpy as np
from pm4py.objects.log.importer.xes import importer as xes_importer
import json
import os


def get_sentences_XES(filename):
  texts = []
  tree = etree.parse(filename)
  root= tree.getroot()
  for element in root.iter():
      #print(element.tag)
      if('}' in element.tag):
          tag= element.tag.split('}')[1]
      else: tag= element.tag
      if(tag== "trace"):
          wordslist = []
          for childelement in element.iterchildren():
              if('}' in childelement.tag):
                  ctag= childelement.tag.split('}')[1]
              else: ctag= childelement.tag
              if (ctag=="event"):
                  for grandchildelement in childelement.iterchildren():
                      if(grandchildelement.get('key')=='concept:name'):
                          event_name=grandchildelement.get('value')
                      #    print(event_name)
                          wordslist.append(event_name.replace(' ',''))
          texts.append(' '.join(wordslist))
  return texts


def cluster(logname,clusters_num=300):

  texts= get_sentences_XES(logname+'.xes')
  vectorizer = CountVectorizer(ngram_range=(1,3))
  X = vectorizer.fit_transform(texts)

  binair= (X>0).astype(int)

  dataset=binair

  km = KMeans(n_clusters=clusters_num, init='k-means++', max_iter=100, n_init=1,
              verbose=True)

  print("Clustering sparse data with %s" % km)
  t0 = time()
  km.fit(dataset)
  print("done in %0.3fs" % (time() - t0))
  print()

  print("Silhouette Coefficient__candidateSublogs: %0.3f"
        % metrics.silhouette_score(dataset, km.labels_, sample_size=1000))

  print()



  with open('KMeansClusterLabels.csv', 'w') as f:
      f.write("traceIndex , clusterlabel \n")
      for count,item in enumerate(km.labels_):
          f.write("%s , %s \n" % (count, item) )


def traceList(logname):
  variant = xes_importer.Variants.ITERPARSE
  parameters = {variant.value.Parameters.TIMESTAMP_SORT: True}
  log = xes_importer.apply(logname + '.xes', variant=variant, parameters=parameters)
  event_log = log
  print("Number of traces：",len(event_log))
  print("Keyword:",dict(event_log[0].attributes).keys())

  key=set()
  value=[]
  for k in range (len(event_log)):
    trace=[]
    for j in range(len(event_log[k])):
      trace.append(event_log[k][j]["concept:name"])
    for t in range(len(trace)):
      key.add(trace[t])
  key=list(set(key))
  #print("key",key)
  for num in range(len(key)):
    value.append (num)

  replace= dict(zip(key, value))
  print('activityID:', replace)
  activities_num = len(replace)
  print("activities_num", len(replace))

  trace_list = []
  for k in range(len(event_log)):
      trace = []
      for j in range(len(event_log[k])):
          trace.append(event_log[k][j]["concept:name"])


      final = [replace[i] if i in replace else i for i in trace]

      trace_list.append(final)
  print("len_trace:", len(trace_list))
  return activities_num, trace_list


def direct_relation(array,trace,mu):

    for i in range(len(trace) - 1):
        array[trace[i]][trace[i + 1]]+=1/mu
    return array



def mkdir(path):
    folder = os.path.exists(path)

    if not folder:
        os.makedirs(path)
        print
        "---  new folder...  ---"
        print
        "---  OK  ---"

    else:
        print
        "---  There is this folder!  ---"





def adjMatrix(logname,clusters_num=300):

    file1 = pd.read_csv(
        r'KMeansClusterLabels.csv')
    file1 = np.array(file1)

    small = []
    for t in range(clusters_num):
        small1 = []
        for item in file1:
            sh = item[1]
            if t == sh:
                # print(item)
                small1.append(item[0])
        small.append(small1)
    print("len_clusterNum:",len(small))
    #print(small)

    activities_num, trace_list = traceList(logname)
    mkdir(logname)
    for s in range(len(small)):
        array = [[0] * activities_num for _ in range(activities_num)]
        for j in range(len(small[s])):
            array = direct_relation(array, trace_list[small[s][j]], len(small[s]))


        k1 = str(s)
        dic1 = dict(array=array)
        b = json.dumps(dic1)
        f2 = open(logname+'/' + k1 + '.json',
                  'w')
        f2.write(b)
        f2.close()
    print('adjacency matrices have been successfully saved in folder:', logname)


def main():
    usage = """\
    usage:
        driftDetection.py [-l value] [-c value] log_file_path
    options:
        -l -- the name of the event log
      	-c -- integer denoting the number of candidate sub-logs (default 300)
        """
    import getopt, sys

    try:

        opts, args = getopt.getopt(sys.argv[1:], "l:c:")
        if len(args) == 0:
            print(usage)
            return

        logname = 'BPI_Challenge_2012'
        clusters_num = 300
        args=['BPI_Challenge_2012.xes']

        for opt, value in opts:
            if opt == '-l':
                logname = str(value)
            elif opt == '-c':
                clusters_num = int(value)

        print("--------------------------------------------------------------")
        print(" Log: ", args[0])
        print(" logname: ", logname)
        print(" clusters_num: ", clusters_num)
        print("--------------------------------------------------------------")

        cluster(logname, clusters_num)
        adjMatrix(logname, clusters_num)
    except getopt.GetoptError:
        print(usage)
    except SyntaxError as error:
        print(error)
        print(usage)
    return 0


# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    main()





# See PyCharm help at https://www.jetbrains.com/help/pycharm/
