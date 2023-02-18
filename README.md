GTC: Trace Clustering in Event Logs

Python packages required
	lxml (http://lxml.de/)
	numpy (http://www.numpy.org/)
	sklearn(https://scikit-learn.org/)
	pm4py(https://pm4py.fit.fraunhofer.de/)

How to use **************************************************

Command line usage:

	First:
	      graph_generation.py [-l value] [-c value]  log_file_path
	      options:
		    -l -- the name of the event log (BPI_Challenge_2012)
		    -c -- integer denoting the number of candidate sub-logs (default 300)

	Second:
          Clustering.py [-g value] [-n value] [-f value]
          options:
                -g grp_num -- the number of graphs, integer, default value is 300
                -n final_class_num -- the final number of target classes, integer, default value is 4
                -f filename -- the file that holds the graph's adjacency matrix, default BPI_Challenge_2012

Examples:

	First:
	      graph_generation.py -l BPI_Challenge_2012 -c 300 BPI_Challenge_2012.xes
	      or
	      graph_generation.py BPI_Challenge_2012.xes

	Second:
	      Clustering.py -g 300 -n 4 -f BPI_Challenge_2012

Explain:

	After  the First step running, some files would be generated, such as:
          BPI_Challenge_2012 -- Adjacency matrix of the graph
          KMeansClusterLabels.csv -- Kmeans clustering results

	Then, the Second step will generate 3 files:
          BPI_Challenge_2012_gcn -- Convolution result of the graph
          BPI_Challenge_2012_class -- Adjacency matrix of subclasses
          BPI_Challenge_2012_class_gcn -- Convolution result of the subclasses
 	
Note: 
	
	if you have 'Error reading file ', please using absolute path of log file.

A reference for users **************************************************

	If you have any quetions for this code, you can email: leilei_lin@126.com.
	We would also be happy to discuss the topic of tace clustering with you.



