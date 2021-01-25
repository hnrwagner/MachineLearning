import numpy as np
import matplotlib.pylab as plt
import matplotlib.cm as cm

from sklearn import tree

import collections
import pydotplus
# Problem: No module named 'pydotplus' ---> 1) open         : anaconda powershell promt 
#                                      ---> 2) copy & paste : conda install -c conda-forge pydotplus

from sklearn.tree import export_graphviz






import seaborn 
seaborn.set()

# load data

#main_path = 'c:/ronald/'
# load the data
#a_11_a_22 = np.genfromtxt(main_path + 'A11_A22.txt')
a_11_a_22 = np.genfromtxt('A11_A22.txt')
# a66 = np.genfromtxt(main_path + 'A66.txt')
# d11_d22 = np.genfromtxt(main_path + 'D11_D22.txt')
# d66 = np.genfromtxt(main_path + 'D66.txt')
# stack the arrays
X = np.vstack([a_11_a_22]).transpose()
#X = np.vstack([a_11_a_22, a66, d11_d22, d66]).transpose()


#bucklingload = np.genfromtxt(main_path + 'Buckling_Load.txt')
bucklingload = np.genfromtxt('Buckling_Load.txt')
#
plt.hist(bucklingload, bins = 50)
plt.xlabel('Buckling Load [N]')
plt.ylabel('Count')

# Put the bucklingload into container in order to apply the Tree algorithm, it only works with integer classes, so continous buckling loads are not feasable

spaced_array = np.array([0.95])
bins = spaced_array * np.max(bucklingload)

bucklingload_digitize = np.digitize(bucklingload, bins)

plt.plot(bucklingload_digitize, bucklingload, 'bo')
plt.xlabel('Class')
plt.ylabel('Buckling Load [N]')

# Visualize the buckling load distribution in each class

bucklingload_bins = {}
for num, i in enumerate(bins):
    where = np.where(bucklingload_digitize == num)[0]
    bucklingload_bins[num] = bucklingload[where]
    plt.hist(bucklingload[where])
    plt.title('Class {} - Total count {}'.format(num, len(bucklingload[where])))
    plt.xlabel('Buckling Load [N]')
    plt.ylabel('Count')
    plt.show()
    
    
bucklingload_classes = bucklingload_digitize.astype(int)
unique, counts = np.unique(bucklingload_classes, return_counts=True)


# From here on, the Machine Learning part beginns


# initialize the algorithm
# look up the specific parameters
clf = tree.DecisionTreeClassifier(max_depth = 3, max_features  = 1, class_weight = 'balanced')
# train the classifier on the stacked laminate parameters and the buckling load classes
clf = clf.fit(X,bucklingload_classes)

# these are the class names, just to be sure
classe_names = clf.classes_
classe_names

# Export as dot file
dot_data = export_graphviz(clf, out_file= None, 
                feature_names = ['A11_A22'],
                class_names = [str(f) for f in clf.classes_],
                rounded = True, proportion = True, 
                precision = 2, filled = True)
#
graph = pydotplus.graph_from_dot_data(dot_data)
#
colors = ('green', 'red')
edges = collections.defaultdict(list)
#
for edge in graph.get_edge_list():
    edges[edge.get_source()].append(int(edge.get_destination()))
#
for edge in edges:
    edges[edge].sort()    
    for i in range(2):
        dest = graph.get_node(str(edges[edge][i]))[0]
        dest.set_fillcolor(colors[i])
#
graph.write_png('tree.png')










    