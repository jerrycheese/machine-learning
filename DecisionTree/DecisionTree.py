"""
Decision Tree with drawing

@author: Jerry
@date: Oct 25, 2018
"""

import numpy as np
from matplotlib import pyplot as plt
import math
import random

class DecisionTree(object):
    
    def __init__(self):
        self.__root = None
        self.__feature_names = []
        self.__tree_level_height = 0.2
        self.__tree_min_spacing = 0.01
        self.__tree_node_height = 0.04
        self.__tree_node_width = 0

    def train(self, X, y, feature_names):
        # build dataset =  [X, y]
        data = []
        for i in range(len(X)):
            data.append(X[i] + [y[i]])
        data = np.array(data)
        
        n_samples, n_col = data.shape
        feature_indexes = list(range(n_col - 1))
        root = self.__build(data, feature_indexes)

        self.__root = root
        self.__feature_names = feature_names

        return self

    def predict(self, X):
        pass

    def plot(self):
        root = self.__root
        if not root:
            return 

        size = self.__calcuTreeSize(root)

        fig = plt.figure()
        fig.clf()

        ax1 = plt.subplot(111, frameon=False)
        self.__plotTree(ax1, root, size)
        plt.show()

    def __plotTree(self, ax, root, xy):
        if 'label' in root:
            # leaf
            self.__plotNode(ax, root['label'], xy, 
                node=dict(boxstyle='circle', fc='white'))
            return 
        
        feature_name = self.__feature_names[root['feature']['index']]
        self.__plotNode(ax, feature_name, xy)

        # n_child = len(root['children'])
        # most_left_x = xy[0] - ((n_child - 1)*(child_spacing + 0.1))/2.0
        # y_child = xy[1] - self.__tree_level_height
        # for i in range(n_child):
        #     child = root['children'][i]
        #     x_child = most_left_x + i *(child_spacing + 0.1)
        #     xy_child = (x_child, y_child)
        #     self.__plotArrow(ax, xy, xy_child, str(child['value']))
        #     self.__plotTree(ax, child['node'], xy_child, a, level + 1)  
        
        size = root['size']
        n_child = len(root['children'])
        y_child = xy[1] - self.__tree_level_height
        x_child = xy[0] - size[0]/2.0
        for i in range(n_child):
            child = root['children'][i]
            child_size = child['node']['size']
            xy_child = (x_child + child_size[0]/2.0, y_child)
            self.__plotArrow(ax, xy, xy_child, str(child['value']))
            self.__plotTree(ax, child['node'], xy_child)
            x_child += self.__tree_min_spacing + child_size[0]

    def __plotNode(self, ax, title, xy, node=None):
        if not node:
            node = dict(boxstyle='round', fc='white')
        return ax.annotate(title, 
                xy=xy,
                ha='center',
                bbox=node)
    
    def __plotArrow(self, ax, xy_begin, xy_end, annotation = '', arrow=None):
        if not arrow:
            arrow = dict(arrowstyle='->',connectionstyle='arc3')
        
        if annotation:
            xy_text = (xy_begin[0] + xy_end[0])/2.0, (xy_begin[1] + xy_end[1])/2.0
            ax.text(xy_text[0], xy_text[1], annotation, va='center', ha='center')

        return ax.annotate('', 
            xy=xy_end, xytext=xy_begin,
            va='center', ha='center',
            arrowprops=arrow)

    def __calcuTreeSize(self, root):
        word_width = 0.018
        self_height = self.__tree_node_height
        if 'label' in root:
            self_width = len(root['label']) * word_width
            root['size'] = (self_width, self_height)
            return root['size']
        self_width = len(self.__feature_names[root['feature']['index']]) * word_width
        size = [0, 0]
        n_child = len(root['children'])
        for child in root['children']:
            size_child = self.__calcuTreeSize(child['node'])
            size[0] += size_child[0]
            size[1] = max(size_child[1], size[1])
        size[0] += self.__tree_min_spacing*(n_child - 1)
        size[1] += self.__tree_level_height + self.__tree_node_height
        size[0] = max(size[0], self_width)
        root['size'] = size
        return tuple(size)

    def __getMostY(self, data):
        """Get the label y which has the biggest number of occurrences
        
        Parameters
        ----------
        data : Data set

        Return
        ------
        (`appear times`, `label y`)
        """
        occur_map = self.__countOccurrence(data, -1)
        return max(zip(occur_map.values(), occur_map.keys()))

    def __countOccurrence(self, data, index):
        """Count the number of occurrences of each value at column index of `index` in data

        Parameters
        ----------
        data : Data set
        index : column index in `data`

        Return
        ------
        A map with map['$value'] = $appear_times
        """
        arr = data[:, index].tolist()
        spans, occur_map = set(arr), {}
        for value in spans:
            occur_map[value] = arr.count(value)
        return occur_map

    def __calcuEntropy(self, data):
        y_map = self.__countOccurrence(data, -1)
        K, m = len(y_map.keys()), len(data)
        H = 0
        for y, count in y_map.items():
            r = 1.0*count/m
            H += r*math.log2(r)
        return -H

    def __calcuContidionEntropy(self, data, feature):
        D, n_D = data, len(data)

        # calcu the condition entropy
        feature_data = self.__splitByFeature(data, feature['index'])
        H_D_A = 0
        for e in feature_data:
            Di, n_Di = e['data'], len(e['data'])
            H_Di = self.__calcuEntropy(Di)
            H_D_A += 1.0*n_Di/n_D * (-H_Di)
        H_D_A = -H_D_A
        return H_D_A

    def __pickFeatureByRandom(self, data, feature_indexes):
        features = []
        for i in feature_indexes:
            features.append({
                'index': i,
                'span': list(set(data[:,i].tolist()))
            })
        feature = random.choice(features)
        return feature

    def __pickFeatureByIEG(self, data, feature_indexes):
        """Pick a feature, which has the highest Infomation Entropy Gain

        Parameters
        ----------
        data : Data set
        feature_indexes : List of feature indexes

        Return
        ------
        (`feature`, `IEG`)
        `feature` = {'index': .., 'span': ...}
        """

        # calcu span
        features = []
        for i in feature_indexes:
            features.append({
                'index': i,
                'span': list(set(data[:,i].tolist()))
            })

        max_ieg, max_ieg_feature = 0, None
        D = data

        for f in features:
            H_D = self.__calcuEntropy(D)
            H_D_A = self.__calcuContidionEntropy(D, f)
            ieg = H_D - H_D_A
            if ieg > max_ieg:
                max_ieg = ieg
                max_ieg_feature = f
        
        return max_ieg_feature, max_ieg
    
    def __pickFeatureByIGR(self, data, feature_indexes):
        # calcu span
        features = []
        for i in feature_indexes:
            features.append({
                'index': i,
                'span': list(set(data[:,i].tolist()))
            })

        max_igr, max_igr_feature = 0, None
        D, n_D = data, len(data)
        for f in features:
            H_D = self.__calcuEntropy(D)
            H_D_A = self.__calcuContidionEntropy(D, f)
            ieg = H_D - H_D_A

            feature_data = self.__splitByFeature(data, f['index'])
            H_A_D = 0
            for e in feature_data:
                Di, n_Di = e['data'], len(e['data'])
                r = 1.0*n_Di/n_D
                H_A_D += r*math.log2(r)
            
            H_A_D = -H_A_D
            if H_A_D == 0:
                igr = 0
            else:
                igr = ieg/H_A_D
            if igr > max_igr:
                max_igr = igr
                max_igr_feature = f
        
        return max_igr_feature, max_igr

    def __splitByFeature(self, data, feature_index):
        feature_data = []
        for row in data:
            key = row[feature_index]
            value = row.tolist()

            for e in feature_data:
                if e['key'] == key:
                    e['data'].append(value)
                    break
            else:
                e = {
                    'key': key,
                    'data': [value]
                }
                feature_data.append(e)
        for f in feature_data:
            f['data'] = np.array(f['data'])
        return feature_data

    def __build(self, data, feature_indexes, min_ieg=0.0):
        """Build random forest

        Parameters
        ----------
        data: Data set
        feature_indexes: Avaliable feature's indexes
        min_ieg: Threshold to do split

        Return
        ------
        Tree node with dict type
        """
        # node = {
        #     'label': ..,
        #     'data': [...]
        #     # or
        #     'feature': {'index': .., 'span': ..}
        #     'children': [
        #       {'value': ..., 'node': ...},
        #       {'value': ..., 'node': ...},
        #       {'value': ..., 'node': ...},
        #     ]
        # }
        node = {}
        count, most_y = self.__getMostY(data)

        if (not feature_indexes or len(feature_indexes) == 0) or (count == len(data)):
            node = {'label': most_y, 'data': data.tolist()}
            return node

        # ****** key part: choose a feature to do split!
        # way 1: IEG
        # feature, ieg = self.__pickFeatureByIEG(data, feature_indexes)
        # if ieg <= min_ieg:
        #     # labeled as `most_y`, which is most appeared in data
        #     node = {'label': most_y, 'data': data.tolist()}
        #     return node
        
        # way 2: Random
        feature = self.__pickFeatureByRandom(data, feature_indexes)
        
        # way 3:entropy
        # feature, igr = self.__pickFeatureByIGR(data, feature_indexes)

        feature_indexes = feature_indexes.copy()
        
        # split data by `feature`
        node['feature'] = feature
        feature_data = self.__splitByFeature(data, feature['index'])

        # do split
        feature_indexes.remove(feature['index'])
        children = []
        for e in feature_data:
            feature_value = e['key']
            sub_data = e['data']
            children.append({
                'value': feature_value,
                'node': self.__build(sub_data, feature_indexes, min_ieg)
            })
        node['children'] = children
        return node
 
