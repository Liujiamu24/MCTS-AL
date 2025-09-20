# -*- coding: utf-8 -*-
"""
Created on Thu Aug 14 21:29:18 2025

@author: 30865
"""

from models import models
from Voxelization import Voxelization
from data_extraction import data_extraction
from tqdm import tqdm
from typing import Any, Set, Dict, Tuple, List
from collections import defaultdict
from dataclasses import dataclass, field
import numpy as np
import matplotlib.pyplot as plt
import random
import os
import math

class signal(Exception):
    pass

'''
Sampling through MCTS

:param rd_num: iteration number in the AL
        target: average volume fraction to maintain
        num_select: best candidates selected in single iteration
        n_dim: dimensionality of problem
        rollouts: number of rollouts in the MCTS sampling
        retrain: if retrain the CNN models or just load them
        
        N: visit counts of each node
        children: the chidren of a certain node
        value: value of each node
        exploration weight: a factor balancing exploration and exploitation
'''
@dataclass
class MCTS:
    rd_num: int
    target: float = 0.5
    num_select: int = 20
    n_dim: int = 27
    rollouts: int = 100
    retrain: bool = True
    
    N: Dict[Any, int] = field(default_factory=lambda: defaultdict(int))
    children: Dict[Any, Set] = field(default_factory=lambda: defaultdict(set))
    value: Dict[Any, float] = field(default_factory=lambda: defaultdict(float))
    exploration_weight: float = 1000.0
    
    def __post_init__(self):
        self.round_name = f'Round{self.rd_num}'
        self.model_folder = os.getcwd() + f'/models/{self.round_name}'
        self.input_x, self.input_y = data_extraction(self.rd_num)
        self.models = models(self.input_x, self.input_y, self.rd_num)
        
        if not self.retrain:
            try:
                self.models.load_models()
            except:
                self.models.ensembled_training()
        else:
            self.models.ensembled_training()
        
        self.models.load_models()
    
    '''
    Choose the node with best uct as the next node
    '''
    def choose(self, node):
        """Choose the best successor of node."""
        if node not in self.children:
            children = self.find_children(node)
            if children:
                return random.choice(list(children))
            return node

        log_N_vertex = math.log(self.N[node])

        def uct(n):
            """Upper confidence bound for trees"""
            n_tup = tuple(n)
            return self.value[n_tup] + self.exploration_weight * math.sqrt(
                log_N_vertex / (self.N[n_tup] + 1)
            )

        if node not in self.children:
            self.children[node] = self.find_children(node)
            
        self.exploration_weight = self.exploration_weight * 0.95
        media_node = max(self.children[node], key=uct)

        if node in self.children[node]:
            print(True)
        
        if uct(media_node) > uct(node):
            self.N[tuple(node)] = 0
            return media_node
        else:
            return node
    
    def do_rollout(self, node):
        node_tuple = tuple(node)
        self.value[node_tuple] = self.oracle(node)
        [self.children[node_tuple], _] = self.find_children(node)
        self.N[node_tuple] += 1
    
    """
    Find child nodes from a specific root node
        
    :param node: root node (tuple of floats)
    :param num2change: number of units to perturb (default 2)
    :param max_retry: maximum retry attempts per position
    :return: list of child nodes (tuples)
    """
    def find_children(self, node, num2change=1, max_retry=100):

        class position_retry(Exception):
            pass
        all_tup = []
        n_units = len(node)
        positions = list(range(n_units))
        
        for pos_idx in range(len(positions)):
            try:
                retry_count = 0
                success = False
                
                while not success and retry_count < max_retry:
                    try:
                        tup = list(node)
                        tup = [round(x, 1) for x in tup]  # 确保精度
                        if num2change > 1:
                            main_index = positions[pos_idx]
                            other_indices = np.random.choice(
                                [i for i in range(n_units) if i != main_index],
                                num2change - 1,
                                replace=False
                            )
                            indexes_all = [main_index] + list(other_indices)
                        else:
                            indexes_all = [positions[pos_idx]]
                        
                        total_deviation = 0
                        
                        for idx in indexes_all:
                            flag = np.random.randint(0, 2)
                            
                            if tup[idx] <= 0.2 + 1e-5:
                                flag = 0 
                            elif tup[idx] >= 0.8 - 1e-5:
                                flag = 1 
                            
                            if flag == 0: 
                                max_increase = 0.8 - tup[idx]
                                steps = int(np.round(max_increase, 1) / 0.1)
                                if steps < 1:
                                    continue 
                            else:
                                max_decrease = tup[idx] - 0.2
                                steps = int(np.round(max_decrease, 1) / 0.1)
                                if steps < 1:
                                    continue
                            
                            path = np.random.randint(1, steps + 1)
                            difference = path * 0.1 * (1 if flag == 0 else -1)
                            
                            tup[idx] += difference
                            tup[idx] = round(tup[idx], 1)
                            total_deviation += difference
                        
                        indexes_possible = [i for i in range(n_units) if i not in indexes_all]
                        
                        allo_result = self.allocation(tup, indexes_possible, total_deviation)
                        if abs(total_deviation) > 1e-5 and not allo_result[1]:
                            retry_count += 1
                            continue
                        tup = allo_result[0]
                        
                        valid = True
                        for val in tup:
                            if val < 0.2 - 1e-5 or val > 0.8 + 1e-5:
                                valid = False
                                break
                        
                        if not valid:
                            retry_count += 1
                            continue 
                        
                        tup_tuple = tuple(round(x, 1) for x in tup)
                        self.value[tuple(tup)] = self.oracle(list(tup_tuple))
                        all_tup.append(tup_tuple)
                        success = True    
                            
                    except signal:
                        if not success and retry_count > max_retry:
                            print(f"warning: position {pos_idx} cannot generate valid design after {max_retry} tries")
                            raise position_retry 
                        else:
                            retry_count += 1   
            except position_retry:
                pos_idx -= 1
       
        all_values = [self.value[tup] for tup in all_tup]
        print(len(all_tup))
        
        return set(all_tup), all_values
    
    '''
    Allocating the deviation to the other cells
    
    :param tup: tuple required to be balancing (tuple of floats)
    :param indexes: possible positions to balance the deviation (default 2)
    :param deviation: total deviation
    :return: balanced tuple (tuple)
    '''
    def allocation(self, tup, indexes, deviation):
        if abs(deviation) > 1e-5:
            deviation = np.round(deviation, 1) ##########high risk
            path = int(abs(deviation)/0.1)
            
            splitnums = int(np.random.choice(np.arange(1, path + 1e-6)))
            tup = np.array(tup)
            if splitnums > len(tup):
                splitnums = len(tup)
            
            if splitnums == 1:
                whe = np.where(((tup - deviation) < 0.80001) &
                               ((tup - deviation) > 0.19999))[0]
                whe = np.intersect1d(whe, indexes)            
                if len(whe) == 0:
                    raise signal
                index1 = np.random.choice(whe)
                tup[index1] -= deviation * ( 1 if deviation > 0 else -1 )
            
            else:
                residual = np.random.choice(np.arange(1, path + 0.1), splitnums-1)
                
                for i in range(splitnums):
                    'calculate the splition paths'
                    if i == 0:
                        difference = 0.1*residual[i]
                    elif i == splitnums - 1:
                        difference = 0.1*(path - residual[i-1])
                    else:
                        difference = 0.1*(residual[i] - residual[i-1])
                        
                    'detect which units meet the requirements'
                    whe = np.where(((tup - deviation) < 0.80001) &
                               ((tup - deviation) > 0.19999))[0]  
                    whe = np.intersect1d(whe, indexes)
                    if len(whe) == 0:
                        raise signal
                    index1 = np.random.choice(whe)
                    tup[index1] -= difference * ( 1 if deviation > 0 else -1 )
        else:
            tup = tup
        
        return tup, True
    
    '''
    Calculate the specific stiffness/strength and check the designs
    
    :param x: input design
    :return: the property of input x
    '''
    def oracle(self, x):
        try:
            x_array = np.array(x)
            if x_array.ndim == 1:
                x_array = x_array.reshape(1, -1)
            
            averX = np.sum(x_array, axis=1) / self.n_dim
            voxelized_mat = Voxelization(x_array.reshape(-1, 3, 3, 3))
            
            pred = self.models.ensembled_prediction(
                np.array(voxelized_mat).reshape(len(voxelized_mat), 60, 60, 60, 1))
            pred = np.array(pred).flatten()
            
            pred_ratio = pred / averX
            
            out_of_range = (averX < self.target * 0.95) | (averX > self.target * 1.05)
            pred_ratio[out_of_range] = 1 / ((np.abs(averX[out_of_range] - self.target) * 10000) + 1) 
            
            if len(pred_ratio) == 1:
                return float(pred_ratio[0])
            return pred_ratio
        except Exception as e:
            print(f"Error in oracle: {e}")
            return 0.0
    
    
    """
    Find the Pareto frontier from a two-dimensional array.
    
    :param data: A two-dimensional numpy array where rows are points.
    :return: A numpy array with the points on the Pareto frontier.
    """
    def pareto_frontier(self, data):
        # Sort data by the first dimension (x)
        indices = np.argsort(data[:, 0])
        indices = indices[::-1]
        data_sorted = data[data[:, 0].argsort()]
        data_sorted = data_sorted[::-1]
        pareto_front = [data_sorted[0]]
        pareto_indices = [indices[0]]
        for i, point in enumerate(data_sorted[1:]):
            if point[1] > pareto_front[-1][1]:  
                pareto_front.append(point)
                pareto_indices.append(indices[i + 1])
        return np.array(pareto_indices)
    
    """
    Comprehensive consideration of property and distance,
    to make sure the diversity of the candidates
    
    :param sample_input: samples sampled through MCTS
    :param num: how many candidates to select
    :param plot_save: whether save the figures of visualization
    :return ind: the indexes of optimal candidates
    """
    def pareto_evaluation(self, sample_input, num, 
                          plot_save = False): #Euclidean distance + pred score #pareto front
        n_dim = self.n_dim
        all_input = self.input_x.reshape(-1,n_dim)
        sample_input = sample_input.reshape(-1,n_dim)
        sample_dist = []##nearest neighbor distance
        for i in sample_input:
            dist_temp=1e12
            for n in all_input:
                dist= np.linalg.norm(i - n)
                if dist < dist_temp:
                    dist_temp = round(dist,10)
            sample_dist.append(dist_temp)
        sample_dist = np.array(sample_dist)
        sample_score = np.array([self.value[tuple(inputx)] for inputx in sample_input])
        data = np.concatenate((sample_dist.reshape(-1,1),sample_score.reshape(-1,1)),axis=1)
        print(data.shape)
        pareto_front = self.pareto_frontier(data)
        while len(pareto_front) < num:
            remaining_data = np.delete(data, pareto_front, axis=0)
            remaining_indices = np.delete(np.arange(data.shape[0]), pareto_front)
            pareto_front2 = self.pareto_frontier(remaining_data)
            pareto_front = np.concatenate((pareto_front,remaining_indices[pareto_front2]))
        
        ind = np.random.choice(pareto_front,num,replace=False)
        if plot_save == True:
            plt.figure()
            plt.scatter(sample_dist,sample_score,label='all samples')
            plt.scatter(sample_dist[pareto_front],sample_score[pareto_front],label='pareto-front samples')
            plt.scatter(sample_dist[ind],sample_score[ind],label='selected samples')
            plt.title('distance VS score')
            plt.xlabel('distance')
            plt.ylabel('score')
            plt.legend()
            plt.show()
            plt.savefig(f'{self.model_folder}/distance VS score_{self.target}.png')
            ##############
            total = np.concatenate((all_input.reshape(-1,n_dim), sample_input.reshape(-1,n_dim)), axis=0)
            print('TSNE shape',total.shape)
            from sklearn.manifold import TSNE
            keep_dims = 2
            prp = 40
            tsne = TSNE(n_components=keep_dims,
                        perplexity=prp,
                        random_state=42,
                        n_iter=5000,
                        n_jobs=-1)
            total_tsne = tsne.fit_transform(total)
            total_tsne0 = total_tsne[:, 0]
            total_tsne1 = total_tsne[:, 1]
            print('TSNE done!')
            plt.figure()
            plt.scatter(total_tsne[:len(all_input), 0], total_tsne[:len(
                all_input), 1], color='black', label='initial data')
            plt.scatter(total_tsne[len(all_input):, 0], total_tsne[len(
                all_input):, 1], color='blue', label='sampled data')
            plt.scatter(total_tsne0[len(all_input)+ind], total_tsne1[len(
                all_input)+ind], color='green', label='top-sampled data')
            plt.title('TSNE ')
            plt.colorbar()
            plt.legend()
            plt.savefig(f'{self.model_folder}/TSNE_{self.target}.png')
            
            #################################################################
            from sklearn.decomposition import PCA
            "PCA"
            pca = PCA(n_components=2, whiten=True)
            total_pca = pca.fit_transform(total)
            plt.figure()
            plt.scatter(total_pca[:len(all_input), 0], total_pca[:len(
                all_input), 1], color='black', label='initial data')
            plt.scatter(total_pca[len(all_input):, 0], total_pca[len(
                all_input):, 1], color='blue', label='sampled data')
            plt.scatter(total_pca[len(all_input)+ind, 0], total_pca[len(all_input) +
                        ind, 1], color='green', label='top-sampled data')
            plt.title('PCA')
            plt.colorbar()
            plt.legend()
            plt.savefig(f'{self.model_folder}/PCAs_{self.target}.png')
        
        return ind
    
    """
    MCTS from a single initial point
    
    :param initial_X: initial point to start the MCTS
    :param initial_Y: property of the initial point
    :return Flag: whether there are valid designs
            X_next: result of a single tree
    """
    def single_tree(self, initial_X: np.ndarray, initial_Y: float) -> Tuple[bool, np.ndarray]:
        X = self.input_x.reshape(-1, self.n_dim)
        node_root = tuple(initial_X)  # 确保节点是可哈希类型
        
        boards = []
        
        for _ in tqdm(range(self.rollouts), desc="MCTS Rollouts"):
            self.do_rollout(node_root)
            
            boards.extend(self.children.get(node_root, []))
            
            node_root = self.choose(node_root)

            print('next root node score:', self.value[tuple(node_root)])
        
        unique_boards = set(boards)
        new_x = []
        
        for board in unique_boards:
            board_array = np.array(board)
            if not any(np.allclose(board_array, x) for x in X):
                new_x.append(board_array)
        
        if not new_x:
            return False, np.zeros((0, self.n_dim))
        
        new_x = np.array(new_x)
        print(f'unique number of boards: {len(new_x)}')
        
        'choose the samples, 5 top, 3 top pareto and 2 random'
        top_n = 5
        new_pred = np.array([self.value[tuple(x)] for x in new_x])
        
        top_indices = np.argsort(new_pred)[-top_n:]
        top_prediction = new_x[top_indices]
        
        other_indices = np.argsort(new_pred)[:-top_n]
        if len(other_indices) > 0:
            high_quality_mask = new_pred[other_indices] > np.max(new_pred) * 0.8
            other_high_quality = new_x[other_indices][high_quality_mask]       
            ind2 = self.pareto_evaluation(other_high_quality, 3)
            top_rank = other_high_quality[ind2]
        else:
            top_rank = np.zeros((0, self.n_dim))
        
        if len(new_x) > 2:
            X_rand = new_x[np.random.choice(len(new_x), 2, replace=False)]
        else:
            X_rand = new_x.copy()
        
        X_next = np.vstack((top_prediction, top_rank, X_rand))
        print(X_next.shape)
        return True, X_next

    '''
    Select five start points, and do MCTS rollouts for each of the start point
    :return top_X: optimal designs in a single run of MCTS rollouts
    '''
    def single_run(self):
        X = self.input_x.reshape(-1, self.n_dim)
        averX = [np.sum(f)/27 for f in X]
        averX = np.array(averX)
        
        Y = np.array(self.input_y)
        whe = np.where((averX > self.target * 1.05)|(averX < self.target * 0.95))
        Y[whe[0]] = 0
        properties = [Y[i]/averX[i] for i in range(len(X))]
        properties = np.array(properties)
        
        'remove those outside of the target range'
        valid_idx = np.where(properties > 1)[0]
        properties = properties[valid_idx]
        Y = Y[valid_idx]
        X = X[valid_idx]
        
        top_select = 4 #highest
        random_select = 1 #random
        ind = np.argsort(properties)[-top_select:]#####
        ind2 = np.random.choice(np.argsort(properties)[:-top_select], random_select, replace = False)
        ind = np.concatenate((ind,ind2))

        x_initial = X[ind]
        y_initial = Y[ind]
        property_initial = properties[ind]
        X_top=[]
        for i in range(top_select+random_select):
          current_X = x_initial[i]
          current_Y = y_initial[i]
          current_property = property_initial[i]
          self.exploration_weight = 1000
          print("current value:", current_property)
          print("current X:", current_X)
          Flag, x = self.single_tree(current_X, current_Y)
          if Flag:
              X_top.append(x)

        top_X = np.vstack(X_top)
        print(f'top selection number is {len(top_X)}')
        print(f'top selection are {top_X}')
        return top_X
    
    '''
    Repeat the MCTS rollouts, remove repeated designs
    and select the optimal candidates
    
    :param runtimes: repeat times for the MCTS rollouts
    :return candidates: candidate designs of this iteration
    :return scores: property of the candidates
    '''
    def Optimization(self, runtimes = 1):
        all_samples = []
        for i in range(runtimes):
            temp_samples = self.single_run()
            all_samples.append(temp_samples)
        
        all_samples = np.vstack(all_samples)
        all_samples = [list(x) for x in all_samples]
        all_samples = set(tuple(x) for x in all_samples)
        all_samples = np.array([np.array(f) for f in all_samples]).reshape(-1,3,3,3)
        
        aver_volume = np.array([np.sum(f)/self.n_dim for f in all_samples])
        ind_rational = np.where((aver_volume < self.target * 1.05) & (aver_volume > self.target * 0.95))[0]
        samples_rational = all_samples[ind_rational]
        aver_volume = aver_volume[ind_rational]
        mats_voxelized = Voxelization(samples_rational)
        predictions = self.models.ensembled_prediction(mats_voxelized)
        scores = np.array([predictions[i]/aver_volume[i] for i in range(len(aver_volume))])
        
        top_ind = self.pareto_evaluation(samples_rational, 
                                         self.num_select, plot_save=True)
        
        candidates = samples_rational[top_ind].reshape(-1,3,3,3)
        scores = scores[top_ind]
        
        return candidates, scores