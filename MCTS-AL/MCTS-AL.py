from models import models
from Voxelization import Voxelization
from data_extraction import data_extraction
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import random
import os

'''
Sampling through MCTS

:prarm rd_num: iteration nubmer in the AL
        target: average volume fraction to maintain
        num_select: best candidates selected in single iteration
        n_dim: dimensionality of problem
        rollouts: number of rollouts in the MCTS sampling
        retrain: if retrain the CNN models or just load them
'''
class MCTS:
    def __init__(self, rd_num, target = 0.5, num_select = 20,
                 n_dim = 27, rollouts = 100, retrain = True):
        self.round_name = f'Round{rd_num}'
        self.model_folder = os.getcwd() + f'/models/{self.round_name}'
        # self.data_folder = os.getcwd() + f'/data/{self.round_name}'
        self.input_x, self.input_y = data_extraction(rd_num)
        self.rd_num = rd_num
        self.target = target
        self.num_select = num_select
        self.models = models(self.input_x, self.input_y, rd_num)
        self.n_dim = n_dim
        self.rollouts = rollouts
        
        if not retrain:
            try:
                self.models.load_models()
            except:
                self.models.ensembled_training()
        
        else:
            self.models.ensembled_training()      
    

    '''
    Find child nodes from a specific root node
    
    :params root: root node 
    :return all_children: child nodes
            all_value: predictions of the child nodes
    '''
    def do_rollout(self, root):
        positions = [p for p in range(0, len(root))]
        increment = 0.1
        
        all_children = []
        for position in positions:
            tup = list(root)
            
            flip = random.randint(0,5)
            if flip == 0:
              tup[position] += increment
            elif flip == 1:
                tup[position] -= increment
            elif flip == 2:
              for i in range(int(self.n_dim/5)):
                position_2 = random.randint(0, len(tup)-1)
                tup[position_2] = random.randint(1, 8)/10
            elif flip ==3:
              for i in range(int(self.n_dim/10)):
                position_2 = random.randint(0, len(tup)-1)
                tup[position_2] = random.randint(1, 8)/10
            else:
              tup[position] = random.randint(1, 8)/10

            tup[position] = round(tup[position],2)
            tup = np.array(tup)
            
            whe1 = np.where(tup<0.2)
            whe2 = np.where(tup>0.8)
            tup[whe1[0]] = 0.2
            tup[whe2[0]] = 0.8
            all_children.append(tup)

        all_value = self.oracle(all_children)
        
        return all_children, all_value
    
    '''
    Calculate the specific stiffness/strength and check the designs
    
    :param x: input design
    :return pred2: the property of input x
    '''    
    def oracle(self, x):
        try:
            averX = np.round([np.sum(f)/27 for f in x], 2)
            averX = np.array(averX)
            voxelized_mat = Voxelization(np.array(x).reshape(-1,3,3,3))
            whe = np.where((averX > self.target + 0.1)|(averX < self.target - 0.1))
            pred2 = self.models.ensembled_prediction(np.array(voxelized_mat).reshape(len(voxelized_mat),60,60,60,1))
            pred2 = np.array(pred2).reshape(len(voxelized_mat))
            pred2 = [pred2[i]/averX[i] for i in range(len(pred2))]
            
            ########new
            for i in whe[0]:
                pred2[i] = 1/(abs(averX[i] - self.target)*10000)
        except:
           voxelized_mat = Voxelization(np.array(x).reshape(1,3,3,3))
           averX = np.round([np.sum(f)/27 for f in x], 2)
           averX = np.array(averX)
           pred2 = self.models.ensembled_prediction(np.array(voxelized_mat).reshape(1,60,60,60,1))
           pred2 = np.array(pred2).reshape(1)
           pred2 = [pred2[i]/averX[i] for i in range(len(pred2))]

           if averX < self.target - 0.1 or averX > self.target + 0.1:
                pred2 = 1/(abs(averX - self.target)*10000)
                
        pred2 = np.array(pred2)
        print('average V:',averX)
        print('relative S:',pred2)
        return pred2
    
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
            num: how many candidates to select
            plot_save: whether save the figures of visualization
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
        sample_score = self.oracle(sample_input)
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
           initial_Y: property of the initial point
    :return Flag: whether there are valid designs
            X_next: rerult of a single tree
    """
    def single_tree(self, initial_X, initial_Y):
        X = self.input_x.reshape(-1, self.n_dim)
        Flag = True
        
        boards = []
        #conduct 100 rollouts
        node_root = initial_X
        for i in tqdm(range(0, self.rollouts, 1)):
            all_nodes, all_values = self.do_rollout(node_root)
            node_root = all_nodes[np.argmax(all_values)]
            boards.append(all_nodes)
        
        # make sure every proposed combination is unique and not in original combinations
        new_x = []
        new_pred = []
        boards = np.array(boards).reshape(-1, self.n_dim)
        boards = np.unique(boards, axis=0)
        predictions = self.oracle(boards)
        
        for x, prediction in zip(boards, predictions):
          temp_x = np.array(x)
          same = np.all(temp_x == X.reshape(len(X), self.n_dim), axis=1)
          has_true = any(same)
          if has_true == False:
            new_pred.append(prediction)
            new_x.append(temp_x)
        new_x= np.array(new_x)
        new_pred = np.array(new_pred)
        print(f'Unique number of designs: {len(new_x)}')
        
        # top samples
        top_n = 5
        ind = np.argsort(new_pred)
        top_prediction =  new_x[ind[-top_n:]]
        
        # top ranking (by pareto front) samples
        new_x2 = new_x[ind[:-top_n]]
        sample_score = new_pred[ind[:-top_n]]
        whe=np.where(sample_score > max(sample_score) * 0.8)[0]
        ###############################
        whe1 = np.where(sample_score > 0)[0]
        whe = np.intersect1d(whe, whe1)
        new_x3 = new_x2[whe]
        if len(whe) == 0:
            Flag = False
            new_x3 = np.zeros([27,1])
            X_next = new_x3
        
        elif not len(whe) == 0:
            print("Number of rational designs:", len(whe))
            print(new_x3)    
            ind2 = self.pareto_evaluation(X, new_x3, 3)
            top_rank = new_x3[ind2]

            # random samples
            X_rand = [new_x2[random.randint(0, len(new_x2)-1)] for i in range(2)]
        
            X_next = np.concatenate([top_prediction, top_rank, X_rand])
            
        return Flag, X_next

    '''
    Select five start points, and do MCTS rollouts for each of the start point
    :return top_X: optimal designs in a single run of MCTS rollouts
    '''
    def single_run(self):
        X = self.input_x.reshape(-1, self.n_dim)
        averX = [np.sum(f)/27 for f in X]
        averX = np.array(averX)
        
        Y = np.array(self.input_y)
        whe = np.where((averX > self.target + 0.1)|(averX < self.target - 0.1))
        Y[whe[0]] = 0
        properties = [Y[i]/averX[i] for i in range(len(X))]
        properties = np.array(properties)
        
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
          print("current value:", current_property)
          print("current X:", current_X)
          Flag, x = self.single_tree(current_X, current_Y)
          if Flag:
              X_top.append(x)

        top_X = np.vstack(X_top)
        print(f'top selection are {top_X}')
        return top_X
    
    '''
    Repeat the MCTS rollouts, remove repeated designs
    and select the optimal candidates
    
    :param runtimes: repeat times for the MCTS rollouts
    :return candidates: candidate designs of this iteration
            scores: property of the candidates
    '''
    def Optimization(self, runtimes = 5):
        all_samples = []
        for i in range(runtimes):
            temp_samples = self.single_run()
            all_samples.append(temp_samples)
        
        all_samples = np.vstack(all_samples)
        all_samples = [list(x) for x in all_samples]
        all_samples = set(tuple(x) for x in all_samples)
        all_samples = np.array([np.array(f) for f in all_samples]).reshape(-1,3,3,3)
        
        aver_volume = np.array([np.sum(f)/self.n_dim for f in all_samples])
        ind_rational = np.where((aver_volume < self.target + 0.1) & (aver_volume > self.target - 0.1))[0]
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
    
    
MCTS = MCTS(1, retrain=False)
candidates, scores = MCTS.Optimization()


