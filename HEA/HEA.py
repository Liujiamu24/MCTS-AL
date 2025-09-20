# -*- coding: utf-8 -*-
"""
Created on Fri Aug 15 12:45:47 2025

@author: 30865
"""
from models import models
from Voxelization import Voxelization
from data_extraction import data_extraction
from tqdm import tqdm
from typing import Any, Set, Dict
from collections import defaultdict
from dataclasses import dataclass, field
import numpy as np
import matplotlib.pyplot as plt
import random
import os
import math

'''
Active optimization through HEA

:param rd_num: iteration number in the AL
        retrain: if retrain the CNN models or just load them
        pm: possibility of mutation
        M: population size
        target: average volume fraction to maintain
        cand_num: best candidates selected in single iteration
        exploration_weight: a factor to measure the exploration and exploitation
        
        F: fitness of each individual
        score: score of each individual
        pc: choosing possibility as parents of each individual
        A: total age of each individual
'''
@dataclass
class HE:
    rd_num: int  
    retrain: bool  
    pm: float = 0.5
    M: int = 100
    target: float = 0.5
    cand_num: int = 20
    exploreation_weight: int = 100
    
    F: Dict[Any, float] = field(default_factory=lambda: defaultdict(float))
    score: Dict[Any, float] = field(default_factory=lambda: defaultdict(float))
    pc: Dict[Any, float] = field(default_factory=lambda: defaultdict(float))
    A: Dict[Any, int] = field(default_factory=lambda: defaultdict(int))
    def __post_init__(self):
        self.round_name = f'Round{self.rd_num}'
        self.model_folder = os.getcwd() + f'/models/{self.round_name}'
        self.initial_population, self.scores = data_extraction(self.rd_num)
        self.models = models(self.initial_population, self.scores, self.rd_num)
        self.population = []
        self.generation = 1
        
        'allocate fitness for the initial population'
        for i in range(len(self.initial_population)):
            pop = self.initial_population[i]
            pop = pop.reshape(-1)
            
            if tuple(pop) not in self.F:
                self.score[tuple(pop)] = self.scores[i]
                pop = pop.reshape(-1)
                averx = np.mean(pop)
                # print(averx)
                
                if (averx > self.target * 1.05) or (averx < self.target * 0.95):
                    score = 1 / ((np.abs(averx - self.target) * 10000) + 1) 
                else:
                    score = self.score[tuple(pop)] / averx
                    
                self.score[tuple(pop)] = score
                self.F[tuple(pop)] = score + self.exploreation_weight * math.sqrt(
                                            math.log(self.generation) / (self.A[tuple(pop)] + 1))
            
            self.population.append(tuple(pop))
        
        total_F = sum([self.F[pop] for pop in self.population])
        for pop in self.population:
            self.pc[pop] = self.F[pop] / total_F
        
        'Train the model or load the model'
        if not self.retrain:
            try:
                self.models.load_models()
            except:
                self.models.ensembled_training()
        else:
            self.models.ensembled_training()
        
        self.models.load_models()
    
    def group_update(self):
        # Store current population before update
        old_population = set(self.population)
        
        # Sort and select the top M individuals
        sorted_population = sorted(self.population, key=lambda x: self.F[x])
        new_population = sorted_population[-self.M:]
        
        # Update ages: +1 for survivors, 0 for eliminated individuals
        for individual in old_population:
            if individual in new_population:
                self.A[individual] += 1  # Increase age for survivors
            else:
                self.A[individual] = 0  # Reset age for eliminated individuals
        
        self.population = new_population
        total_F = sum([self.F[pop] for pop in self.population])
        for pop in self.population:
            self.pc[pop] = self.F[pop] / total_F
            
    def hybridization(self):
        'hybridization'
        max_attempts = 100  # Maximum attempts to generate a unique child
        attempt = 0
        
        while attempt < max_attempts:
            chosen_weights = [self.pc[individual] for individual in self.population]
            parents = random.choices(self.population, weights=chosen_weights, k=2)
            parent1 = np.array(parents[0])
            parent2 = np.array(parents[1])
            r = random.random()
            
            child = r * parent1 + (1 - r) * parent2
            
            'mutation'
            if random.random() < self.pm:
                mut = np.random.normal(0, 0.5, len(child))
                child += mut
            
            # Ensure each element is between 0.2 and 0.8
            child = np.clip(child, 0.2, 0.8)
            
            child = np.round(child, 1)
            child_tuple = tuple(child)
            
            # Check if child already exists in population
            if child_tuple not in self.population:
                break
                
            attempt += 1
        
        if attempt == max_attempts:
            # If we can't generate a unique child after max_attempts, skip this hybridization
            print(f"Warning: Could not generate unique child after {max_attempts} attempts")
            return
        
        self.population.append(child_tuple)
        
        # Initialize age for new child
        self.A[child_tuple] = 0
        
        voxelized_mat = Voxelization(child.reshape(-1, 3, 3, 3))
        
        'child fitness'
        score = self.models.ensembled_prediction(
                    np.array(voxelized_mat).reshape(len(voxelized_mat), 60, 60, 60, 1))
        score = np.array(score).flatten()
        averx = np.mean(child)
           
        if (averx > self.target * 1.05) or (averx < self.target * 0.95):
            score = 1 / ((np.abs(averx - self.target) * 10000) + 1) 
        else:
            score = score / averx
        
        self.score[child_tuple] = score
        self.F[child_tuple] = score + self.exploreation_weight * math.sqrt(
                                    math.log(self.generation) / (self.A[child_tuple] + 1))   
         
    def evolution(self):
        """executing the evolution"""
        generations = 500
        
        for _ in tqdm(range(generations), desc="Reproduction"):
            self.generation += 1
            
            'generate 20 new individuals in a year'
            for _ in range(50):
                self.hybridization()
            self.group_update()
            
            max_pop = max(self.population, key=lambda x: self.score[x])
            print('current best score: ', self.score[max_pop])
        
        # return the best candidates
        sorted_pop = sorted(self.population, key=lambda x: self.score[x])
        best_candidates = sorted_pop[-self.cand_num:]
        
        return best_candidates, [self.score[c] for c in best_candidates]
            
            
        
    
    
    
    
    
    
    
    