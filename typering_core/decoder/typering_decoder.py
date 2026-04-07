import os
import copy
import json
import pickle
import numpy as np
import torch
import torch.nn.functional as F
from typing import List, Tuple, Union, Dict


class TypeRingIMUDecoder:
    ''' The IMU only TypeRing decoder.
    '''
    
    
    def __init__(self, char2finger_path:str, corpus_path:str,
            unigrams_path:str, bigrams_path:str):
        self.char2finger = json.load(open(char2finger_path, 'r'))
        self.corpus = pickle.load(open(corpus_path, 'rb'))
        self.unigrams = pickle.load(open(unigrams_path, 'rb'))
        self.bigrams = pickle.load(open(bigrams_path, 'rb'))
        
        
    def set_touch_model(self, char2finger:Union[str,dict]):
        if isinstance(char2finger, str):
            self.char2finger = json.load(open(char2finger, 'r'))
        else: self.char2finger = copy.deepcopy(char2finger)
        
    
    def decode(self, pre:str, fingers:List[str], n_candidate:int=5):
        ''' Decode current top-N input word.
        Args:
            pre: str, the previous input word, could be None.
            fingers: List[str], a list of finger labels.
            n_candidate: int, the number of top candidates to return.
        Returns:
            List[str], a list of top-N candidates, could be None.
        '''
        corpus = self.corpus
        char2finger = self.char2finger
        unigrams = self.unigrams
        bigrams = self.bigrams
        
        candidates = []     # (word, prob)
        for word in corpus:
            if len(word) != len(fingers): continue
            
            # touch model
            prob = 1.0
            for c, finger in zip(word, fingers):
                prob *= char2finger[c][finger]
            
            # language model
            if pre is None or pre not in corpus:     # unigrams
                prob *= unigrams.get(word, 1e-5)
            elif pre in bigrams:    # bigrams
                prob *= bigrams.get(pre, 1e-5).get(word, 1e-5)
            else: prob *= 1e-5
            candidates.append((word, prob))
            
        candidates = sorted(candidates, key=lambda x: x[1], reverse=True)
        return [c[0] for c in candidates[:n_candidate]] if len(candidates) > 0 else None
    
    
class TypeRingDecoder:
    ''' The full TypeRing decoder.
    '''
    
    def __init__(self, char2finger_path:str, touch_cloud_path:str,
            touch_cloud_rlt_path:str, corpus_path:str, unigrams_path:str,
            bigrams_path:str, abs_weight:float=0.4):
        self.char2finger = json.load(open(char2finger_path, 'r'))
        self.touch_cloud = pickle.load(open(touch_cloud_path, 'rb'))
        self.touch_cloud_rlt = pickle.load(open(touch_cloud_rlt_path, 'rb'))
        self.corpus = pickle.load(open(corpus_path, 'rb'))
        self.unigrams = pickle.load(open(unigrams_path, 'rb'))
        self.bigrams = pickle.load(open(bigrams_path, 'rb'))
        self.abs_weight = abs_weight
        
    
    def set_touch_model(self, char2finger:Union[str,dict], touch_cloud:Union[str,dict],
            touch_cloud_rlt:Union[str,dict]):
        if isinstance(char2finger, str):
            self.char2finger = json.load(open(char2finger, 'r'))
        else: self.char2finger = copy.deepcopy(char2finger)
        if isinstance(touch_cloud, str):
            self.touch_cloud = pickle.load(open(touch_cloud, 'rb'))
        else: self.touch_cloud = copy.deepcopy(touch_cloud)
        if isinstance(touch_cloud_rlt, str):
            self.touch_cloud_rlt = pickle.load(open(touch_cloud_rlt, 'rb'))
        else: self.touch_cloud_rlt = copy.deepcopy(touch_cloud_rlt)
        
    
    def decode(self, pre:str, fingers:List[str], positions:np.ndarray, n:int=5):
        ''' Decode current top-N input word.
        Args:
            pre: str, the previous input word, could be None.
            fingers: List[str], a list of finger labels.
            positions: np.ndarray, [N, 2], a 2D array of finger positions.
            n: int, the number of top candidates to return.
        Returns:
            List[str], a list of top-N candidates, could be None.
        '''
        corpus = self.corpus
        char2finger = self.char2finger
        touch_cloud = self.touch_cloud
        touch_cloud_rlt = self.touch_cloud_rlt
        unigrams = self.unigrams
        bigrams = self.bigrams
        abs_weight = self.abs_weight
        rlt_weight = 1.0 - abs_weight
        
        candidates = []     # (word, prob)
        for word in corpus:
            if len(word) != len(fingers): continue
            
            # touch model
            prob = 1.0
            for i in range(len(word)):
                c2, finger, position2 = word[i], fingers[i], positions[i]
                if i == 0:  # abs model only
                    if not np.isnan(position2[0]):
                        mu, _, scale, inv = touch_cloud[c2]
                        dx = position2[0] - mu
                        prob *= scale * np.exp(-0.5*dx*dx*inv)
                    else: prob *= char2finger[c2][finger]
                else:  # abs + rlt model
                    c1, position1 = word[i-1], positions[i-1]
                    if c2 in touch_cloud_rlt[c1] and not np.isnan(position2[0]) \
                            and not np.isnan(position1[0]):
                        mu, _, scale, inv = touch_cloud[c2]
                        dx = position2[0] - mu
                        mu, _, scale2, inv2 = touch_cloud_rlt[c1][c2]
                        dx2 = position2[0] - position1[0] - mu
                        prob *= abs_weight * (scale * np.exp(-0.5*dx*dx*inv)) \
                            + rlt_weight * (scale2 * np.exp(-0.5*dx2*dx2*inv2))
                    elif not np.isnan(position2[0]):
                        mu, _, scale, inv = touch_cloud[c2]
                        dx = position2[0] - mu
                        prob *= scale * np.exp(-0.5*dx*dx*inv)
                    else: prob *= char2finger[c2][finger]
                
            # language model
            if pre is None or pre not in corpus:     # unigrams
                prob *= unigrams.get(word, 1e-5)
            elif pre in bigrams:    # bigrams
                prob *= bigrams.get(pre, 1e-5).get(word, 1e-5)
            else: prob *= 1e-5
            candidates.append((word, prob))
            
        candidates = sorted(candidates, key=lambda x: x[1], reverse=True)
        return [c[0] for c in candidates[:n]] if len(candidates) > 0 else None
