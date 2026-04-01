import os
import sys
import json
import copy
import time
import numpy as np
from typing import List, Tuple
from colorama import Fore, Back, Style


class TypeRingInterface:
    ''' The interface for TypeRing decoders.
    '''
    
    
    def __init__(self, phrases_path:str, n_phrase:int=10, gap:int=4):
        self.alphabet = 'qwertyuiopasdfghjklzxcvbnm'
        self.all_phrases = []
        for phrases in open(phrases_path, 'r'):
            self.all_phrases.append(phrases.strip() + ' ')
        self.n_phrase = n_phrase
        self.gap = gap
        self.finger_names = ['left index', 'left middle', 'left ring/pinky',
            'right index', 'right middle', 'right ring/pinky']
        self.reset()
        
    
    def _sample_phrases(self):
        while True:
            self.phrases = []
            for _ in range(self.n_phrase):
                self.phrases.append(np.random.choice(self.all_phrases))
            cnt = len(set(''.join(self.phrases)))
            if cnt >= 27: break
        
        
    def _delete(self, n:int):
        length = min(n, len(self.current_input))
        self.current_input = self.current_input[:-length]
        return length
        
    
    def _delete_char(self):
        if len(self.current_input) > 0 and self.current_input[-1] == '*':
            length = self._delete(1)
            if length > 0:
                self.fingers = self.fingers[:-length]
                self.positions = self.positions[:-length]
            self.candidates.clear()
            self.candidate_idx = 0
            self.delete_char_cnt += 1
            if not self.last_delete_char:
                self.delete_cnt += 1
            self.last_delete_char = True
    
    
    def _delete_word(self):
        total_length = 0
        while len(self.current_input) > 0 and self.current_input[-1] == ' ':
            length = self._delete(1)
            total_length += length
            if length > 0:
                self.fingers = self.fingers[:-length]
                self.positions = self.positions[:-length]
        while len(self.current_input) > 0 and self.current_input[-1] in self.alphabet+'*':
            length = self._delete(1)
            total_length += length
            if length > 0:
                self.fingers = self.fingers[:-length]
                self.positions = self.positions[:-length]
        self.candidates.clear()
        self.candidate_idx = 0
        self.last_delete_char = False
        if total_length > 0:
            self.delete_word_cnt += 1
            self.delete_cnt += 1
        
        
    def _select(self):
        if len(self.candidates) <= 0: return
        candidate = self.candidates[self.candidate_idx]
        self._delete(len(candidate)+1)
        self.current_input += f'{candidate} '
        self.candidates.clear()
        self.candidate_idx = 0
        
    
    def _next_phrase(self):
        self.committed_inputs.append(copy.deepcopy(self.current_input))
        self.committed_fingers.append(copy.deepcopy(self.fingers))
        self.committed_positions.append(copy.deepcopy(self.positions))
        self.phrase_idx += 1
        self.current_input = ''         # the current input phrase
        self.candidates.clear()         # word candidates to be selected
        self.candidate_idx = 0          # the index of seleted candidate
        self.fingers.clear()            # previous finger sequence
        self.positions.clear()          # previous touch positions
        if self.phrase_idx >= self.n_phrase:
            self.quit = True
        else: self.current_phrase = self.phrases[self.phrase_idx]
        
        
    def reset(self):
        self.quit = False
        self._sample_phrases()
        self.phrase_idx = 0
        self.current_phrase = self.phrases[0]
        self.current_input = ''         # the current input phrase
        self.candidates = []            # word candidates to be selected
        self.candidate_idx = 0          # the index of seleted candidate
        self.fingers = []               # previous finger sequence
        self.positions = []             # previous touch positions
        self.tic = None                 # start typing timestamp
        self.toc = None                 # end typing timestamp
        self.last_timestamp = None
        self.input_time = 0.0
        self.delete_time = 0.0
        self.select_time = 0.0
        # data to save
        self.delete_cnt = 0
        self.delete_char_cnt = 0
        self.delete_word_cnt = 0
        self.last_delete_char = False   # if last event is to delete a char, set to True
        self.switch_cnt = 0
        self.switch_row_cnt = 0         # select, right long tap
        self.switch_col_cnt = 0         # select, right tap
        self.last_switch = False        # if last event is to switch, set to True
        self.committed_inputs = []
        self.committed_fingers = []
        self.committed_positions = []
        
        
    def set_phrases(self, phrases:List[str]):
        self.phrases = copy.deepcopy(phrases)
        self.n_phrase = len(phrases)
        self.phrase_idx = 0
        self.current_phrase = self.phrases[0]
        
    
    def update_candidates(self, candidates:List[str]):
        self.candidates = copy.deepcopy(candidates)
        self.candidate_idx = 0
        if len(self.candidates) > 0:
            candidate = self.candidates[self.candidate_idx]
            self._delete(len(candidate)+1)
            self.current_input += f'{candidate} '
        
        
    def decode(self, event:Tuple[str, float], position:Tuple[float,float]=None):
        ''' Decode an event and update the input phrase.
        Args:
            event: the event name and timestamp,
                name in {'left tap', 'left thumb', 'left index',
                'left middle', 'left ring/pinky',
                'right tap', 'right thumb', 'right index',
                'right middle', 'right ring/pinky'}.
        Returns:
            anything used in higer level interface.
                e.g.: when a space is input, might need to decode top-N candidates.
        '''
        res = None
        gap = self.gap
        event_name, send_t, receive_t = event
        if position is None: position = (np.nan, np.nan)
                  
        if event_name in self.finger_names:    # character
            # if self.tic is None: self.tic = receive_t
            if self.phrase_idx == 0 and len(self.current_input) == 0:
                self.tic = receive_t
            self.toc = receive_t
            if self.last_timestamp is not None:
                self.input_time += (receive_t - self.last_timestamp)
            self._select()
            self.last_delete_char = False
            self.last_switch = False
            self.current_input += '*'
            self.fingers.append(event_name)
            self.positions.append(position)
        
        elif event_name == 'left thumb' or event_name == 'right thumb':   # space
            if self.last_timestamp is not None:
                self.input_time += (receive_t - self.last_timestamp)
            self._select()
            self.last_delete_char = False
            self.last_switch = False
            if len(self.current_input) > 0 and self.current_input[-1] == '*':
                # find the prefix word
                r = len(self.current_input) - 1
                while r >= 0 and self.current_input[r] == '*': r -= 1
                while r > 0 and self.current_input[r-1] == ' ': r -= 1
                l = r
                while l > 0 and self.current_input[l-1] in self.alphabet: l -= 1
                if l < r: pre = self.current_input[l:r]
                else: pre = None
                # find the target word
                l, r = len(self.current_input), len(self.current_input)
                while l > 0 and self.current_input[l-1] == '*': l -= 1
                if (l == 0 or self.current_phrase[l-1] == ' ') \
                    and (r < len(self.current_phrase) and self.current_phrase[r] == ' ') \
                    and (' ' not in self.current_phrase[l:r]):
                    target = self.current_phrase[l:r]
                else: target = None
                res = {'type': 'data', 'prefix': copy.deepcopy(pre),
                    'target': copy.deepcopy(target),
                    'fingers': copy.deepcopy(self.fingers[l:r]),
                    'positions': copy.deepcopy(self.positions[l:r])}
            if len(self.current_input) >= len(self.current_phrase) \
                    and self.current_input[-1] != '*':
                self._next_phrase()
            if len(self.current_input) > 0 and self.current_input[-1] != ' ':
                self.current_input += ' '
                self.fingers.append(event_name)
                self.positions.append(position)
            if self.quit: res = {'type': 'quit'}
         
        elif event_name == 'left tap':     # delete
            if self.last_timestamp is not None:
                self.delete_time += (receive_t - self.last_timestamp)
            if len(self.current_input) > 0:
                if self.current_input[-1] == '*':
                    self._delete_char()
                else: self._delete_word()
                self.last_switch = False
        
        elif event_name == 'right tap':    # switch column
            if self.last_timestamp is not None:
                self.select_time += (receive_t - self.last_timestamp)
            self.last_delete_char = False
            if len(self.candidates) > 0:
                base = self.candidate_idx - self.candidate_idx % gap
                cycle = min(len(self.candidates) - base, gap)
                self.candidate_idx = base + (self.candidate_idx + 1) % cycle                
                candidate = self.candidates[self.candidate_idx]
                self._delete(len(candidate)+1)
                self.current_input += f'{candidate} '
                self.switch_col_cnt += 1
                if not self.last_switch:
                    self.last_switch = True
                    self.switch_cnt += 1
                
        elif event_name == 'left long tap':
            if self.last_timestamp is not None:
                self.delete_time += (receive_t - self.last_timestamp)
            if len(self.current_input) > 0:
                self._delete_word()
                self.last_switch = False
        
        elif event_name == 'right long tap':    # switch row
            if self.last_timestamp is not None:
                self.select_time += (receive_t - self.last_timestamp)
            self.last_delete_char = False
            if len(self.candidates) > 0:
                self.candidate_idx = self.candidate_idx + gap
                if self.candidate_idx >= len(self.candidates):
                    self.candidate_idx = self.candidate_idx % gap
                candidate = self.candidates[self.candidate_idx]
                self._delete(len(candidate)+1)
                self.current_input += f'{candidate} '
                self.switch_row_cnt += 1
                if not self.last_switch:
                    self.last_switch = True
                    self.switch_cnt += 1
                    
        self.last_timestamp = receive_t
        return res
    
    
    def save_data(self):
        phrases = self.phrases
        length = sum(len(p) for p in phrases)
        time_cost = self.toc - self.tic
        wpm = 60 * 0.2*length / time_cost
        word_cnt, char_cnt = 0, 0
        for phrase in phrases:
            word_cnt += len(phrase.strip().split(' '))
            char_cnt += len(phrase.strip()) + 1
        uncorrected_word_cnt = 0
        for phrase, committed in zip(phrases, self.committed_inputs):
            p_words = phrase.strip().split(' ')
            c_words = committed.strip().split(' ')
            for i in range(min(len(p_words), len(c_words))):
                if p_words[i] != c_words[i]: uncorrected_word_cnt += 1
            uncorrected_word_cnt += abs(len(p_words) - len(c_words))
        uer = (uncorrected_word_cnt + self.delete_cnt + self.switch_cnt) \
            / (word_cnt + self.delete_cnt)
        cer = uncorrected_word_cnt / word_cnt
        print("\033[H\033[J", end="")
        res = {'phrases': phrases,
            'tic': self.tic, 'toc': self.toc,
            'input_time': self.input_time,
            'delete_time': self.delete_time,
            'select_time': self.select_time,
            'wpm': wpm, 'uer': uer, 'cer': cer,
            'word_cnt': word_cnt,
            'char_cnt': char_cnt,
            'uncorrected_word_cnt': uncorrected_word_cnt,
            'delete_cnt': self.delete_cnt,
            'delete_word_cnt': self.delete_word_cnt,
            'delete_char_cnt': self.delete_char_cnt,
            'switch_cnt': self.switch_cnt,
            'switch_col_cnt': self.switch_col_cnt,
            'switch_row_cnt': self.switch_row_cnt,
            'committed_inputs': self.committed_inputs,
            'committed_fingers': self.committed_fingers,
            'committed_positions': self.committed_positions}
        return res
        
        
    def print(self):
        ''' Print the current state to terminal.
        '''
        print("\033[H\033[J", end="")
        print(f'{self.current_phrase}\n{self.current_input}')
        print('', flush=True)
        if len(self.candidates) > 0:
            for i, candidate in enumerate(self.candidates):
                end = '\n' if i % 4 == 3 else ' '
                if i == self.candidate_idx:
                    print(f'{Fore.BLACK}{Back.WHITE}{Style.BRIGHT}'
                        + f'{candidate}{Style.RESET_ALL}', end=end)
                else: print(f'{candidate}', end=end)
            print('', flush=True)
