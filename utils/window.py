from __future__ import annotations

import math
import numpy as np
from typing import TypeVar, Generic

_T = TypeVar('_T')

class Window(Generic[_T]):
  def __init__(self, window_length:int, window:list[_T]=None):
    self.window_length = window_length
    self.window:list[_T] = [] if window is None else window
  
  def push(self, data:_T):
    self.window.append(data)
    if len(self.window) > self.window_length:
      self.window.pop(0)

  def clear(self):
    self.window.clear()

  def first(self) -> _T:
    return self.window[0]

  def last(self) -> _T:
    return self.window[-1]

  def get(self, index:int) -> _T:
    return self.window[index]

  def head(self, length:int) -> Window[_T]:
    return Window[_T](self.window_length, self.window[:length])

  def tail(self, length:int) -> Window[_T]:
    return Window[_T](self.window_length, self.window[-length:])

  def capacity(self):
    return len(self.window)

  def empty(self):
    return len(self.window) == 0

  def full(self):
    return len(self.window) == self.window_length

  def sum(self, func:function=lambda x:x):
    return sum(map(func, self.window))

  def count(self, func:function=lambda x:x):
    return len(list(filter(lambda x:x == True, map(func, self.window))))

  def map(self, func:function=lambda x:x) -> Window:
    return Window(self.window_length, list(map(func, self.window)))

  def argmax(self) -> tuple[int, _T]:
    if self.capacity() == 0:
      return 0
    index = 0
    value = self.window[0]
    for i, v in enumerate(self.window):
      if v > value:
        value = v
        index = i
    return index, value

  def to_numpy(self):
    return np.array(self.window)
  
  def to_numpy_inside(self):
    return np.array([x.to_numpy() for x in self.window])
  
  def feature(self) -> list[float]:
    x = np.array(self.window)
    std = np.std(x)
    min = np.min(x)
    max = np.max(x)
    mean = np.mean(x)
    sc = np.mean((x - mean) ** 3) / pow(std, 3)
    ku = np.mean((x - mean) ** 4) / pow(std, 4)
    if math.isnan(ku):
      sc = 0
      ku = 0
    return [mean, min, max, sc, ku]
  
  def set_to_last_value(self):
    for i in range(self.window_length - 1):
      if hasattr(self.window[i], "assigned_by"):
        self.window[i].assigned_by(self.window[-1])
      else:
        self.window[i] = self.window[-1]
    return self