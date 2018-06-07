import numpy as np
import os.path
from abc import ABCMeta, abstractmethod


class WordEmbedding(metaclass=ABCMeta):

    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def get_embedding_matrix(self):
        pass

    @abstractmethod
    def get_dict(self):
        pass

    @abstractmethod
    def load(self):
        pass
