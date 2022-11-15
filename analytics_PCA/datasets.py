
from sklearn.model_selection import train_test_split
from torchvision import datasets
from typing import Callable, Optional


class CIFAR10Red(datasets.CIFAR10):

    def __init__(self,
                 root: str,
                 train: bool = True,
                 transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None,
                 download: bool = False,
                 portion=0.1,
                 ) -> None:
        
        super().__init__(root, train, transform, target_transform,
                         download)
        self.portion = portion
        
        self.reduce_dataset()
    
    
    def reduce_dataset(self):
    
        self.data, __, self.targets, __ = \
            train_test_split(self.data, self.targets, train_size=self.portion,
                             random_state=69, stratify=self.targets)

