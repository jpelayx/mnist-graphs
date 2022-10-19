from rgb_slic import RGBSLIC

import torchvision.datasets as datasets 
import torchvision.transforms as T

class SuperPixelGraphCIFAR100(RGBSLIC):
    ds_name = 'CIFAR100'
    def get_ds_name(self):
        self.features.sort()
        return  './cifar100/{}-n{}-c{}-{}'.format('train' if self.train else 'test', 
                                                 self.n_segments, 
                                                 self.compactness,
                                                 '-'.join(self.features))
    def get_labels(self):
        return list(range(100))
    def load_data(self):
        cifar100_root = './cifar100/{}'.format('train' if self.train else 'test')
        return datasets.CIFAR100(cifar100_root, train=self.train, download=True, transform=T.ToTensor())