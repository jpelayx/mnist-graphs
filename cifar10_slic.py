from rgb_slic import RGBSLIC

import torchvision.datasets as datasets 
import torchvision.transforms as T

class SuperPixelGraphCIFAR10(RGBSLIC):
    ds_name = 'CIFAR10'
    num_classes = 10
    def get_ds_name(self):
        self.features.sort()
        return  './cifar10/{}-n{}-c{}-{}'.format('train' if self.train else 'test', 
                                                 self.n_segments, 
                                                 self.compactness,
                                                 '-'.join(self.features))
    def get_labels(self):
        return list(range(10))
    def load_data(self):
        cifar10_root = './cifar10/{}'.format('train' if self.train else 'test')
        return datasets.CIFAR10(cifar10_root, train=self.train, download=True, transform=T.ToTensor())