from color_slic import ColorSLIC

import torchvision.datasets as datasets 
import torchvision.transforms as T

class SuperPixelGraphSTL10(ColorSLIC):
    ds_name = 'STL10'
    def get_ds_name(self):
        return  './stl10/{}-n{}-c{}'.format('train' if self.train else 'test', 
                                            self.n_segments, 
                                            self.compactness)
    def get_ds_name_with_features(self):
        self.features.sort()
        return  './stl10/{}-n{}-c{}-{}'.format('train' if self.train else 'test', 
                                               self.n_segments, 
                                               self.compactness,
                                               '-'.join(self.features))

    def get_labels(self):
        return list(range(10))
    def load_data(self):
        split = 'train' if self.train else 'test'
        stl10_root = f'./stl10/{split}'
        return datasets.STL10(stl10_root, split=split, download=True, transform=T.ToTensor())