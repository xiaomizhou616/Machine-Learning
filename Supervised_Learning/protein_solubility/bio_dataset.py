from torch.utils.data import Dataset
import numpy as np

class BioDataset(Dataset):
    def __init__(self, csv_file, split='train', percentage=1., is_entire=True, fold=0, val=False):
        self.split = split
        
        if self.split not in ['train', 'test']:
            raise ValueError('Wrong split entered! Please use split="train", or split="test"')
            
        if percentage < 0. or percentage > 1.:
            raise ValueError('Wrong percentage entered! "percentage" should be within [0,1].')
        
        if fold not in [ii for ii in range(10)]:
            raise ValueError('Wrong fold entered! "fold" should be an integer from 0 to 9.')
    
        data_f = open(csv_file, 'r')
        contents = data_f.read().splitlines()[1:]
        data_f.close()
        contents = [content.split(',')[1:] for content in contents]
        contents = [[float(entry) for entry in content] for content in contents]
        
        if self.split == 'test':
            contents = contents[2200:]
        elif self.split == 'train':
            contents = contents[:int(2200*percentage)]
            if not is_entire:
                fold_size = len(contents) // 10
                if val:
                    contents = contents[fold*fold_size:(fold+1)*fold_size]
                else:
                    contents = contents[:fold*fold_size] + contents[(fold+1)*fold_size:]
            
        self.contents = np.array(contents, 'float32')[:,:-1]
        self.labels = np.array([contents[ii][-1]>=0.5 for ii in range(len(contents))], 'float32')
        
    def __len__(self):
        return len(self.contents)
        
    def __getitem__(self, idx):
        return self.contents[idx], self.labels[idx]