from torch.utils.data import Dataset
import numpy as np

def preprocessing(csv_file):
    data_f = open(csv_file, 'r')
    contents = data_f.read().splitlines()[1:]
    data_f.close()
    contents = [content.split(',')[1:] for content in contents]
    data_type = ['n','c','c','c','c','c','c','c','c','c','n','n','n','n','c','n','n','n','n','n']
    
    col = [contents[ii][0] for ii in range(len(contents))]
    col = [float(col[ii]) for ii in range(len(col))]
    encode = np.array(col, dtype='float32')
    encode = (encode - encode[:21600].min())/(encode[:21600].max()- encode[:21600].min())
    encode = encode[...,None]
    
    for cc in range(1,20):
        if data_type[cc] == 'c':
            col = [contents[ii][cc] for ii in range(len(contents))]
            set_col = list(set(col))
            encode_col = [[float(col[ii]==set_col[jj]) for jj in range(len(set_col))] for ii in range(len(col))]
            encode_col = np.array(encode_col, dtype='float32')
            encode = np.concatenate((encode, encode_col), axis=1)
        elif data_type[cc] == 'n':
            col = [contents[ii][cc] for ii in range(len(contents))]
            col = [float(col[ii]) for ii in range(len(col))]
            encode_col = np.array(col, dtype='float32')
            encode_col = (encode_col - encode_col[:21600].min())/(encode_col[:21600].max()-encode_col[:21600].min())
            encode_col = encode_col[...,None]
            encode = np.concatenate((encode, encode_col), axis=1)
    
    col = [contents[ii][20] for ii in range(len(contents))]
    encode_col = [[float(col[ii]=='yes')] for ii in range(len(col))]
    encode_col = np.array(encode_col, dtype='float32')
    encode = np.concatenate((encode, encode_col), axis=1)
    return encode

class BankDataset(Dataset):
    def __init__(self, _contents, split='train', percentage=1., is_entire=True, fold=0, val=False):
        self.split = split
        
        if self.split not in ['train', 'test']:
            raise ValueError('Wrong split entered! Please use split="train", or split="test"')
            
        if percentage < 0. or percentage > 1.:
            raise ValueError('Wrong percentage entered! "percentage" should be within [0,1].')
        
        if fold not in [ii for ii in range(10)]:
            raise ValueError('Wrong fold entered! "fold" should be an integer from 0 to 9.')
            
        if self.split == 'test':
            contents = _contents[21600:]
        elif self.split == 'train':
            contents = _contents[:int(21600*percentage)]
            if not is_entire:
                fold_size = len(contents) // 10
                if val:
                    contents = contents[fold*fold_size:(fold+1)*fold_size]
                else:
                    contents = np.concatenate((contents[:fold*fold_size],contents[(fold+1)*fold_size:]), axis=0)
            
        self.contents = contents[:,:-1]
        self.labels = contents[:,-1]
        
    def __len__(self):
        return len(self.contents)
        
    def __getitem__(self, idx):
        return self.contents[idx], self.labels[idx]