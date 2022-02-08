
from sklearn.semi_supervised import LabelSpreading
import torch.optim as optim
import torchvision
from torch.utils import data
import torchvision.transforms as transforms
import pickle
import pandas
import torch

class SpectraDataset(data.Dataset):
    def __init__(self,pickle_path,spectra_size,samples_per_class):
        self.spectra_data = pickle.load(open(pickle_path,"rb"))
        print("original ",pickle_path," dataset size: ",len(self.spectra_data))
        self.spectra_size = spectra_size #50.000
        self.spectra_data = self.balance(samples_per_class)
        #self.spectra_flattened = 

    def __len__(self):
        return len(self.spectra_data)

    def __getitem__(self, index):
        moz_indexes = self.spectra_data[index][0]
        abundance = torch.FloatTensor(self.spectra_data[index][1])
        label = self.spectra_data[index][5]

        moz_indexes_2D = torch.LongTensor([[0]*len(moz_indexes),moz_indexes]) # moz_indexes in 2D array
        torch_spectra = torch.squeeze(torch.sparse_coo_tensor(moz_indexes_2D,abundance,torch.Size([1,self.spectra_size])).to_dense()) # the sparse_coo_tensor is returning an extra dimension we dont need N x 1 x 50.000
        X = torch_spectra
        Y = label

        return X,Y

    def balance(self,samples_per_class):
        filtered_spectra = []
        modified_counter = 0
        unmodified_counter = 0
        for i in range(len(self.spectra_data)):
            if self.spectra_data[i][5] == 0 and unmodified_counter < samples_per_class:
                filtered_spectra.append(self.spectra_data[i])
                unmodified_counter += 1                
            if self.spectra_data[i][5] == 1 and modified_counter < samples_per_class:
                filtered_spectra.append(self.spectra_data[i])  
                modified_counter += 1

        if unmodified_counter != samples_per_class or modified_counter != samples_per_class:
            print("Theres not enough data to comply with the samples per class requirement")
            exit()
        return filtered_spectra  

