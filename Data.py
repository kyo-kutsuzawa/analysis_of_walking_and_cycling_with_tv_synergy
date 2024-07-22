"""
 Data: class to load data

 (c) Felipe Moreira Ramos - Tohoku University, Sendai, Japan -
 

 Date: 20180910
"""

import warnings
import numpy as np
from scipy import io

class Data:
    class Obj:
        class Info:
            plane = None
            start = None
            target = None
            t_onset = None
            t_end = None
            
        emg = None
        emgtime = None
        pos = None
        postime = None
        info = None
        
        def __init__(self):
            self.info = self.Info()
        
    obj = None
    
    def __init__(self):
        self.obj = None

    def getFromCSVDATA(self,emgtime,emg,length):
        ntrial = length
        self.obj = np.array([self.Obj() for i in range(ntrial)])
        for i in range(ntrial):
            self.obj[i].emg = np.array(emg[i])
            self.obj[i].emgtime=np.array(emgtime[i])
            self.obj[i].pos = np.array([])
            self.obj[i].postime = np.array([])
            self.obj[i].info.plane = 0
            self.obj[i].info.start = 0
            self.obj[i].info.target = 0
            self.obj[i].info.t_onset = 0
            self.obj[i].info.t_end = 0
        return self
        
    
    def getFromMat(self,data):
        if not 'emg' in data.dtype.names or not 'emgtime' in data.dtype.names or not 'pos' in data.dtype.names or not 'postime' in data.dtype.names or not 'info' in data.dtype.names:
            warnings.warn('Data input must be a structure with .emg, .emgtime, .pos, .postime and .info fields',stacklevel=2)
            return
        
        ntrial = data.size
        self.obj = np.array([self.Obj() for i in range(ntrial)])
        
        for i in range(ntrial):
            self.obj[i].emg = np.array(data['emg'][0][i])
            self.obj[i].emgtime = np.array(data['emgtime'][0][i][0])
            self.obj[i].pos = np.array(data['pos'][0][i])
            self.obj[i].postime = np.array(data['postime'][0][i][0])
            self.obj[i].info.plane = data['info'][0][i]['plane'][0][0][0][0]
            self.obj[i].info.start = data['info'][0][i]['start'][0][0][0][0]
            self.obj[i].info.target = data['info'][0][i]['target'][0][0][0][0]
            self.obj[i].info.t_onset = data['info'][0][i]['t_onset'][0][0][0][0]
            self.obj[i].info.t_end = data['info'][0][i]['t_end'][0][0][0][0]

        
        #print(self.obj[159].info.t_end)
        return self
    
    def getFromMonkey(self,data):
        #100Hz
        ntrial = data.size
        downsample = 2 #downsample to 50Hz, Ts = 0.02s
        self.obj = np.array([self.Obj() for i in range(ntrial)])
        
        Nneurons = trial[0]['Neuron'].item().size
        for i in range(ntrial):
            self.obj[i].emgtime = np.array(data[i]['Time'].item()[::downsample,0])
            self.obj[i].pos = np.array(data[i]['HandVel'].item()[::downsample,0:2])# X and Y
            
            Ntime = self.obj[i].emgtime.size
            
            neurons = trial[0]['Neuron'].item()
            emg = np.zeros((Nneurons,Ntime))
            for n in range(Nneurons):
                spikes = np.zeros(Ntime)
                
                if neurons['Spike'][n].item().size: #Somes neurons don't have spikes
                    for k in range(4,Ntime-1):
                        spikes[k] = countSpikes(neurons['Spike'][n].item()[:,0],
                                                self.obj[i].emgtime[k+1],
                                                self.obj[i].emgtime[k-4])
                emg[n,:] = spikes
            self.obj[i].emg = emg
            print("Trial ", i, "\\", ntrial-1)
            
        return self
            
    def __getitem__(self,key):
        if self.obj is None:
            warnings.warn('Data not defined',stacklevel=2)
            return
        
        return self.obj[key]
    
    def __len__(self):
        if self.obj is None:
            warnings.warn('Data not defined',stacklevel=2)
            return
        
        return self.obj.shape[0]
    #def __setitem__(self, key):

"""
@Felipe
count the number of spikes that occur between 'after' and 'before'
The inequation returns a boolean array, and sum() counts the number of True Booleans
"""
def countSpikes(spike,after,before):
    n = sum(spike < after)-sum(spike < before);
    return n

if __name__ == '__main__':
    """
    print('Loading raw EMG data (reaching to 8 target in frontal and sagittal planes)')
    data_S6 = io.loadmat('data_S6')
    data = Data().getFromMat(data_S6['data'])
    emgchannels = data_S6['emgchannels']
    emgchannels = np.array([emgchannels[i][0][0] for i in range(emgchannels.size)])
    
    ntrial = data.obj.shape[0]
    for i in range(ntrial):
        print("trial: %i\tplane: %i\tstart: %i\ttarget: %i"%(i,data[i].info.plane,data[i].info.start,data[i].info.target))
    """
    stevenson = io.loadmat('Stevenson_2011_e1')
    print("Loaded")
    trial = stevenson['Subject'][0]['Trial'].item()
    #print(trial[0]['HandVel'].item()[:,0:2])
    #print(trial[0]['Neuron'].item()['Spike'][0].item()[:,0])
    data = Data().getFromMonkey(stevenson['Subject'][0]['Trial'].item())
    
    np.save('Stevenson_2011_e1.npy',data.obj)
    read_obj = np.load('Stevenson_2011_e1.npy')
    read_data = Data()
    read_data.obj = read_obj
    print(data[0].emg[0,:])
    print(read_data[0].emg[0,:])
