"""
create_dataset.py:  Creates train and test dataset for the channel modeling
"""
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import pickle
import sys
import argparse
from datetime import date

path = os.path.abspath('..')
if not path in sys.path:
    sys.path.append(path)
    
    
from mmwchanmod.sim.chanmod import LinkState
from mmwchanmod.learn.models import DataConfig


class DataSet(object):
    """
    Configuration for each data set
    """
    def __init__(self):
        self.folders = []
        self.train_test_split = False
        self.test_size = 0  
        self.desc = 'data set'


class ConfigReader(object):
    """
    Class to read a configuration file
    """
    
    def __init__(self, config_path):
        """
        Constructor
        """
        self.config_path = config_path
        
    def read_float(self,words):        
        """
        Reads a floating point value from the words in a line
        """
        if len(words) != 2:
            raise ValueError('Expecting a single argument after %s' % words[0])
        return float(words[1])
    def read_int(self,words):        
        """
        Reads an integer value from the words in a line
        """
        if len(words) != 2:
            raise ValueError('Expecting a single argument after %s' % words[0])
        return int(words[1])
    def read_string(self,words):
        """
        Reads a string value from the words in a line
        """
        if len(words) != 2:
            raise ValueError('Expecting a single argument after %s' % words[0])
        return words[1]    
    def read_string_array(self,words):
        """
        Reads a list of strings from words in a line
        """
        if len(words) < 2:
            raise ValueError('Expecting at least one argument after %s' % words[0])
        return words[1:]
    
    def read_string_joined(self,words):
        """
        Reads a list of strings and joins them to a single string
        """
        if len(words) < 2:
            raise ValueError('Expecting at least one argument after %s' % words[0])
        sep = ' '
        return sep.join(words[1:])
    
    def read_data_set(self, data_sets, words):
        """
        Reads a list of folders for a new data group
        """
        if len(words) < 3:
            raise ValueError('Expecting at least two arguments after %s' % words[0])
        data_sets[words[1]] = words[2:]

    
    def parse(self):        
        """
        Parses the configuration file

        Returns
        -------
        cfg : ConfigData structure
            Configuration data structure
        """
    
        # Open the configuration file
        try:
            fh = open(self.config_path, 'r')
        except:
            raise ValueError('Cannot open %s ' % self.config_path)
            
        # Create default configuration
        cfg = DataConfig()
        data_sets = dict()
        
        # Set the date
        cfg.date_created = date.today().strftime("%d-%m-%Y")        
            
        # Loop over lines
        line_num = 0
        for line in fh:
            words = line.split()
            nwords = len(words)
            line_num += 1
            
            if nwords == 0:
                continue
            try:
                if words[0] == 'fc':
                    cfg.fc = self.read_float(words)
                elif words[0] == 'rx_types':
                    cfg.rx_types = self.read_string_array(words)
                elif words[0] == 'pl_max':
                    cfg.pl_max = self.read_float(words)
                elif words[0] == 'name':
                    cfg.name = self.read_string(words)
                elif words[0] == 'npaths_max':
                    cfg.npaths_max = self.read_int(words)
                elif words[0] == 'tx_pow_dbm':
                    cfg.tx_pow_dbm = self.read_float(words)  
                elif words[0] == 'data_set':
                    name = self.read_string(words)
                    ds = DataSet()
                    data_sets[name] = ds
                elif words[0] == 'folder':
                    if (ds is None):
                        raise ValueError('Keyword folder before data_set')
                    ds.folders = self.read_string_array(words)
                elif words[0] == 'test_size':
                    if (ds is None):
                        raise ValueError('Keyword test_size before data_set')
                    ds.test_size = self.read_float(words)
                    ds.train_test_split = True 
                elif words[0] == 'desc':
                    ds.desc = self.read_string_joined(words)
                    
                else:
                    raise ValueError('Unknown keyword %s' % words[0])
            except ValueError as err:
                # Add line number
                err1 = 'Error parsing %s line %d\n' % (self.config_path, line_num)
                err1 = err1 + err.args[0]
                
                raise ValueError(err1)
                
        return cfg, data_sets
    
class PathReader(object):
    """
    Class to read files in a single folder.
    
    The data format for the files in the csv_data_fmt.md file    
    """
    
    def __init__(self, folder, cfg):
        """
        Constructor

        Parameters
        ----------
        folder : string
            path for the folder to parse
        cfg:  DataConfig object
            data configuration object
        """
        self.folder = folder
        self.cfg = cfg
        
        # Parsing constants
        self.ind_tx_pos = 0 
        self.ind_rx_pos = 3
        self.ndim = 3
        self.ind_rx_type = 6
        self.ind_npaths = 7
        self.ind_los = 8
        self.ind_path = 9
        self.nangle = 4
        self.ind_ang = 2
        self.ind_dly = 1
        self.ind_gain = 0
        self.col_per_path = 12
                    
        
    
    def read_csv(self):
        """
        Reads the CSV files in the folder 
        
        If found the data is placed in path_arr, tx_pos and rx_pos
        
        
        Returns
        -------
        found:  boolean
            Indicates if files were found        
                
        """
        csv_files = ['paths.csv', 'tx_pos.csv', 'rx_pos.csv']
        arrs = []
        for f in csv_files:
            
            # Get the full path for the desired CSV file
            # and check if it exists
            csv_path = os.path.join(self.folder, f)
            if not os.path.exists(csv_path):
                arrs = []                
                break
            
            # Read the csv file
            arr = np.array(pd.read_csv(csv_path), dtype=np.float32)
            
            # Remove the index column and add to list
            arr = arr[:,1:]
            arrs.append(arr)
        
        # Save the arrays
        if arrs != []:
            self.path_arr, self.tx_pos, self.rx_pos = arrs
            return True
        else:
            return False            
    
    def get_link_state(self):
        """
        Parses the path_arr data to compute:
            * dvec:  The distance between the TX and RX
            * link_state:  The link state (no_link, los_link or nlos_link)
            * rx_type:  The type of the RX
            
        Values are stored in the data dictionary
        """
        
        # Create empty data dictionary
        self.data = {}
        
        # Get the number of TX and RX 
        self.nrx = self.rx_pos.shape[0]
        self.ntx = self.tx_pos.shape[0]        
        
        # Get the RX position of each link
        self.rx_link_pos = self.path_arr[:,self.ind_rx_pos:self.ind_rx_pos+self.ndim]
        
        # Find the closest RX        
        dsq = np.sum((self.rx_link_pos[:,None,:] - self.rx_pos[None,:,:])**2,\
                     axis=2)
        self.rx_ind = np.argmin(dsq, axis=1)
        self.rx_dist = np.min(dsq, axis=1)
        
        # Get the RX position of each link
        self.tx_link_pos = self.path_arr[:,self.ind_tx_pos:self.ind_tx_pos+self.ndim]
        
        # Find the closest TX
        dsq = np.sum((self.tx_link_pos[:,None,:] - self.tx_pos[None,:,:])**2,\
                     axis=2)
        self.tx_ind = np.argmin(dsq, axis=1)
        self.tx_dist = np.min(dsq, axis=1)
        
        # Get total number of links and number of links in the path array
        # Note the path array only has links in non-outage
        self.nlink_tot = self.nrx*self.ntx
        self.nlink_arr = self.path_arr.shape[0]
        
        # Get the distance vectors
        dmat = self.tx_pos[None,:,:] - self.rx_pos[:,None,:]
        self.data['dvec'] = dmat.reshape((self.nlink_tot, self.ndim))
                
        # Create the path array to link index mapping
        self.link_ind = self.tx_ind + self.rx_ind*self.ntx
        
        # Get the RX type
        rx_type_arr = self.path_arr[:,self.ind_rx_type]
        rx_type = np.zeros(self.nlink_tot, dtype=int)        
        self.rx_type_node = np.zeros(self.nrx, dtype=int)
        for i in range(self.nrx):
            # Find the links with the desired rx index
            I = np.where(self.rx_ind == i)[0]
            
            # Set the RX types from the array file
            if len(I) > 0:
                self.rx_type_node[i] = rx_type_arr[I[0]]
                
            # Set the links 
            rx_type[i*self.ntx:(i+1)*self.ntx] = self.rx_type_node[i]
            
        self.data['rx_type'] = rx_type
        
        # Set the link state        
        los = self.path_arr[:,self.ind_los].astype(int)
        Ilos = np.where(los == 1)[0]
        link_state = np.tile(LinkState.no_link, self.nlink_tot).astype(int)
        link_state[self.link_ind] = LinkState.nlos_link
        link_state[self.link_ind[Ilos]] = LinkState.los_link
        self.data['link_state'] = link_state
        
        
        
    def get_path_data(self):
        """
        Gets the path data for each link
        
        Creates the arrays:
            * los_pl:  Path loss data for the LOS links
            * los_ang:  AoA and AoD for the LOS links
            * los_dly:  AoA and AoD for the LOS links
            * nlos_pl:  Path loss data for the NLOS links
            * nlos_ang:  AoA and AoD for the NLOS links
            * nlos_dly:  AoA and AoD for the NLOS links            
        """
        
        # Angles are stored in the file in opposite order to the use
        # in the file        
        Iang = np.array([3,2,1,0], dtype=np.int)
    
        # Find the LOS links
        los = self.path_arr[:,self.ind_los].astype(int)
        #Ilos = np.where(los == 1)[0]
        #Ilos1 = self.link_ind[Ilos]
        
        # Create arrays
        los_pl = np.zeros(self.nlink_tot, dtype=np.float32)
        los_ang = np.zeros((self.nlink_tot,self.nangle), dtype=np.float32)
        los_dly = np.zeros((self.nlink_tot,), dtype=np.float32)
        nlos_pl = np.zeros((self.nlink_tot, self.cfg.npaths_max),\
                           dtype=np.float32)
        nlos_ang = np.zeros((self.nlink_tot,self.cfg.npaths_max,self.nangle),\
                            dtype=np.float32)
        nlos_dly = np.zeros((self.nlink_tot,self.cfg.npaths_max),\
                            dtype=np.float32)
            
        # Set the maximum path loss on paths by default
        los_pl[:] = self.cfg.pl_max
        nlos_pl[:,:] = self.cfg.pl_max
        
        # Indices for the path gains
        Igain = np.arange(0,self.cfg.npaths_max)*self.col_per_path +\
                 self.ind_path + self.ind_gain
        
        
        for i in range(self.nlink_arr):
            ind = self.link_ind[i]
            
            # Compute the column index of the first path
            col_ind =  self.ind_path
            
            # Compute the number of paths to process
            npaths = self.path_arr[i,self.ind_npaths].astype(int)
            npaths = np.minimum(npaths, self.cfg.npaths_max)
            
            # Get the path losses for the link
            pl_link = cfg.tx_pow_dbm - self.path_arr[i,Igain[:npaths]]
            
            # Only count paths up to the maximum path loss
            npaths1 = np.sum(pl_link <= cfg.pl_max)
            npaths = np.minimum(npaths, npaths1)
            
            # If there are no paths, mark link as nolink and go to next 
            # link
            if npaths == 0:
                self.data['link_state'][ind] = LinkState.no_link 
                continue
                        
            # If it is a LOS channel, put the data in the
            # first path into the LOS data
            if los[i]:
                los_pl[ind] = self.cfg.tx_pow_dbm -\
                        self.path_arr[i, col_ind+self.ind_gain]
                los_dly[ind] = self.path_arr[i, col_ind+self.ind_dly]                        
                los_ang[ind,:] = \
                    self.path_arr[i, col_ind+self.ind_ang+Iang]
                col_ind += self.col_per_path
                npaths -= 1
                
            
            # Process the NLOS paths
            for j in range(npaths):
                # Stop after path losses exceed maximum value
                pli = self.cfg.tx_pow_dbm -\
                        self.path_arr[i, col_ind+self.ind_gain]
                if pli >= cfg.pl_max:
                    break
                
                # Add path
                nlos_pl[ind,j] = self.cfg.tx_pow_dbm -\
                        self.path_arr[i, col_ind+self.ind_gain]
                nlos_dly[ind,j] = self.path_arr[i, col_ind+self.ind_dly]                        
                nlos_ang[ind,j,:] = \
                    self.path_arr[i, col_ind+self.ind_ang+Iang]
                col_ind += self.col_per_path
                    
        # Save results        
        self.data['los_pl'] = los_pl
        self.data['los_ang'] = los_ang
        self.data['los_dly'] = los_dly
        self.data['nlos_pl'] = nlos_pl
        self.data['nlos_ang'] = nlos_ang
        self.data['nlos_dly'] = nlos_dly        
        
    def print_stats(self):
        """
        Prints statistics on the data
        """
        print('   nrx = %d (' % self.nrx, end='')
        for i, rx  in enumerate(self.cfg.rx_types):
            cnt = np.sum(self.rx_type_node == i)            
            print('%s: %d ' % (rx, cnt), end='')
        print(')')
        print('   ntx = %d ' % self.ntx)
        dmin = np.min(self.data['dvec'], axis=0)
        dmax = np.max(self.data['dvec'], axis=0)
        dbnd = dmax - dmin
        print('   area = %5.1f x %5.1f x %5.1f' % (dbnd[0], dbnd[1], dbnd[2]) )
        
        nlink = self.nrx*self.ntx
        frac_los  = np.mean(self.data['link_state'] == LinkState.los_link)
        frac_nlos = np.mean(self.data['link_state'] == LinkState.nlos_link)
        frac_out  = np.mean(self.data['link_state'] == LinkState.no_link)
        print('   nlink = %d  (LOS: %5.3f  NLOS: %5.3f  NoLink: %5.3f)' %\
              (nlink, frac_los, frac_nlos, frac_out))
        

class ParseCSV(object):
    """
    Recursively finds and parses CSV files starting at a top folder
    """
    def __init__(self, top_folder, folders, cfg):
        """
        Constructor

        Parameters
        ----------
        top_folder : string
            Top level folders to find CSV files
        folders:  list of strings
            Folders with top level to parse
        cfg:  DataConfig object
            Configuration data
        """
        self.top_folder = top_folder
        self.folders = folders
        self.cfg = cfg
        
    def parse(self):
        """
        Parse CSV files in the folders
        """
        self.data = None
        
        print('Searching files in %s' % self.top_folder)
        for folder0 in self.folders:  
            folder = os.path.join(self.top_folder, folder0)
            self.parse_recursive(folder)
            
        
              
    def parse_recursive(self, folder):
        """
        Recursively finds and parses CSV files starting at a top folder
        
        Parameters
        ----------
        folder:  file path
            folder path      
        """
        
        # Create a data reader       
        pr = PathReader(folder, self.cfg)
        
        # Read the CSV files in the folder if any
        found = pr.read_csv()
        if found:
            pr.get_link_state()
            pr.get_path_data()
            
            if self.data is None:
                # For the first time, get the dictionary
                self.data = pr.data
                nlinks = pr.data['link_state'].shape[0]
            else:
                for k in self.data:
                    v = pr.data[k]
                    if len(v.shape) == 1:
                        self.data[k] = np.hstack((self.data[k], v))
                    else:
                        self.data[k] = np.vstack((self.data[k], v))
                    
                    nlinks = v.shape[0]    

            # Count the number of link types 
            rx_type, cnts = np.unique(pr.data['rx_type'], return_counts=True)
            
            # Print the folder without the top_folder
            i0 = len(self.top_folder)            
            print('%s:' % folder[i0:])
            pr.print_stats()
        
        # Recursive go through sub-folders
        subfolders = [f.path for f in os.scandir(folder) if f.is_dir()]
        
        for f in subfolders:    
            self.parse_recursive(f)
        
"""
Parse arguments from command line
"""
csv_dir_default = 'C:\\Users\\sdran\\Google Drive (sr663@nyu.edu)\\RanganGroup' +\
          '\\Drone_Antenna_Modeling_WX\\PublicData\\uav_data_csv'   
parser = argparse.ArgumentParser(description='Builds the data set for training')
parser.add_argument(\
    '--csv_dir',action='store', default=csv_dir_default, \
    help='directory of input CSV files')
parser.add_argument(\
    '--model_dir',action='store',\
    default='../../data', help='directory of output train-test files which'\
        +' are the input to the training')
   

args = parser.parse_args()
csv_dir =  args.csv_dir
model_dir = args.model_dir

"""
Process the CSV files in each directory
"""

# Read the configuration file
config_fn = 'config.txt'
config_path = os.path.join(csv_dir, config_fn)    

rdr = ConfigReader(config_path)
cfg, data_sets = rdr.parse()  


# Create the output directory if needed
if not os.path.exists(model_dir):
    os.mkdir(model_dir)
    print('Created output directory %s' % model_dir)
    
# Loop over data sets
for ds_name in data_sets:
    
    # Get the data set info
    ds = data_sets[ds_name]
    
    # Parse the folders
    parser = ParseCSV(csv_dir, ds.folders, cfg)
    parser.parse()
    
    # Get data
    data = parser.data
    if data is None:
        print('Data set %s:  No CSV files found' % ds_name)
        continue
    
     
    # Output file path
    out_fn = '%s.p' % ds_name
    out_fn = os.path.join(model_dir, out_fn)
                 
                  
    # Train test split
    if ds.train_test_split:
        
        # Get random indices for training and test
        nlink = data['dvec'].shape[0]
        nts = int(nlink*ds.test_size)
        ntr = nlink - nts    
        I = np.random.permutation(nlink)
        
        train_data = dict()
        test_data = dict()
        
        for k in data:
            train_data[k] = data[k][I[:ntr]]
            test_data[k] = data[k][I[ntr:]]
                   
    
    else:
        # If there is no train-test split, all the data
        # is for test and training dictionary is empty
        test_data = data
        train_data = dict()
        
        
    # Save results
    cfg.desc = ds.desc
    with open(out_fn, 'wb') as fp:
        pickle.dump([cfg, train_data, test_data], fp)            
    print('Data set written to %s' % out_fn)
        
    
    

    
    
        





