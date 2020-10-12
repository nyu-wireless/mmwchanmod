"""
chanmod.py:  Methods for modeling multi-path channels
"""

import numpy as np

class LinkState(object):
    """
    Static class with link states
    """
    no_link = 0
    los_link = 1
    nlos_link = 2
    nlink_state = 3
    
    
class MPChan(object):
    """
    Class for storing list of rays
    """
    nangle = 4
    
    def __init__(self):
        """
        Constructor
        
        Creates an empty channel
        """
        # Parameters for each ray
        self.pl  = np.zeros(0, dtype=np.float32)
        self.dly = np.zeros(0, dtype=np.float32)
        self.angle = np.zeros((0,MPChan.nangle), dtype=np.float32)
        self.link_state = LinkState.no_link        
        
    def comp_omni_path_loss(self):
        """
        Computes the omni-directional channel gain

        Returns
        -------
        pl_omni:  float
            Omni-directional path loss
        """
        if self.link_state == LinkState.no_link:
            pl_omni = np.inf
        else:
            pl_min = np.min(self.pl)
            pl_lin = 10**(-0.1*(self.pl-pl_min))
            pl_omni = pl_min-10*np.log10(np.sum(pl_lin) )
            
        return pl_omni
        
        

    
    
        

