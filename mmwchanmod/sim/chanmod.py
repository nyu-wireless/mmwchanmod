"""
chanmod.py:  Methods for modeling multi-path channels
"""

import numpy as np

from mmwchanmod.common.constants import LinkState    
    
class MPChan(object):
    """
    Class for storing list of rays
    """
    nangle = 4
    aoa_phi_ind = 0
    aoa_theta_ind = 1
    aod_phi_ind = 2
    aod_theta_ind = 3
    ang_name = ['AoA_Phi', 'AoA_theta', 'AoD_phi', 'AoD_theta']
    
    large_pl = 250
    
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
    
    def comp_dir_path_loss(self, tx_elem, rx_elem):
        """
        Computes the directional path loss
        
        Parameters
        ----------
        tx_elem:  object with a response() method
             Models the TX element
        

        Returns
        -------
        pl_gain:  float
            Effective path loss with gain
        """
        if self.link_state == LinkState.no_link:
            pl_gain = MPChan.large_pl
        else:
            
            # Compute the channel gain
            rx_theta = self.ang[:,MPChan.aoa_theta_ind]
            rx_phi = self.ang[:,MPChan.aoa_phi_ind]
            gain_rx = rx_elem.response(rx_phi, rx_theta)
            
            # Compute the channel gain
            tx_theta = self.ang[:,MPChan.aod_theta_ind]
            tx_phi = self.ang[:,MPChan.aod_phi_ind]
            gain_tx = tx_elem.response(tx_phi, tx_theta)
            
            # Compute the path loss with the gain
            pl_gain_path = self.pl - gain_rx - gain_tx
            pl_min = np.min(pl_gain_path)
            pl_lin = 10**(-0.1*(pl_gain_path-pl_min))
            pl_gain = pl_min-10*np.log10(np.sum(pl_lin) )
            
        return pl_gain
    
    def rms_dly(self):
        """
        Computes the RMS delay spread

        Returns
        -------
        dly_rms:  float
            RMS delay spread (std dev weighted by paths)
        """
        if self.link_state == LinkState.no_link:
            dly_rms = 0
        else:
            # Compute weights
            pl_min = np.min(self.pl)
            w = 10**(-0.1*(self.pl-pl_min))
            w = w / np.sum(w)

            # Compute weighted RMS            
            dly_mean = w.dot(self.dly)
            dly_rms = np.sqrt( w.dot((self.dly-dly_mean)**2) )
                    
        return dly_rms
    
        
        

    
    
        

