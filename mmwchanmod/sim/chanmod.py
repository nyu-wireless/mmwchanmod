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
        self.ang = np.zeros((0,MPChan.nangle), dtype=np.float32)
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
    
def dir_path_loss(tx_arr, rx_arr, chan, return_elem_gain=True,\
                       return_bf_gain=True):
    """
    Computes the directional path loss between RX and TX arrays

    Parameters
    ----------
    tx_arr, rx_arr : ArrayBase object
        TX and RX arrays
    chan : MPChan object
        Multi-path channel object
    return_elem_gain : boolean, default=True
        Returns the TX and RX element gains
    return_bf_gain : boolean, default=True
        Returns the TX and RX beamforming gains

    Returns
    -------
    pl_eff:  float
        Effective path loss with BF gain 
    tx_elem_gain, rx_elem_gain:  (n,) arrays
        TX and RX element gains on each path in the channel
    tx_bf_gain, rx_nf_gain:  (n,) arrays
        TX and RX BF gains on each path in the channel
    
    """
    
    if chan.link_state == LinkState.no_link:
        pl_eff = MPChan.large_pl
        tx_elem_gain = np.array(0)
        rx_elem_gain = np.array(0)
        tx_bf = np.array(0)
        rx_bf = np.array(0)
        
    else:
    
        # Get the angles of the path
        # Note that we have to convert from inclination to elevation angle
        aod_theta = 90 - chan.ang[:,MPChan.aod_theta_ind]
        aod_phi = chan.ang[:,MPChan.aod_phi_ind]
        aoa_theta = 90 - chan.ang[:,MPChan.aoa_theta_ind]
        aoa_phi = chan.ang[:,MPChan.aoa_phi_ind]
        
        tx_sv, tx_elem_gain = tx_arr.sv(aod_phi, aod_theta, return_elem_gain=True)
        rx_sv, rx_elem_gain = rx_arr.sv(aoa_phi, aoa_theta, return_elem_gain=True)
        
        
        # Compute path loss with element gains
        pl_elem = chan.pl - tx_elem_gain - rx_elem_gain
        
        # Select the path with the lowest path loss
        im = np.argmin(pl_elem)
        
        # Beamform in that direction
        wtx = np.conj(tx_sv[im,:])
        wtx = wtx / np.sqrt(np.sum(np.abs(wtx)**2))
        wrx = np.conj(rx_sv[im,:])
        wrx = wrx / np.sqrt(np.sum(np.abs(wrx)**2))
        
        # Compute the gain with both the element and BF gain
        # Note that we add the factor 10*np.log10(nanttx) to 
        # account the division of power across the TX antennas
        tx_bf = 20*np.log10(np.abs(tx_sv.dot(wtx)))
        rx_bf = 20*np.log10(np.abs(rx_sv.dot(wrx)))
        pl_bf = chan.pl - tx_bf - rx_bf
        
        # Subtract the TX and RX element gains
        tx_bf -= tx_elem_gain
        rx_bf -= rx_elem_gain
        
        # Compute effective path loss
        pl_min = np.min(pl_bf)
        pl_lin = 10**(-0.1*(pl_bf-pl_min))
        pl_eff = pl_min-10*np.log10(np.sum(pl_lin) )
    
    # Get outputs
    if not (return_bf_gain or return_elem_gain):
        return pl_eff
    else:
        out =[pl_eff]
        if return_elem_gain:
            out.append(tx_elem_gain)
            out.append(rx_elem_gain)
        if return_bf_gain:
            out.append(tx_bf)
            out.append(rx_bf)
        return out
    
def dir_path_loss_multi_sect(tx_arr_list, rx_arr_list, chan, return_elem_gain=True,\
                       return_bf_gain=True, return_arr_ind=True):
    """
    Computes the directional path loss between list of RX and TX arrays.
    This is typically used when the TX or RX have multiple sectors

    Parameters
    ----------
    tx_arr_list, rx_arr_list : list of ArrayBase objects
        TX and RX arrays
    chan : MPChan object
        Multi-path channel object
    return_arr_ind : boolean, default=True
        Returns the index of the chosen array
    return_elem_gain : boolean, default=True
        Returns the TX and RX element gains
    return_bf_gain : boolean, default=True
        Returns the TX and RX beamforming gains

    Returns
    -------
    pl_eff:  float
        Effective path loss with BF gain 
    ind_tx, ind_rx: int
        Index of the selected TX and RX arrays
    tx_elem_gain, rx_elem_gain:  (n,) arrays
        TX and RX element gains on each path in the channel
    tx_bf_gain, rx_nf_gain:  (n,) arrays
        TX and RX BF gains on each path in the channel
    
    """
    
    if chan.link_state == LinkState.no_link:
        pl_eff = MPChan.large_pl
        tx_elem_gain = np.array(0)
        rx_elem_gain = np.array(0)
        ind_rx = 0
        ind_tx = 0
        tx_bf = np.array(0)
        rx_bf = np.array(0)
        
    else:
    
        # Get the angles of the path
        # Note that we have to convert from inclination to elevation angle
        aod_theta = 90 - chan.ang[:,MPChan.aod_theta_ind]
        aod_phi = chan.ang[:,MPChan.aod_phi_ind]
        aoa_theta = 90 - chan.ang[:,MPChan.aoa_theta_ind]
        aoa_phi = chan.ang[:,MPChan.aoa_phi_ind]
        
        
        # Loop over the array combinations to find the best array 
        # with the lowest path loss
        pl_min = MPChan.large_pl
        for irx, rx_arr in enumerate(rx_arr_list):
            for itx, tx_arr in enumerate(tx_arr_list):
        
                tx_svi, tx_elem_gaini = tx_arr.sv(aod_phi, aod_theta,\
                                                return_elem_gain=True)
                rx_svi, rx_elem_gaini = rx_arr.sv(aoa_phi, aoa_theta,\
                                                return_elem_gain=True)
                    
                # Compute path loss with element gains
                pl_elemi = chan.pl - tx_elem_gaini - rx_elem_gaini
                
                # Select the path with the lowest path loss
                pl_mini = np.min(pl_elemi)
                
                if pl_mini < pl_min:
                    pl_min = pl_mini
                    im = np.argmin(pl_elemi)
                    tx_sv = tx_svi
                    rx_sv = rx_svi
                    tx_elem_gain = tx_elem_gaini
                    rx_elem_gain = rx_elem_gaini
                    ind_rx = irx
                    ind_tx = itx
                               
        # Beamform in that direction
        wtx = np.conj(tx_sv[im,:])
        wtx = wtx / np.sqrt(np.sum(np.abs(wtx)**2))
        wrx = np.conj(rx_sv[im,:])
        wrx = wrx / np.sqrt(np.sum(np.abs(wrx)**2))
        
        # Compute the gain with both the element and BF gain        
        tx_bf = 20*np.log10(np.abs(tx_sv.dot(wtx)))
        rx_bf = 20*np.log10(np.abs(rx_sv.dot(wrx)))
        pl_bf = chan.pl - tx_bf - rx_bf 
        
        # Subtract the TX and RX element gains
        tx_bf -= tx_elem_gain
        rx_bf -= rx_elem_gain
        
        # Compute effective path loss
        pl_min = np.min(pl_bf)
        pl_lin = 10**(-0.1*(pl_bf-pl_min))
        pl_eff = pl_min-10*np.log10(np.sum(pl_lin) )
    
    # Get outputs
    if not (return_bf_gain or return_elem_gain):
        return pl_eff
    else:
        out =[pl_eff]
        if return_arr_ind:
            out.append(ind_tx)
            out.append(ind_rx)
        if return_elem_gain:
            out.append(tx_elem_gain)
            out.append(rx_elem_gain)
        if return_bf_gain:
            out.append(tx_bf)
            out.append(rx_bf)
        return out
            
        
    
    
        
        

    
    
        

