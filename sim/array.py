"""
array.py:  Classes for modeling antenna arrays
"""
import numpy as np
import sim.antenna
from common.phyconsts import PhyConst
from common.spherical import sph_to_cart, spherical_add_sub
from sim.antenna import plot_pattern


class ArrayBase(object):
    """
    Base class for an antenna array 
    """
    def __init__(self, elem=None, fc=28e9, elem_pos=np.array([[0,0,0]])):
        """
        Constructor        

        Parameters
        ----------
        elem : ElemBase object 
            Element for the antenna array.  If `None`, it sets
            it to an isotropic element
        elem_pos : (nant,3) array
            Element positions
        fc:  float
            Frequency in Hz.  Used for narrowband response
        """
        if elem is None:
            self.elem = sim.antenna.ElemIsotropic()
        else:
            self.elem = elem
        self.elem_pos = elem_pos
        self.fc = fc
        
    def sv(self,phi,theta,include_elem=True):
        """
        Gets the steering vectors for the array

        Parameters
        ----------
        phi : (n,) array
            azimuth angle in degrees
        theta : (n,) array
            elevation angle in degrees
        include_elem : boolean
            Indicate if the element pattern is to be included

        Returns
        -------
        usv:  (n,nant) array
            the steering vectors for each angle

        """
        # Convert scalar values to vector
        if np.isscalar(phi):
            phi = np.array([phi])
        if np.isscalar(theta):
            theta = np.array([theta])
        
        # Get unit vectors in the direction of the rays
        # Note the conversion from elevation to inclination
        u = sph_to_cart(1, phi, 90-theta)
        
        # Compute the delay along each path in wavelengths
        lam = PhyConst.light_speed/self.fc
        dly = u.dot(self.elem_pos.T)/lam
        
        # Phase rotation      
        usv = np.exp(1j*2*np.pi*dly)
        
        # Add element pattern if requested.
        # Note the element gain is converted to linear scale
        if include_elem:
            elem_gain = 10**(0.05*self.elem.response(phi,theta))
            usv = usv * elem_gain[:,None]
            
        return usv
    
    def conj_bf(self,phi,theta):
        """
        Gets the conjugate beamforming vectors for the array.
        The conjugate beamforming vectors are simply
        the normalized conjugates steering vectors

        Parameters
        ----------
        phi : (n,) array
            azimuth angle in degrees
        theta : (n,) array
            elevation angle in degrees

        Returns
        -------
        w:  (n,nant) array
            the BF vectors for each angle pair
        """
        
        # Convert scalar values to vector
        single_w = (np.isscalar(theta) and np.isscalar(phi))        
        if np.isscalar(phi):
            phi = np.array([phi])
        if np.isscalar(theta):
            theta = np.array([theta])   
            
        w = self.sv(phi,theta,include_elem=False)        
        wmag = np.sqrt(np.sum(np.abs(w)**2,axis=1))
        w = np.conj(w) / wmag[:,None]
        
        # Convert back to single vector if only scalar angles were used
        if single_w:
            w = w.ravel()
        return w
    
    
    def plot_pattern(self,w,include_elem=True,**kwargs):
        """
        Plots the array pattern for a given beamforming vector

        Parameters
        ----------
        w : (nant,) array
            Beamforming vector
        
        **kwargs : dictionary
            See the plot_pattern() method

        Returns
        -------
        See the plot_pattern() method

        """
        
        pat_fn = lambda phi, theta:\
            20*np.log10(np.abs(self.sv(phi,theta,include_elem=include_elem).dot(w)))
        
        return plot_pattern(pat_fn, **kwargs)
        
        

        
class URA(ArrayBase):
    """
    Uniform rectangular array.
    
    By default, the array elements are placed on the y-z plane so that
    with unit weights the maximum gain is along the x-axis
    """
    
    def __init__(self,nant,sep=None,**kwargs):
        """
        Constructor

        Parameters
        ----------
        nant : (2,) array
            nant[0]=number of elements in the y direction, and nant[1]
            is the number in the z-direction
        sep : (2,) array or None
            The separation in meters.  None will default to lambda/2
        **kwargs : list of parameters
            See ArrayBase
        """
        
        # Super constructor
        ArrayBase.__init__(self, **kwargs)
        
        # If the antenna separation is not specified, set to lambda/2
        
        if sep is None:
            lam = PhyConst.light_speed / self.fc
            sep = np.array([lam / 2, lam/2])
            
        # Compute the antenna positions
        nant_tot = nant[0]*nant[1]
        xind = np.arange(nant_tot) % nant[0]
        yind = np.arange(nant_tot) // nant[0]
        self.elem_pos = np.column_stack(\
            (np.zeros(nant_tot), xind*sep[0], yind*sep[1]))
            

class RotatedArray(ArrayBase):
    """
    A rotated array.    
    """
    def __init__(self,arr,phi0=0,theta0=0):
        """
        Constructor

        Parameters
        ----------
        arr:  ArrayBase object
            The base array, typically with boresight in the global direction
            of (phi,theta)=(0,0)            
        phi0, theta0:  floats
            Azimuth and elevation angle to rotate the array.
            This will be the boresight if the original array has its 
            boresight at (phi,theta)=(0,0)            
        """
        self.arr = arr
        self.phi0 = phi0
        self.theta0 = theta0
        
    def global_to_local(self,phi,theta):
        """
        Converts global to local angles

        Parameters
        ----------
        phi, theta : array of floats
            azimuth and elevation angles in the global coordinate system

        Returns
        -------
        phi1, theta1 : array of floats
            azimuth and elevation angles in the local coordinate system
        """
        
        # Note the conversion from elevation to inclination
        phi1, theta1 = spherical_add_sub(\
                self.phi0,90-self.theta0,phi,90-theta,sub=True)   
        return phi1, theta1
        
        
        
    def sv(self,phi,theta,include_elem=True):
        """
        Gets the steering vectors for the array

        Parameters
        ----------
        phi, theta : array of floats
            azimuth and elevation angles in the global coordinate system
        include_elem : boolean
            Indicate if the element pattern is to be included

        Returns
        -------
        usv:  (n,nant) array
            the steering vectors for each angle

        """
        # Convert scalar values to vector
        if np.isscalar(phi):
            phi = np.array([phi])
        if np.isscalar(theta):
            theta = np.array([theta])

        # Rotate angles    
        phi1, theta1 = self.global_to_local(phi,theta)        
        
        # Call the array class method with the rotated angles
        usv = self.arr.sv(phi1,theta1,include_elem)
                      
        return usv
    
    def conj_bf(self,phi,theta):
        """
        Gets the conjugate beamforming vectors for the array.
        The conjugate beamforming vectors are simply
        the normalized conjugates steering vectors

        Parameters
        ----------
        phi, theta : array of floats
            azimuth and elevation angles in the global coordinate system

        Returns
        -------
        w:  (n,nant) array
            the BF vectors for each angle pair
        """

        # Rotate angles    
        phi1, theta1 = self.global_to_local(phi,theta)        
        
        # Call the array class method with the rotated angles
        w = self.arr.conj_bf(phi1,theta1)
                            
        return w
        
    
            
        
            
        
        
        
        
        
    
            
        
