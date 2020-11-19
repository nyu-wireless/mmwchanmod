"""
antenna.py:  Classes for antenna modeling
"""
import numpy as np
import matplotlib.pyplot as plt
from  mmwchanmod.common.spherical import spherical_add_sub

def plot_pattern(pattern_fn,ntheta=None, theta=0,\
                 phi=0, nphi=None, plot_type='rect_phi', ax=None,\
                 ax_label=True, **kwargs):
        """
        Computes and optionally plots a pattern

        Parameters
        ----------
        pattern_fn :  function of (phi,theta) returning a scalar
            The function to plot, e.g. gain
        theta : int or array
            If integer represents the number of theta values to plot
            If an array, the values to plot.          
        nphi:  None or int
            Number of points in azimuth to plot.  If `None`, the azimuth
            points will be taken from `phi`
        ntheta:  None or int
            Number of points in elevation to plot.  If `None`, the elevation
            points will be taken from `theta`
            
        phi, theta:  arrays
            Array of azimuth and elevation angles of the plot.
            Used only if `nphi` or `ntheta` are specified
        plot_type : string
            `rect_phi`, `polar_phi`:  Rectangular or plot plot of the 
            pattern vs. phi.  One curve per `theta` value.
            `rect_theta`, `polar_theta`:  Rectangular or plot plot of the 
            pattern vs. theta.  One curve per `phi` value.
            `2d`:  2D imshow plot
            `none`:  No plot.
        ax_label:  Boolean
            Indicates if axis is to be labeled
        kwargs:  dictionary
            remaining arguments passed to the plot function.
                        

        Returns
        -------
        phi:  (nphi,) array
            Array of azimuth angles of the plot
        theta:  (ntheta,) array
            Array of elevation angles of the plot            
        val:  array
            Values of the pattern 
        ax:  axis
            Image 
        
        """
        # Create uniform angles, if requested
        if not (nphi is None):
            phi = np.linspace(-180,180,nphi)
        elif np.isscalar(phi):
            phi = np.array([phi])
        if not (ntheta is None):
            theta = np.linspace(-90,90,ntheta)
        elif np.isscalar(theta):
            theta = np.array([theta])           
        nphi = len(phi)
        ntheta = len(theta)
                
        # Create meshgrid of points
        phi_mat, theta_mat = np.meshgrid(phi, theta)
        phi_vec = phi_mat.ravel()
        theta_vec = theta_mat.ravel()
        
        # Compute the pattern
        v = pattern_fn(phi_vec, theta_vec)
        v = v.reshape((ntheta, nphi))
        
        # Get plot axis if not supplied
        if (ax is None) and (plot_type != 'none'):
            if (plot_type in ['polar_phi', 'polar_theta']):
                ax = plt.axes(projection = 'polar')
            else:
                ax = plt.axes()
        
        # Rectangular plot
        im = None
        if plot_type == 'rect_phi':
            ax.plot(phi, v.T, **kwargs)
            if ax_label:
                ax.set_xlabel('Azimuth (deg)')
                ax.set_xlim([-180,180])
 
        elif plot_type == 'polar_phi':           
            ax.plot(np.radians(phi), v.T, **kwargs)
        elif plot_type == 'rect_theta':
            ax.plot(theta, v, **kwargs)
            if ax_label:
                ax.set_xlabel('Elevation (deg)')
                ax.set_xlim([-90,90])
            
        elif plot_type == 'polar_theta':           
            ax.plot(np.radians(theta), v, **kwargs)
        elif plot_type == '2d':
            im = ax.imshow(np.flipud(v),extent=[np.min(phi),np.max(phi),\
                                np.min(theta),np.max(theta)],aspect='auto',\
                           **kwargs)
            if ax_label:
                ax.set_xlabel('Azimuth (deg)')
                ax.set_ylabel('Elevation (deg)')
                
        elif not plot_type == 'none':
            raise ValueError('Unknown plot type %s' % plot_type)
            
      
            
        if plot_type == 'none':
            return phi, theta, v
        else:
            return phi, theta, v, ax, im
            
        
            
        
        
class ElemBase(object):
    """
    Base class for the antenna element
    """
    def response(self,phi,theta):
        """
        The directivity response of the element.        

        Parameters
        ----------
        phi:  array
            Azimuth angles in degrees
        theta: array
            Elevation angles in degrees
       
        Returns
        -------
        gain:  array
            Antenna gain in dBi
        """
        raise NotImplementedError('response method not implemented')
    
    def compute_gain_mean(self, ns=1000):
        """
        Compute the mean gain.  For a lossless antenna, this should be zero.
        So, it can be used for calibration

        Parameters
        ----------
        ns : int, optional
            Number of samples used in the averaging computation

        Returns
        -------
        gain_mean:  float
            Mean of the gain in dBi.  

        """        
        # Generate random angles
        phi = np.random.uniform(-180,180,ns)
        theta = np.random.uniform(0,180,ns)
        
        # Compute weigths
        w = np.sin(theta*np.pi/180)
        
        # Get gain in linear scale
        gain = self.response(phi, theta)
        gain_lin = 10**(0.1*gain)
        
        # Find the mean gain 
        gain_mean = np.mean(gain_lin*w)/ np.mean(w)
        gain_mean = 10*np.log10(gain_mean)    

        return gain_mean            
    
    def plot_pattern(self,**kwargs):
        """
        Plots the gain pattern

        Parameters
        ----------
        **kwargs : dictionary
            See the plot_pattern() method

        Returns
        -------
        See the plot_pattern() method

        """
        return plot_pattern(self.response, **kwargs)
        
    
    
    

class ElemIsotropic(ElemBase):
    """
    Base class for the antenna element
    """
    
    def __init__(self):
        """
        Constructor
        """
        ElemBase.__init__(self)
        
        
    def response(self,phi,theta):
        """
        The directivity response of the element.
        The default implementation returns 

        Parameters
        ----------
        phi:  array
            Azimuth angles in degrees
        theta: array
            Elevation angles in degrees
       
        Returns
        -------
        gain:  array
            Antenna gain in dBi
        """
        gain = np.zeros(phi.shape)
        return gain
    
class Elem3GPP(ElemBase):
    """
    Class for computing the element gain using 3GPP BS model
    """
    def __init__(self, phi0=0, theta0=0, phibw=120, thetabw=65):
        """
        Constructor

        Parameters
        ----------
        phi0, theta0 : scalars
            Center azimuth and elevation angle in degrees
        phibw, thetabw : scalars
            Azimuth and elevation half-power beamwidth in degrees
            A value <0 indicates that there is no directivity 
            in the azimuth or elevation directions

        """
        
        # Super class 
        ElemBase.__init__(self)

        self.phi0 = phi0
        self.theta0 = theta0
        self.phibw = phibw
        self.thetabw = thetabw
        self.gain_max = 0
        
        # Other parameters
        self.slav = 30  # vertical side lobe
        self.Am = 30    # min gain
            
        # Calibrate
        self.calibrate()
        
    def response(self,phi,theta):
        """
        Computes antenna gain for angles
        
        Parameters
        ----------
        phi:  array
            Azimuth angles in degrees
        theta: array
            Elevation angles in degrees
            
        Returns
        -------
        gain:  array
            Antenna gain in dBi
        """
        # Rotate the angles relative to element boresight.
        # Note the conversion from inclination to elevation angles
        if (self.theta0 != 0) or (self.phi0 != 0):
            phi1, theta1 = spherical_add_sub(\
                self.phi0,90-self.theta0,phi,90-theta,sub=True)           
        else:
            phi1 = phi
            theta1 = theta
            
        # Put the 
        
        # Put the phi from -180 to 180
        phi1 = phi1 % 360
        phi1 = phi1 - 360*(phi1 > 180)
            
            
        if self.thetabw > 0:            
            Av = -np.minimum( 12*(theta1/self.thetabw)**2, self.slav)
        else:
            Av = 0
        if self.phibw > 0:
            Ah = -np.minimum( 12*(phi1/self.phibw)**2, self.Am)
        else:
            Ah = 0
        gain = self.gain_max - np.minimum(-Av-Ah, self.Am)
        return gain
        
    
    def calibrate(self, ns=10000):
        """
        Calibrates the maximum antenna gain

        Parameters
        ----------
        ns : int
            Number of samples used in the calibration
        """
        
        # Compute the mean gain
        gain_mean = self.compute_gain_mean(ns)        
        
        # Adjust the max gain
        self.gain_max = self.gain_max - gain_mean
        

        

        
        
        
        

