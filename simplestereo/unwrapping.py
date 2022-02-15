"""
phaseUnwrapping
=======
Contains different phase unwrapping strategies.
"""

from simplestereo import _unwrapping


def infiniteImpulseResponse(phase, tau=1):
    """
    Unwrap a 2D phase map. 
    
    Method from "Noise robust linear dynamic system for 
    phase unwrapping and smoothing", Estrada et al, 2011,
    DOI: 10.1364/OE.19.005126
    
    Parameters
    ----------
    phase : ndarray
        A 2D array containing the wrapped phase values.
    tau : float, optional
        Noise regularization parameter. Accept values from 0 to 1.
        Lower values used for higher error. 
        Default to 1.
    
    Returns
    -------
    ndarray
        Unwrapped phase with same dimensions and type of `phase`.
        
        
    See Also
    --------
    <https://doi.org/10.1016/j.optlaseng.2012.01.008>
    
    
    .. todo::
        If called twice, it may not work. To be fixed!
    """
    return _unwrapping.infiniteImpulseResponse(phase, tau)
        
