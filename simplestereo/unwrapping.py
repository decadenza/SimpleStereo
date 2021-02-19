"""
phaseUnwrapping
=======
Contains different phase unwrapping strategies.
"""
#import ctypes

import numpy as np
import cv2

from simplestereo import _unwrapping

def infiniteImpulseResponse(phase, tau=1):
    # FUNZIONA! In teoria deve essere tau<1.
    # Loro usano tau=0.4 ma ho avuto risultati non buoni.
    # Al contrario di come dicono, secondo me t=1 significa poco noise.
    # E piccolo (es. 0.4) compensa per forte noise.
    # 2D simultaneous phase unwrapping and filtering: A review and comparison
    # DOI: 10.1364/OE.19.005126 (https://doi.org/10.1364/OE.19.005126)
    # DOI: https://doi.org/10.1016/j.optlaseng.2012.01.008
    # TODO: Lento, da implementare come estensione C++!!!
    # VEDERE CODICE ORIGINALE LINK NEL PAPER
    # N.B. Può essere lanciato più volte a cascata con t piccolo.
    unwrapped = _unwrapping.infiniteImpulseResponse(phase, tau)
    return unwrapped
        
