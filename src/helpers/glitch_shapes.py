import numpy as np
from scipy.optimize import fsolve
from scipy.interpolate import interp1d
from pycbc.waveform import get_fd_waveform, get_td_waveform
from lisagwresponse import ResponseFromStrain
from helpers.config import *

import warnings
warnings.filterwarnings("ignore")

from lisaglitch import LPFLibraryGlitch, Glitch

class RealGlitch(Glitch):
    """Represents a one-sided double exponential glitch in the case where t_rise=t_fall

    Args:
        t_fall: Falling timescale
        amp: relative amplitude scale
    """

    def __init__(
        self,
        scale: float,
        run: int,
        index: int,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)

        self.scale = scale
        self.run = run
        self.index = index

    def compute_signal(self, t) -> np.ndarray:
        """Computes the one-sided double exponential model in the case where t_rise=t_fall.

        Args:
            t (array-like): Times to compute glitch model for.

        Returns:
            Computed model (array-like)
        """
        glitch = LPFLibraryGlitch(path=lpf_library_path, inj_point=self.inj_point, t0=self.t0,
                                       t_inj=self.t_inj, run=self.run, glitch=self.index)

        return self.scale*glitch.compute_signal(t)