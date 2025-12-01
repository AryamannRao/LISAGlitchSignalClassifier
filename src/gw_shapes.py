"""
Filename: gw_shapes.py
Author: William Mostrenko
Created: 2025-08-07
Description: Classes for gravitational wave shapes.
"""

import numpy as np
from scipy.optimize import fsolve
from scipy.interpolate import interp1d
from pycbc.waveform import get_fd_waveform, get_td_waveform
from lisagwresponse import ResponseFromStrain

class ReducedOneSidedDoubleExpGW(ResponseFromStrain):
    """Represents a one-sided double-exponential gw in the case where t_rise=t_fall

    Args:
        t_fall: Falling timescale
        amp: relative amplitude scale
    """
    def __init__(
        self,
        t_fall: float,
        amp: float,
        t_inj: float,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)

        self.t_fall = float(t_fall)
        self.amp = float(amp)
        self.level = self.amp * self.t_fall
        self.t_inj = float(t_inj)
        self.duration = self.compute_duration()

    def compute_duration(self) -> float:
        """compute an approximate duration for the gw"""
        roots = lambda t: self.compute_signal(t) - self.amp / 30

        guess = self.t_fall + 50

        return float(fsolve(roots, guess)[0])

    def compute_hcross(self, t) -> np.ndarray:
        return self.compute_signal(t)

    def compute_hplus(self, t) -> np.ndarray:
        return self.compute_signal(t)

    def compute_signal(self, t) -> np.ndarray:
        """Computes the one-sided double exponential model in the case where t_rise=t_fall.

        Args:
            t (array-like): Times to compute GW model for.

        Returns:
            Computed model (array-like)
        """
        offset = 405
        delta_t = t - self.t_inj + offset

        signal = self.level * delta_t * np.exp(-delta_t / self.t_fall) / self.t_fall**2

        return np.where(delta_t >= 0, signal, 0)


class BinaryInspiralGW(ResponseFromStrain):
    """Represents a GW resulting from a binary merger.
    """
    def __init__(self, m1, m2, d, e, t_inj, domain='freq', **kwargs,) -> None:
        super().__init__(**kwargs)
        self.t_inj = float(t_inj)
        self.m1 = float(m1)
        self.m2 = float(m2)
        self.d = float(d)
        self.e = float(e)
        self.domain = domain
        self.f_lower = self.compute_flower()

        if self.domain == 'time':
            hp_td, hc_td = get_td_waveform(approximant="IMRPhenomXHM",
                mass1=self.m1, mass2=self.m2, delta_t=1e-3/self.f_lower, f_lower=self.f_lower, 
                                           distance=self.d, eccentricity=self.e)
            self.hp = hp_td
            self.hc = hc_td
            
        elif self.domain == 'freq':
            hp_fd, hc_fd = get_fd_waveform(approximant="IMRPhenomXHM",
                                       mass1=self.m1, mass2=self.m2, delta_f=self.f_lower/1e3,
                                       f_lower=self.f_lower, distance=self.d)
            self.hp = hp_fd.to_timeseries()
            self.hc = hc_fd.to_timeseries()
        self.amp_hp = max(self.hp)
        self.amp_hc = max(self.hc)
        self.duration = self.compute_duration()

    def compute_flower(self) -> float:
        c = 3e8
        G = 6.67e-11
        M = (self.m1 + self.m2)*2e30
        r = 6
        return (c**3)/(((6*r)**1.5)*np.pi*G*M)

    def compute_duration(self) -> float:
        """compute an approximate duration for the gw"""
        h = np.sqrt(self.hp**2 + self.hc**2)
        threshold = max(h)/30
        
        mask = h >= threshold
        duration = np.sum(mask) * 0.25
        
        return duration

    def compute_hplus(self, t) -> np.ndarray:
        return self.compute_signal(self.hp, self.amp_hp, t)

    def compute_hcross(self, t) -> np.ndarray:
        return self.compute_signal(self.hc, self.amp_hc, t)

    def compute_signal(self, h, amp, t):
        h.start_time = 0
        peak_time = h.sample_times[np.where(np.array(h) == amp)][0]
        delta_t = self.t_inj - t[0] - 405
        h = h.cyclic_time_shift(-(peak_time - delta_t))
        h.start_time = 0
        
        mask1 = h.sample_times < delta_t + 10000
        sigslice = h[mask1]
        tslice = h.sample_times[mask1]
        tslice = tslice + t[0]
        f = interp1d(tslice, sigslice, 
                         bounds_error=False, fill_value=0)
        return f(t)
