import numpy as np
from scipy.optimize import fsolve
from scipy.interpolate import interp1d
from pycbc.waveform import get_fd_waveform, get_td_waveform
from lisagwresponse import ResponseFromStrain

import warnings
warnings.filterwarnings("ignore")

class BinaryInspiralGW(ResponseFromStrain):
    """Represents a GW resulting from a binary merger.
    """
    def __init__(self, m1, m2, d, t_inj, spin1=0.0, spin2=0.0, domain='freq', **kwargs,) -> None:
        super().__init__(**kwargs)
        self.t_inj = float(t_inj)
        self.m1 = float(m1)
        self.m2 = float(m2)
        self.d = float(d)
        self.spin1 = float(spin1)
        self.spin2 = float(spin2)
        self.domain = domain
        self.f_lower = self.compute_flower()

        if self.domain == 'time':
            hp_td, hc_td = get_td_waveform(approximant="IMRPhenomXHM",
                mass1=self.m1, mass2=self.m2, delta_t=1e-3/self.f_lower, f_lower=self.f_lower, 
                                           distance=self.d, spin1z = self.spin1, spin2z = self.spin2)
            self.hp = hp_td
            self.hc = hc_td
            
        elif self.domain == 'freq':
            hp_fd, hc_fd = get_fd_waveform(approximant="IMRPhenomXHM",
                                       mass1=self.m1, mass2=self.m2, delta_f=self.f_lower/1e3,
                                       f_lower=self.f_lower, distance=self.d, 
                                           spin1z = self.spin1, spin2z = self.spin2)
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
        
        mask1 = h.sample_times < delta_t + 20000
        sigslice = h[mask1]
        tslice = h.sample_times[mask1]
        tslice = tslice + t[0]
        f = interp1d(tslice, sigslice, kind='cubic',
                         bounds_error=False, fill_value=0)
        return f(t)