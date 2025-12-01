import os
from gw_shapes import ReducedOneSidedDoubleExpGW, BinaryInspiralGW

from scipy.signal.windows import tukey
from pytdi.michelson import X2, Y2, Z2
from gwpy.timeseries import TimeSeries, TimeSeriesDict
from lisainstrument import Instrument
from pytdi import Data

from lisaglitch import RectangleGlitch, ShapeletGlitch, OneSidedDoubleExpGlitch, TwoSidedDoubleExpGlitch

def create_gws(gws, pipe, gw_path, orbits_path):
    if os.path.exists(gw_path):
        os.remove(gw_path)
    for gw in gws:
        if gw['type'] == 'BinaryInspiralGW':
            gw = BinaryInspiralGW(m1=gw['m1'], m2=gw['m2'], d=gw['d'], e=gw['e'], t_inj=gw['t_inj'] + pipe['t0'],
                        gw_beta=pipe['gw_beta'], gw_lambda=pipe['gw_lambda'], orbits=orbits_path,
                        dt=pipe['dt'], size=pipe['size'], t0=pipe['t0'], domain=gw['domain'])

            gw.write(path=gw_path, mode="a")
        elif gw['type'] == 'ReducedOneSidedDoubleExpGW':
            gw = ReducedOneSidedDoubleExpGW(t_inj=gw['t_inj'] + pipe['t0'], t_fall=gw['t_fall'],
                    amp=gw['amp'], gw_beta=pipe['gw_beta'], gw_lambda=pipe['gw_lambda'], orbits=orbits_path,
                                            t0=pipe['t0'], size=pipe['size'], dt=pipe['dt'])

            gw.write(path=gw_path, mode="a")

def create_glitches(glitches, pipe, glitch_path):
    if os.path.exists(glitch_path):
        os.remove(glitch_path)
    for glitch in glitches:
        if glitch['type'] == 'OneSidedDoubleExpGlitch':
            glitch = OneSidedDoubleExpGlitch(inj_point=glitch['inj_point'], 
                    t_inj=glitch['t_inj'] + pipe['t0'], t_rise=glitch['t_rise'], t_fall=glitch['t_fall'],
                    level=glitch['level'], t0=pipe['t0'], size=pipe['size'], dt=pipe['dt'])

            glitch.write(path=glitch_path, mode="a")

        elif glitch['type'] == 'TwoSidedDoubleExpGlitch':
            glitch = TwoSidedDoubleExpGlitch(inj_point=glitch['inj_point'], 
                    t_inj=glitch['t_inj'] + pipe['t0'], t_rise=glitch['t_rise'], t_fall=glitch['t_fall'],
                    level=glitch['level'], displacement = glitch['displacement'],
                    t0=pipe['t0'], size=pipe['size'], dt=pipe['dt'])

            glitch.write(path=glitch_path, mode="a")

        elif glitch['type'] == 'RectangleGlitch':
            glitch = RectangleGlitch(width=glitch['width'], 
                    t_inj=glitch['t_inj'] + pipe['t0'], level=glitch['level'],
                        inj_point=glitch['inj_point'], t0=pipe['t0'], size=pipe['size'], dt=pipe['dt'])

            glitch.write(path=glitch_path, mode="a")

        elif glitch['type'] == 'ShapeletGlitch':
            glitch = ShapeletGlitch(level=glitch['level'], beta=glitch['beta'], 
                                    inj_point=glitch['inj_point'], t_inj=glitch['t_inj'] + pipe['t0'],
                                   t0=pipe['t0'], size=pipe['size'], dt=pipe['dt'])

            glitch.write(path=glitch_path, mode="a")

def run_simulation(gws, glitches, pipe, 
                   gw_path, glitch_path, orbits_path, 
                   simulation_path, disable_noise=False):
    create_gws(gws, pipe, gw_path, orbits_path)
    create_glitches(glitches, pipe, glitch_path)

    lisa_instrument = Instrument(size=pipe['size'], dt=pipe['dt'], t0=pipe['t0'],
        orbits=orbits_path, physics_upsampling=1, aafilter=None,
        glitches=glitch_path, gws=gw_path)

    lisa_instrument.disable_dopplers()
    if disable_noise:
        lisa_instrument.disable_all_noises()

    if os.path.exists(simulation_path):
        os.remove(simulation_path)
    
    lisa_instrument.write(simulation_path)

def run_tdi(simulation_path, pipe):
    channels = [X2, Y2, Z2]
    tdi_names = ["X", "Y", "Z"]
    tdi_dict = TimeSeriesDict()

    data = Data.from_instrument(simulation_path)
    data.delay_derivative = None

    for i in range(len(channels)):
        channel = channels[i]
    
        # CALCULATE TDI CHANNEL DATA
        tdi_data = channel.build(**data.args)(data.measurements)
    
        # WINDOW TDI CHANNEL DATA
        window = tukey(tdi_data.size, alpha=0.001)
        tdi_dict[tdi_names[i]] = TimeSeries(tdi_data * window, t0=pipe['t0'], dt=pipe['dt'])
    
    return tdi_dict