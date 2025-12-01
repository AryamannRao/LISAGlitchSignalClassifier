import numpy as np
from scipy.interpolate import interp1d, RectBivariateSpline
from gwpy.timeseries import TimeSeries


def generate_ft_arr(frange, trange, Q):
    fmin, fmax = frange
    tmin, tmax = trange
    
    f=fmin
    f_arr = [fmin]
    while f <= fmax:
        f = f + (f/Q)*(2*np.sqrt(2))
        f_arr.append(f)
    
    t_arr = []
    for f in f_arr:
        t = tmin
        tslice = [tmin]
        while t <= tmax:
            t = t + (Q/f)*(1/(np.sqrt(2)*np.pi))
            tslice.append(t)
        t_arr.append(tslice)

    return f_arr, t_arr

def q_transform(times, signal, Q, frange, trange, resolution):
    
    f_arr, t_arr = generate_ft_arr(frange, trange, Q)

    QT = []
    for i, f in enumerate(f_arr):
        t_slice = t_arr[i]
        out_slice = []
        for t in t_slice:
            tau = Q/(np.sqrt(2)*np.pi*f)
            
            seg = np.where((times > t - 4*tau) & (times < t + 4*tau))[0]
            timeseg = times[seg]
            sigseg = signal[seg]
            sigseg = sigseg*np.hanning(len(sigseg))
            
            window = np.exp(-0.5 * ((timeseg-t)/tau)**2) * np.exp(-2j*np.pi*f*(timeseg-t))
            window /= np.sqrt(np.sum(np.abs(window)**2))
            out_slice.append(np.abs(np.sum(sigseg * window)))
        try:
            f = interp1d(t_slice, out_slice, kind='cubic', fill_value='extrapolate')
        except:
            f = interp1d(t_slice, out_slice, kind='linear', fill_value='extrapolate')
        QT.append(f(t_arr[-1]))

    QT = np.array(QT)
    QT /= np.median(QT, axis=1, keepdims=True)
    QT *= np.log(2)
    
    f_new = np.linspace(np.min(f_arr), np.max(f_arr), resolution, endpoint=True)
    t_new = np.linspace(np.min(t_arr[-1]), np.max(t_arr[-1]), resolution, endpoint=True)
    
    cubicspline = RectBivariateSpline(f_arr, t_arr[-1], QT, kx=3, ky=3)
    QTnew = cubicspline(f_new, t_new)
    QTnew = np.clip(QTnew, 1e-12, None)
    
    return t_new, f_new, QTnew

def generate_qscan(tdi_dict, channel, pipe, event, 
                   frange, trange, Q, resolution):

    signal = TimeSeries(tdi_dict[channel], dt=pipe['dt'], t0=0)
    
    time_secs = signal.times.value
    time_hrs = time_secs/3600
    tevent = event['t_inj']
    
    index = np.where((time_hrs <= tevent/3600 + trange[1]) & (time_hrs >= tevent/3600 + trange[0]))[0]
    sigslice = signal[index].value
    timeslice = signal[index].times.value
    
    t_arr, f_arr, QT = q_transform(timeslice, sigslice, Q=Q, resolution=resolution,
                        frange=frange, trange=(np.min(timeslice),np.max(timeslice)))

    return t_arr, f_arr, QT