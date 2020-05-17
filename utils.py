import numpy as np
import math
import statistics
from scipy.signal import welch
def sway_dispersion(first,second):
    sway = create_ellipse(first,second)
    # the population variace of data, is a measure of the
    # variability(spread or dispersion of data)

    return statistics.pvariance(sway)

def percentile(x):
    return np.percentile(x,25),np.percentile(x,50),np.percentile(x,75)

def sway_are_per_sec(first,second,Time):
    sway = create_ellipse(first,second)
    Time = np.remainder(Time, Time[0])[-1]
    # velocity
    return float(sway/Time)


def zero_crossing_rate(arr):
    # 1d array
    return ((arr[:-1] * arr[1:]) < 0).sum()

def energy(arr):
    return sum(i * i for i in arr)
def sma(x,y,z):
#     Signal magnitude area
    return (sum(abs(x))+sum(abs(y))+sum(abs(z)))/len(x)
def energy_peak(values):
    freqs, psd = welch(values, 202.5)
    psdmax = max(psd)
    freqs_max = np.nan
    if not np.isnan(psdmax):
        index = list(np.where(psd == psdmax)[0])[0]
        freqs_max = freqs[index]
    return(psdmax,freqs_max)

# utility functions to calculate ellipse areas and create ellipse - used to find out sway areas in feature extraction
def calculate_ellipse_area(largest_X_val, largest_Y_val, X_center, Y_center):
    major_axis_len = abs(largest_X_val - X_center)
    minor_axis_len = abs(largest_Y_val - Y_center)
    return math.pi * major_axis_len * minor_axis_len


def create_ellipse(first, second):
    angular_velocity_log = np.vstack((first, second))

    covariance = np.cov(angular_velocity_log.astype(float))

    w, v = np.linalg.eig(covariance)

    sorted_idx = np.argsort(w)
    largest_eig_idx, smallest_eig_idx = sorted_idx[-1], sorted_idx[0]

    largest_eig_val, largest_eig_vec = w[largest_eig_idx], v[:, largest_eig_idx]
    smallest_eig_val, smallest_eig_vec = w[smallest_eig_idx], v[:, smallest_eig_idx]

    angle = np.arctan2(largest_eig_vec[1], largest_eig_vec[0])

    if angle < 0:
        angle += 2 * math.pi

    avg = angular_velocity_log.mean(axis=1)
    # print(avg)
    chisquare_val = 2.4477
    theta_grid = np.linspace(0, 2 * math.pi)
    phi = angle
    X0 = avg[0]
    Y0 = avg[1]
    a = chisquare_val * np.sqrt(largest_eig_val)
    b = chisquare_val * np.sqrt(smallest_eig_val)

    ellipse_x_r = a * np.cos(theta_grid)
    ellipse_y_r = b * np.sin(theta_grid)

    largest_X_val = max(ellipse_x_r)
    largest_Y_val = max(ellipse_y_r)

    ellipse_area = calculate_ellipse_area(largest_X_val, largest_Y_val, X0, Y0)
    return ellipse_area

def entropy_Rate(arr,bins):
    mag = (np.sqrt(np.square((arr[:,0]) + np.square(arr[:,1])  + np.square(arr[:2]) )))
    # sum = np.sum(mag)
    # plt.plot(mag)
    # plt.show()
    # print(np.shape(mag))
    hist = np.histogram(mag, bins=bins, normed=False)
    # print(np.shape(hist))
    entro=0
    sum = np.sum(hist[0])
    for i in range(0,len(hist)):
        if(hist[0][i]!=0):
            # possibility
            pi = hist[0][i]/sum
            entro -= pi*np.log2(pi)
    return entro

# def average_step_time(arr,threshold,interval):
#     #Time/num step time
#     time = float(len(arr)/20)
#     num_step,_ = num_steps(arr[:,0],arr[:,1],arr[:2],threshold,interval)
#     aver_step_time=time/num_step
#     return aver_step_time
