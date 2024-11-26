import numpy as np

def pure_signal(noised_signal,noise):
    pure_signal = np.array([])
    for i in range(len(noised_signal)):
        pure_signal = np.append(pure_signal,noised_signal[i]-noise[i])
    return pure_signal