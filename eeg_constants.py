import numpy as np
eeg = np.load("C:/Users/shali/Downloads/eeg/dataset/CN/sub-056_task-eyesclosed_eeg.csv.npy")
print(eeg.shape, eeg.dtype, np.min(eeg), np.max(eeg))
