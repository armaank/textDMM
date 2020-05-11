import os
import numpy as np
import matplotlib.pyplot as plt

# reading in log files from training run
logpath = 'log.log'

log = np.loadtxt(logpath, delimiter=',', skiprows=1)
loss = log[:,1]
n_epoch = range(1, len(loss) + 1)

# Visualize loss history
plt.figure(figsize=(10,4))
plt.plot(n_epoch, loss, 'r')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training NLL')
plt.grid(True)

plt.tight_layout()
plt.show()

