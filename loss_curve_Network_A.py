import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np


data = pd.read_csv(r"/home/frozen/Experiments_Repitition/Cell_cycleGAN/results/training_dataset_tiledGOWT_Fakulty_Inverse/cycle_gan/loss_log.txt",sep=" ", header=None)

data = pd.DataFrame(data)

array = np.empty(len(data), dtype = int)

for i in range(len(data)):
    array[i] = ++i

D_A = data[9]

G_A = data[11]

cycle_A = data[13]

plt.figure(figsize=(20,10))


plt.plot(array, D_A, label='D_A')

plt.plot(array, G_A, label='G_A')

plt.plot(array, cycle_A, label='Cycle Consistency A' )

plt.grid(True)

plt.title('Loss Kurven CycleGAN')

plt.ylabel('Loss')

plt.xlabel('Step')

#plt.xticks(epoch[::14])

plt.legend()

plt.savefig('LossKurvenCycleGAN_A')

plt.show()

