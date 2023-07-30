import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np


data = pd.read_csv(r"C:\Users\rob\Desktop\PythonPlots\loss_log_Versuch16.txt.txt",sep=" ", header=None)

data = pd.DataFrame(data)
print(data)

array = np.empty(len(data), dtype = int)

for i in range(len(data)):
    array[i] = ++i

D_A = data[17]

G_A = data[19]

cycle_A = data[21]

plt.figure(figsize=(20,10))


plt.plot(array, D_A, label='D_B')

plt.plot(array, G_A, label='G_B')

plt.plot(array, cycle_A, label='Cycle Consistency B' )

plt.grid(True)

plt.title('Loss Kurven CycleGAN')

plt.ylabel('Loss')

plt.xlabel('Step')

#plt.xticks(epoch[::14])

plt.legend()

plt.savefig('LossKurvenCycleGAN')

plt.show()

