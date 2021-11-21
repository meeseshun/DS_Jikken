import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

num_dim = 128

dim = list(range(0, num_dim))
r = 1
a = 0.8
ratios = []

for i in dim:
    V_2 = r**i - (a*r)**i
    V_1 = r**i
    ratios.append(V_2 / V_1)

x = np.arange(0, num_dim, 1)
y =  ratios
plt.xlabel('number of dimentions')
plt.ylabel('V2/V1')
plt.title('curse of dimentions(a=0.8, r=1.0)')
plt.plot(x, y)
plt.grid()
plt.show()

plt.close()