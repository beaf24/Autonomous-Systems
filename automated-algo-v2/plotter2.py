import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


mapp = np.loadtxt("export.txt", delimiter=",")


prob_map = 1 - 1/(1+np.exp(mapp))

# adicionado para thresholding
prob_map[prob_map>=0.7]=1
prob_map[prob_map<=0.3]=0
prob_map[(prob_map > 0.3) & (prob_map < 0.7)] = 0.5

fig, ax = plt.subplots()
plt.figure(frameon=False)
ax.set_axis_off()
fig.add_axes(ax)
im = ax.matshow(prob_map, cmap="Greys")
plt.show()


