import matplotlib as mpl 
import matplotlib.pyplot as plt 

fig = plt.figure()
i=0
xlim=0
ylim=1
def set_limits(self, ax='ax_img', xlim=(0,1), ylim=(0,1)):
        axes[ax].set_xlim(*xlim)
        axes[ax].set_ylim(*ylim)
fig.set_limits(ax="ax_x{}".format(i), xlim=xlim, ylim=ylim)

