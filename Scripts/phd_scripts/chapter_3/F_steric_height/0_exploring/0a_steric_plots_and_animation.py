"""
script to create figure per time step of the steric height
also creates and saves an animation

this will be helpful for data cleaning (if needed)

"""
import sys
import xarray as xr
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
print(matplotlib.get_backend())
import matplotlib.pyplot as plt
from matplotlib import animation
import pandas as pd
import os

# PATHS
workdir = '/Users/iw2g24/PycharmProjects/SLA_analysis/'
script_dir = workdir + 'Scripts/'
data_dir = workdir + 'Data/'
auxscriptdir = script_dir + 'aux_scripts/'
fig_dir = workdir + 'Figures/F_steric_height/monthly_steric_height_plots/'
os.makedirs(fig_dir, exist_ok=True)


sys.path.append(auxscriptdir)
import aux_stereoplot as st
from geometry_izzyv1 import grad_sphere
import aux_func as ft


steric_height_ds_2024 = xr.open_dataset(data_dir + 'steric_height_cocks/steric_height_2002_2024.nc')

sha_time = steric_height_ds_2024.time
sha_lon = steric_height_ds_2024.longitude
sha_lat = steric_height_ds_2024.latitude
sha = steric_height_ds_2024.sha

plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['figure.autolayout'] = True

llon, llat = np.meshgrid(sha_lon, sha_lat)
vmin = float(-5)
vmax = float(5)


#create and save figure for each time step
(fig, ax) = plt.subplots()
# Initial frame
cax = ax.pcolormesh(llon, llat, sha.isel(time=0), cmap='RdBu_r', vmin=vmin, vmax=vmax)
cbar = fig.colorbar(cax)
title = ax.set_title(str(sha.time.isel(time=0).values)[:10])

for i in range(sha.sizes['time']):
    cax.set_array(sha.isel(time=i).values)
    ts = pd.Timestamp(sha.time.isel(time=i).values)
    label = ts.strftime('%Y_%m')
    title.set_text(label)
    fig.savefig(fig_dir + f'{label}.png' , dpi=300, bbox_inches='tight')

plt.close(fig)

# create a figure for the animations
(fig, ax) = plt.subplots()
# Initial frame
cax = ax.pcolormesh(llon, llat, sha.isel(time=0), cmap='RdBu_r', vmin=vmin, vmax=vmax)
cbar = fig.colorbar(cax)
title = ax.set_title(str(sha.time.isel(time=0).values)[:10])

def animate(i):
    print(f"frame {i}")
    cax.set_array(sha.isel(time=i).values)
    title.set_text(str(sha.time.isel(time=i).values)[:10])
    return cax, title

anim = animation.FuncAnimation(
    fig,
    animate,
    frames=sha.sizes['time'],
    interval=100,
    blit=False,
    repeat=False
)


animation_save_name = fig_dir +  'steric_height_2002_2024.mov'
anim.save(animation_save_name, writer='ffmpeg', fps = 10, dpi = 300)

plt.show()
plt.close()
