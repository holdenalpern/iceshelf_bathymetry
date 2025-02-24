import numpy as np
import xarray as xr
from scipy.interpolate import griddata
from tqdm.auto import tqdm
from pathlib import Path
import verde as vd
import os

import sys
sys.path.append("..")

from utilities import lowpass_filter_invpad
from postprocessing import *

"""
Run this script to load inversions, upscale them, and save upscaled beds.
"""

if __name__ == '__main__':
    # load preprocessed and original BedMachine
    ds = xr.open_dataset(Path('processed_data/xr_2000.nc'))
    grid = xr.open_dataset(Path('../raw_data/bedmachine/BedMachineAntarctica-v3.nc'))

    xx, yy = np.meshgrid(ds.x, ds.y)

    # trim original BedMachine, get coordinates
    x_trim = (grid.x >= np.min(xx)) & (grid.x <= np.max(xx))
    y_trim = (grid.y >= np.min(yy)) & (grid.y <= np.max(yy))
    grid = grid.sel(x=x_trim, y=y_trim)
    xx_bm, yy_bm = np.meshgrid(grid.x.values, grid.y.values)

    # interpolate inversion mask to original resolution
    kn = vd.KNeighbors(1)
    kn.fit((xx.flatten(), yy.flatten()), ds.inv_no_muto.values.flatten())
    inv_msk_high = kn.predict((xx_bm, yy_bm))
    inv_msk_high = inv_msk_high.reshape(xx_bm.shape) > 0.5
    inv_msk_low = ds.inv_no_muto.values

    # path to where inversion directories are
    base_path = Path('results')

    # save ensemble with conditioning and density
    print('upscaling beds cd')
    save_upscale(ds, grid, inv_msk_low, inv_msk_high,
                 base_path/'cond_dens',
                 base_path/'cond_dens_geoid_2000.nc',
                 base_path/'cond_dens_geoid_500.nc')

    # save ensemble with conditioning and no density
    print('upscaling beds cnd')
    save_upscale(ds, grid, inv_msk_low, inv_msk_high,
                 base_path/'cond_nodens',
                 base_path/'cond_nodens_geoid_2000.nc',
                 base_path/'cond_nodens_geoid_500.nc')

    # save ensemble with conditioning and no deteministic bouger
    print('upscaling beds c determ')
    save_upscale(ds, grid, inv_msk_low, inv_msk_high,
                 base_path/'cond_deterministic',
                 base_path/'cond_determ_geoid_2000.nc',
                 base_path/'cond_determ_geoid_500.nc')

    # # save ensemble with no conditioning and density
    # print('upscaling beds ucd')
    # save_upscale(grid, inv_msk_high,
    #              base_path/'uncond_dens',
    #              base_path/'uncond_dens_geoid_2000.nc',
    #              base_path/'uncond_dens_geoid_500.nc')

    # # save ensemble with no conditioning and no density
    # print('upscaling beds ucd')
    # save_upscale(grid, inv_msk_high,
    #              base_path/'uncond_nodens',
    #              base_path/'uncond_nodens_geoid_2000.nc',
    #              base_path/'uncond_nodens_geoid_500.nc')

    # # save ensemble with no conditioning and no deteministic bouger
    # print('upscaling beds uc determ')
    # save_upscale(grid, inv_msk_high,
    #              base_path/'uncond_deterministic',
    #              base_path/'uncond_determ_geoid_2000.nc',
    #              base_path/'uncond_determ_geoid_500.nc')