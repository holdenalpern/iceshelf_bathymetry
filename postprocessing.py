import numpy as np
import xarray as xr
from scipy.interpolate import griddata
from tqdm.auto import tqdm
from pathlib import Path
import verde as vd
import os
from utilities import lowpass_filter_invpad



def load_data(ds, res_path, geoid=False, filter=False):
    """
    Load inversions results from directory

    Args:
        ds : Xarray.Dataset preprocessed BedMachine data
        res_path : path to directory with inversion results
        geoid : reference beds to geoid from ellipsoid
        filter : apply Gaussian lowpass filter to remove edges
    Outputs:
        Beds, mean of beds, stdev of beds, densities, losses
    """
    count = 0
    for entry in os.scandir(res_path):
        if 'result' in entry.name and '._' not in entry.name:
            count += 1
    print(f' {count} inversions')
    
    densities = np.zeros(count)
    last_iter = np.zeros((count, *ds.bed.shape))
    losses = np.zeros((count, 47_000))
    
    i = 0
    for entry in os.scandir(res_path):
        if 'result' in entry.name and '._' not in entry.name:
            result = np.load(entry.path, allow_pickle=True).item()
            densities[i] = result['density'][0]
            bed = result['bed_cache']
            if filter==True:
                bed_filt = lowpass_filter_invpad(ds, bed, cutoff=5e3)
                last_iter[i] = bed_filt.reshape(bed.shape)
            else:
                last_iter[i] = bed
            losses[i,:result['loss_cache'].size] = result['loss_cache']
            i += 1
            
            
    mean = np.mean(last_iter, axis=0)
    std = np.std(last_iter, axis=0)

    if geoid==True:
        last_iter -= ds.geoid.values
        mean -= ds.geoid.values
        std -= ds.geoid.values

    return last_iter, mean, std, densities, losses

def upscale_data(ds, grid, data, grid_vals, inv_msk_low, inv_msk_high, outside=True):
    """
    Upscale data to BedMachine v3 500 m resolution. Data and grid_vals are
    not bathymetry specific so that other fields like standard deviation can
    be upscaled as well.
    Args:
        ds : trimmed and coarsened BedMachine xarray.Dataset used for the inversions
        grid : trimmed BedMachine xarray.Dataset with original resolution
        data : array to upscale
        grid_vals : conditioning data at higher resolution
        inv_msk_high : inversion domain at higher resolution
        outside : if True, interpolate between the grid_vals outside inversion domain.
            Use True if interpolating coarse bathymetry to higher resolution bathymetry.
    Outputs:
        Data at 500m BedMachine resolution.
    """
    xx_i, yy_i = np.meshgrid(grid.x.values, grid.y.values)
    pred_coords = np.stack([xx_i.flatten(), yy_i.flatten()]).T
    xx_int = xx_i[~inv_msk_high]
    yy_int = yy_i[~inv_msk_high]
    interp_coords = np.stack([xx_int, yy_int]).T
    interp_vals = grid_vals[~inv_msk_high]
    
    xx_g, yy_g = np.meshgrid(ds.x.values, ds.y.values)
    xx_g = xx_g[inv_msk_low]
    yy_g = yy_g[inv_msk_low]
    interp_coords_grav = np.stack([xx_g.flatten(), yy_g.flatten()]).T
    interp_vals_grav = data[inv_msk_low]
    if outside==True:
        interp_vals_i = np.concatenate([interp_vals, interp_vals_grav])
        interp_coords_i = np.concatenate([interp_coords, interp_coords_grav], axis=0)
        
        upscale = griddata(interp_coords_i, interp_vals_i, pred_coords, method='cubic').reshape(grid.bed.shape)
    else:
        upscale = griddata(interp_coords_grav, interp_vals_grav, pred_coords, method='cubic').reshape(grid.bed.shape)
    upscale = np.where(inv_msk_high, upscale, grid.bed.values)
    return upscale

def save_upscale(ds, grid, inv_msk_low, inv_msk_high, data_path, out_path, out_path_up):
    """
    Load inversions, upscale beds, save upscaled beds

    Args:
        ds : trimmed and coarsened BedMachine xarray.Dataset used for the inversions
        grid : trimmed BedMachine xarray.Dataset at original resolution
        pred_msk : inversion domain mask at higher resolution
        data_path : path to directory with inversions
        out_path : path to save upscaled beds
    Outputs:
        None
    """
    beds, _, _, _, _ = load_data(ds, data_path, geoid=True, filter=True)

    beds_up = np.zeros((beds.shape[0], *grid.bed.shape))
    
    for i in tqdm(range(beds.shape[0])):
        beds_up_i = upscale_data(ds, grid, beds[i], grid.bed.values, inv_msk_low, inv_msk_high)
        beds_up[i,...] = np.where(beds_up_i > grid.surface-grid.thickness, grid.surface-grid.thickness, beds_up_i)

    ii = np.arange(beds.shape[0])
    ds_beds = xr.DataArray(beds, coords = {'i' : ii, 'y' : ds.y.values, 'x' : ds.x.values})
    ds_beds_up = xr.DataArray(beds_up, coords = {'i' : ii, 'y' : grid.y.values, 'x' : grid.x.values})

    # save as netcdf
    ds_beds.to_netcdf(out_path)
    ds_beds_up.to_netcdf(out_path_up)

