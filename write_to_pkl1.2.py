import xarray as xr               # For working with NetCDF datasets
import pandas as pd              # For DataFrame manipulation
import matplotlib.dates as md    # For converting numerical dates to datetime

ds = xr.open_dataset('/media/exxact1/saiful/hypoxia/PEA_SOCalt_DCPtemp_hypxia_oxygen_depth_3D_2018_2020_exp14_2.nc', decode_times=False)
df = ds.to_dataframe()

print('df.columns: ', df.columns) 
print('ds.data_vars: ',ds.data_vars)       # Shows all data variables
print('ds.coords: ', ds.coords)          # Shows coordinate variables
# =============================================================================
ds3 = xr.open_dataset('/media/exxact1/saiful/hypoxia/hypoxiaAI_2020_2025_ROMS_Magesh_v2.nc', decode_times=False)
df3 = ds3.to_dataframe()

print('df3.columns: ', df3.columns) 
print('ds3.data_vars: ',ds3.data_vars)       # Shows all data variables
print('ds3.coords: ', ds3.coords)   

# Reset index so coords become columns
df.reset_index(inplace=True)
df3.reset_index(inplace=True)

# Align columns (intersection, but keep ocean_time explicitly)
common_cols = list(set(df.columns) & set(df3.columns))
if 'ocean_time' not in common_cols:
    common_cols.append('ocean_time')

# Filter and combine
df = df[common_cols]
df3_filtered = df3[common_cols]
df = pd.concat([df, df3_filtered], axis=0, ignore_index=True)


#%%
# =============================================================================

#df = df[(df.index > pd.to_datetime('2013-01-01')) & (df.index < pd.to_datetime('2018-12-31'))]
#df = df[(df.index > pd.to_datetime('2018-01-01')) & (df.index < pd.to_datetime('2018-12-31'))]
#df_test = df[(df.index > pd.to_datetime('2019-01-01')) & (df.index < pd.to_datetime('2019-12-31'))]

# df.reset_index(inplace=True)

df = df.dropna(axis=0)

df = df[(df['SOCalt'] > 0.0) & (df['PEA'] > 0.0) & (df['DCPtemp'] > 0.0)]

df['ocean_time'] = df['ocean_time'].astype(int) - 719529
df['ocean_date_time'] = df['ocean_time'].apply(lambda x: md.num2date(x) if pd.notnull(x) else pd.NaT)
df['ocean_time'] = df['ocean_time']

df['ocean_date'] = pd.to_datetime(df['ocean_date_time']).dt.date
df = df.set_index('ocean_date')

df = df.sort_index()

df_scale_vector_rbf = df[['lat_rho', 'lon_rho', 'ocean_date_time', 'ocean_time', 'SOCalt', 'PEA', 'DCPtemp', 'depth', 'oxyg']]

# write this df to a .pkl file and read it back for faster loading
df_scale_vector_rbf.to_pickle('df_hyp_input_2018_2025.pkl')
print('Execution Finished..')