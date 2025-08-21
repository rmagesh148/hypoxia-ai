# ds = xr.open_dataset('PEA_SOCalt_DCPtemp_hypxia_oxygen_depth_3D_2018_2020_exp14_2.nc', decode_times=False)
    # df = ds.to_dataframe()

    # #df = df[(df.index > pd.to_datetime('2013-01-01')) & (df.index < pd.to_datetime('2018-12-31'))]
    # #df = df[(df.index > pd.to_datetime('2018-01-01')) & (df.index < pd.to_datetime('2018-12-31'))]
    # #df_test = df[(df.index > pd.to_datetime('2019-01-01')) & (df.index < pd.to_datetime('2019-12-31'))]

    # df.reset_index(inplace=True)

    # df = df.dropna(axis=0)

    # df = df[(df['SOCalt'] > 0.0) & (df['PEA'] > 0.0) & (df['DCPtemp'] > 0.0)]

    # df['ocean_time'] = df['ocean_time'].astype(int) - 719529
    # df['ocean_date_time'] = df['ocean_time'].apply(lambda x: md.num2date(x) if pd.notnull(x) else pd.NaT)
    # df['ocean_time'] = df['ocean_time']

    # df['ocean_date'] = pd.to_datetime(df['ocean_date_time']).dt.date
    # df = df.set_index('ocean_date')

    # df = df.sort_index()

    # df_scale_vector_rbf = df[['lat_rho', 'lon_rho', 'ocean_date_time', 'ocean_time', 'SOCalt', 'PEA', 'DCPtemp', 'depth', 'oxyg']]

    # # write this df to a .pkl file and read it back for faster loading
    # df_scale_vector_rbf.to_pickle('df_hyp_input.pkl')