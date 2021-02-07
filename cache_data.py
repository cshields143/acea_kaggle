from mods.data import chunk_days
import pandas as pd
from tqdm import tqdm

bodies = (
    'auser',
    'doganella',
    'luco',
    'petrignano',
    'amiata',
    'lupa',
    'madonna_di_canneto',
    'arno',
    'bilancino',
)

for bodyname in bodies:
    df = pd.read_csv(f'data/imputed/{bodyname}.csv', index_col=0)
    df.index = pd.to_datetime(df.index, format='%Y-%m-%d')
    df.freq = 'D'

    for chunksize in (5,10,15,30,60,90,135,180,270,360):
        df_new = chunk_days(df, chunksize)
        df_new.to_csv(f'data/chunked/{bodyname}{chunksize}.csv')

# from mods.clean import aquifer_pipe, waterspring_pipe, river_pipe, lake_pipe
# import pandas as pd
#
# auser_ys = (
#     'Depth_to_Groundwater_SAL',
#     'Depth_to_Groundwater_CoS',
#     'Depth_to_Groundwater_LT2',
# )
# doganella_ys = (
#     'Depth_to_Groundwater_Pozzo_1',
#     'Depth_to_Groundwater_Pozzo_2',
#     'Depth_to_Groundwater_Pozzo_3',
#     'Depth_to_Groundwater_Pozzo_4',
#     'Depth_to_Groundwater_Pozzo_5',
#     'Depth_to_Groundwater_Pozzo_6',
#     'Depth_to_Groundwater_Pozzo_7',
#     'Depth_to_Groundwater_Pozzo_8',
#     'Depth_to_Groundwater_Pozzo_9',
# )
# luco_ys = (
#     'Depth_to_Groundwater_Podere_Casetta',
# )
# petrignano_ys = (
#     'Depth_to_Groundwater_P24',
#     'Depth_to_Groundwater_P25',
# )
# amiata_ys = (
#     'Flow_Rate_Bugnano',
#     'Flow_Rate_Arbure',
#     'Flow_Rate_Ermicciolo',
#     'Flow_Rate_Galleria_Alta',
# )
# lupa_ys = (
#     'Flow_Rate_Lupa',
# )
# madonna_ys = (
#     'Flow_Rate_Madonna_di_Canneto',
# )
# arno_ys = (
#     'Hydrometry_Nave_di_Rosano',
# )
# bilancino_ys = (
#     'Lake_Level',
#     'Flow_Rate',
# )
#
# insouts = (
#     ('aquifer', 'auser', auser_ys),
#     ('aquifer', 'doganella', doganella_ys),
#     ('aquifer', 'luco', luco_ys),
#     ('aquifer', 'petrignano', petrignano_ys),
#     ('waterspring', 'amiata', amiata_ys),
#     ('waterspring', 'lupa', lupa_ys),
#     ('waterspring', 'madonna_di_canneto', madonna_ys),
#     ('river', 'arno', arno_ys),
#     ('lake', 'bilancino', bilancino_ys),
# )
# cleaners = {
#     'aquifer': aquifer_pipe,
#     'waterspring': waterspring_pipe,
#     'river': river_pipe,
#     'lake': lake_pipe
# }
#
# for bodytype, name, targets in insouts:
#     path = f"data/raw/{bodytype}/{name}.csv"
#     df = pd.read_csv(path)
#     df = cleaners[bodytype](df, *targets)
#     path = f"data/imputed/{name}.csv"
#     df.to_csv(path)
