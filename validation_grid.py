from mods.models import AlwaysZeroPredictor, SimpleLinearPredictor, ForestPredictor, LSTMPredictor
from mods.data import AuserDataset, DoganellaDataset, LucoDataset, PetrignanoDataset, AmiataDataset, LupaDataset, MadonnaDataset, ArnoDataset, BilancinoDataset
from mods.validate import train_test_score
import pandas as pd
import matplotlib.pyplot as plt
import os
from shutil import rmtree
from joblib import Parallel, delayed

def load_data(wrapper, chunk):
    path = f"data/chunked/{wrapper.name}{chunk}.csv"
    df = pd.read_csv(path, index_col=0)
    df.index = pd.to_datetime(df.index)
    return wrapper(df)

models = (
    #AlwaysZeroPredictor,
    #SimpleLinearPredictor,
    #ForestPredictor,
    LSTMPredictor,
)

datasets = dict(
    aquifer=(
        AuserDataset,
        DoganellaDataset,
        LucoDataset,
        PetrignanoDataset,
    ),
    waterspring=(
        AmiataDataset,
        LupaDataset,
        MadonnaDataset,
    ),
    river=(ArnoDataset,),
    lake=(BilancinoDataset,),
)

chunksizes = (
    5,
    10,
    15,
    30,
    60,
    90,
    135,
    180,
    270,
    360,
)

def create_nonextant_dir(path):
    if not os.path.exists(path):
        os.mkdir(path)

ROOT = 'data/metrics/'
create_nonextant_dir(ROOT)
TODO = []

for kind, dss in datasets.items():
    PATH_A = f"{ROOT}{kind}/"
    create_nonextant_dir(PATH_A)

    for ch in chunksizes:
        PATH_B = f"{PATH_A}{ch}/"
        create_nonextant_dir(PATH_B)

        for m in models:
            PATH = f"{PATH_B}{m.name}/"
            create_nonextant_dir(PATH)

            MDL = [m] * len(dss)
            CHUNK = [ch] * len(dss)
            DATA = [load_data(ds, ch) for ds in dss]
            PLOTS = [f"{PATH}{ds.name}.png" for ds in DATA]
            FRAMS = [[f"{PATH}{t}.csv" for t in ds.targets] for ds in DATA]
            TODO = TODO + list(zip(MDL, CHUNK, DATA, PLOTS, FRAMS))

def validate_n_save(m, ch, ds, plot_path, fram_paths):
    data_needed = any(not os.path.exists(fpath) for fpath in fram_paths)
    pic_needed = not os.path.exists(plot_path)
    if data_needed or pic_needed:
        metrics, history = train_test_score(m, ds.X, ds.y)

        if data_needed:
            for i,n in enumerate(ds.targets):
                metrics[n].to_csv(fram_paths[i])

        if pic_needed:
            fig, axs = plt.subplots(len(ds.targets), 1, figsize=(14, 7*len(ds.targets)))
            if len(ds.targets) == 1:
                axs = [axs]
            for i,n in enumerate(ds.targets):
                rmse = metrics[n]['error_sq'].mean() ** 0.5
                mae = metrics[n]['error_abs'].mean()
                title = f"{m.name}, {ch}days, {n} | RMSE {rmse:.7f} | MAE {mae:.7f}"
                x = range(history[n].shape[0])
                axs[i].set_title(title, fontweight='bold')
                axs[i].plot(x, history[n]['actual'])
                axs[i].plot(x, history[n]['predicted'])
            plt.savefig(plot_path)

if __name__ == '__main__':
    for args in TODO:
        validate_n_save(*args)
    # CPUs = len(os.sched_getaffinity(0))
    # Parallel(n_jobs=CPUs, prefer='threads')(delayed(validate_n_save)(*args) for args in TODO)

# exit() ########################################################



# from mods.models import AlwaysZeroPredictor, SimpleLinearPredictor, SimpleForestPredictor, SimpleLSTMPredictor
# from mods.data import AuserDataset, DoganellaDataset, LucoDataset, PetrignanoDataset, AmiataDataset, LupaDataset, MadonnaDataset, ArnoDataset, BilancinoDataset
# from mods.validate import train_test_score
# from shutil import rmtree
# import pandas as pd
# import matplotlib.pyplot as plt
# import numpy as np
# import os
# import warnings
# warnings.filterwarnings('ignore')
#
# def plot_target_predictions(col, chunk, mdl, scaled, unscaled, path):
#
#     fig, axs = plt.subplots(1, 1, figsize=(10,5))
#     fig.suptitle(f"{col} | Chunk Size: {chunk} | Model: {mdl}", fontweight='bold')
#
#     axs.plot(unscaled['actual'])
#     axs.plot(unscaled['predicted'])
#     axs.set_title(f"Unscaled Data | RMSE {unscaled['error_sq'].mean() ** 0.5} | MAE {unscaled['error_abs'].mean()}")
#     axs.text(0,0,
#         f"Standardized Scores\nRMSE {scaled['error_sq'].mean() ** 0.5}\nMAE {scaled['error_abs'].mean()}",
#         transform=axs.transAxes)
#
#     plt.savefig(f"{path}{col}.png")
#
# def saved_validation(rbase, mdl, ds):
#
#     path = f"{rbase}{mdl.name}_{ds.chunksize}/"
#     if not os.path.exists(path):
#         os.mkdir(path)
#     metrics = train_test_score(mdl, ds, ds.getX(), ds.gety())
#     rs = []
#     ms = []
#
#     for col in ds.targets:
#         s, u = metrics[col]['scaled'], metrics[col]['unscaled']
#         rs.append(s['error_sq'].mean() ** 0.5)
#         ms.append(s['error_abs'].mean())
#         plot_target_predictions(col, ds.chunksize, mdl.name, s, u, path)
#         metrics[col]['scaled'].to_csv(f"{path}{col}_scaled.csv")
#         metrics[col]['unscaled'].to_csv(f"{path}{col}_unscaled.csv")
#
#     return {
#         'rmse': np.mean(rs),
#         'mae': np.mean(ms)
#     }
#
# def validate_waterbody(rbase, mdl, datasets):
#     metrics = dict()
#     for dataset in datasets:
#         metrics[dataset.name] = saved_validation(rbase, mdl, dataset)
#     metrics = pd.DataFrame(metrics)
#     metrics['overall'] = metrics.mean(axis=1)
#     metrics.to_csv(f"{rbase}{mdl.name}_{datasets[0].chunksize}/metrics.csv")
#     return metrics['overall']
#
# def validate_aquifers(rbase, mdl, datasets):
#     return validate_waterbody(rbase, mdl, datasets)
#
# def validate_watersprings(rbase, mdl, datasets):
#     return validate_waterbody(rbase, mdl, datasets)
#
# def validate_rivers(rbase, mdl, datasets):
#     return validate_waterbody(rbase, mdl, datasets)
#
# def validate_lakes(rbase, mdl, datasets):
#     return validate_waterbody(rbase, mdl, datasets)
#
# def validate_model(rbase, mdl, datasets):
#     am = validate_aquifers(f"{rbase}aquifer/", mdl, datasets['aquifer'])
#     wm = validate_watersprings(f"{rbase}waterspring/", mdl, datasets['waterspring'])
#     rm = validate_rivers(f"{rbase}river/", mdl, datasets['river'])
#     lm = validate_lakes(f"{rbase}lake/", mdl, datasets['lake'])
#     bodies = {'aquifer':am,'waterspring':wm,'river':rm,'lake':lm}
#     scores = pd.concat(bodies, axis=1)
#     scores.index = am.index
#     scores['overall'] = scores.mean(axis=1)
#     scores.to_csv(f"{rbase}/{mdl.name}_{datasets['aquifer'][0].chunksize}_metrics.csv")
#
# if __name__ == '__main__':
#
#     # initialize
#     results_base = 'data/models/'
#     data_path = 'data/raw/'
#     chunksizes = (5,10,15,30,60,90,180,270,360)
#     if os.path.exists(results_base):
#         rmtree(results_base)
#     os.mkdir(results_base)
#     for body_type in ('aquifer', 'waterspring', 'river', 'lake'):
#         os.mkdir(f"{results_base}{body_type}/")
#
#     # load
#     loaded_data = dict()
#     for ch in chunksizes:
#         print('LOADING', ch)
#         loaded_data[ch] = {
#             'aquifer': (
#                 AuserDataset(f"{data_path}aquifer/auser.csv", ch),
#                 DoganellaDataset(f"{data_path}aquifer/doganella.csv", ch),
#                 LucoDataset(f"{data_path}aquifer/luco.csv", ch),
#                 PetrignanoDataset(f"{data_path}aquifer/petrignano.csv", ch)
#             ),
#             'waterspring': (
#                 AmiataDataset(f"{data_path}waterspring/amiata.csv", ch),
#                 LupaDataset(f"{data_path}waterspring/lupa.csv", ch),
#                 MadonnaDataset(f"{data_path}waterspring/madonna_di_canneto.csv", ch)
#             ),
#             'river': ( ArnoDataset(f"{data_path}river/arno.csv", ch), ),
#             'lake': ( BilancinoDataset(f"{data_path}lake/bilancino.csv", ch) )
#         }
#
#     # validate the individual models
#     for _, dss in loaded_data.items():
#         for model in [AlwaysZeroPredictor,SimpleLinearPredictor,SimpleForestPredictor]:
#             validate_model(results_base, model, dss)
