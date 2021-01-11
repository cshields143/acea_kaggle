import pandas as pd
from .data import AuserDataset, DoganellaDataset, LucoDataset, PetrignanoDataset, AmiataDataset, LupaDataset, MadonnaDataset, ArnoDataset, BilancinoDataset
import matplotlib.pyplot as plt
import numpy as np

def calculate_errors(df):
    df['error'] = df['actual'] - df['predicted']
    df['error_abs'] = df['error'].abs()
    df['error_sq'] = df['error'] * df['error']
    return df

def train_test_score(mdl, data, X, y):
    a = []
    p = []
    sent = int(X.shape[0] * 0.67)

    for i in range(sent, X.shape[0] - 1):
        X_train, y_train = X.iloc[:i], y.iloc[:i]
        a.append(y.iloc[i])
        p.append(mdl().fit(X_train, y_train).predict())

    a = pd.DataFrame(a, index=y.iloc[sent:-1].index, columns=y.columns)
    p = pd.DataFrame(p, index=y.iloc[sent:-1].index, columns=y.columns)
    au = data.unscale(a)
    pu = data.unscale(p)

    metrics = {}
    for c in y.columns:
        s = pd.concat({'actual':a[c],'predicted':p[c]}, axis=1)
        s.index = a.index
        u = pd.concat({'actual':au[c],'predicted':pu[c]}, axis=1)
        u.index = au.index
        metrics[c] = {
            'scaled': calculate_errors(s),
            'unscaled': calculate_errors(u)
        }

    return metrics

def plot_target_predictions(c, ch, mdl, scaled, unscaled, path):

    fig, axs = plt.subplots(1, 1, figsize=(10,5))
    fig.suptitle(f"{c} | Chunk Size: {ch} | Model: {mdl}", fontweight='bold')

    axs.plot(unscaled['actual'])
    axs.plot(unscaled['predicted'])
    axs.set_title(f"Unscaled Data | RMSE {unscaled['error_sq'].mean() ** 0.5} | MAE {unscaled['error_abs'].mean()}")
    axs.text(0,0,
        f"Standardized Scores\nRMSE{scaled['error_sq'].mean() ** 0.5}\nMAE {scaled['error_abs'].mean()}",
        transform=axs.transAxes)

    plt.savefig(f"{path}{c}.png")

def saved_validation(folder, chunk, mdl, ds):

    base = f"{folder}{mdl.name}_{chunk}/{ds.type}/"
    metrics = train_test_score(mdl, ds, ds.getX(), ds.gety())
    rs = []
    ms = []

    for c in ds.targets:
        s, u = metrics[c]['scaled'], metrics[c]['unscaled']
        rs.append(s['error_sq'].mean() ** 0.5)
        ms.append(s['error_abs'].mean())
        plot_target_predictions(c, chunk, mdl.name, s, u, base)
        metrics[c]['scaled'].to_csv(f"{base}{c}_scaled.csv")
        metrics[c]['unscaled'].to_csv(f"{base}{c}_unscaled.csv")

    return {
        'rmse': np.mean(rs),
        'mae': np.mean(ms)
    }

def validate_waterbody(folder, chunk, mdl, paths, classes):
    datasets = list(c(p,chunk) for c,p in zip(classes,paths))
    metrics = pd.DataFrame({ds.name:saved_validation(folder, chunk, mdl, ds) for ds in datasets})
    metrics['overall'] = metrics.mean(axis=1)
    metrics.to_csv(f"{folder}{mdl.name}_{chunk}/{datasets[0].type}/metrics.csv")
    return metrics['overall']

def validate_aquifers(folder, chunk, mdl, path):
    return validate_waterbody(folder, chunk, mdl,
        (f"{path}auser.csv", f"{path}doganella.csv", f"{path}luco.csv", f"{path}petrignano.csv"),
        (AuserDataset, DoganellaDataset, LucoDataset, PetrignanoDataset))

def validate_watersprings(folder, chunk, mdl, path):
    return validate_waterbody(folder, chunk, mdl,
        (f"{path}amiata.csv", f"{path}lupa.csv", f"{path}madonna_di_canneto.csv"),
        (AmiataDataset, LupaDataset, MadonnaDataset))

def validate_rivers(folder, chunk, mdl, path):
    return validate_waterbody(folder, chunk, mdl, (f"{path}arno.csv",), (ArnoDataset,))

def validate_lakes(folder, chunk, mdl, path):
    return validate_waterbody(folder, chunk, mdl, (f"{path}bilancino.csv",), (BilancinoDataset,))

def validate_model(folder, chunk, mdl, path):
    am = validate_aquifers(folder, chunk, mdl, f"{path}aquifer/")
    wm = validate_watersprings(folder, chunk, mdl, f"{path}waterspring/")
    rm = validate_rivers(folder, chunk, mdl, f"{path}river/")
    lm = validate_lakes(folder, chunk, mdl, f"{path}lake/")
    bodies = {'aquifer':am,'waterspring':wm,'river':rm,'lake':lm}
    scores = pd.concat(bodies, axis=1)
    scores.index = am.index
    scores['overall'] = scores.mean(axis=1)
    scores.to_csv(f"{folder}{mdl.name}_{chunk}/metrics.csv")
