from .clean import aquifer_pipe, waterspring_pipe, river_pipe, lake_pipe
import pandas as pd
from sklearn.preprocessing import StandardScaler

def chunk_days(df, ch):
    lose = df.shape[0] % ch
    df = df.iloc[lose:]
    newvals = {c:[] for c in df.columns}
    idx = []
    i = 0
    while i < df.shape[0]:
        vals = df.iloc[i:i+ch-1]
        idx.append(vals.iloc[0].name)
        for c in vals.columns:
            if c.startswith('Rainfall') or c.startswith('Volume'):
                newvals[c].append(vals[c].sum())
            else:
                newvals[c].append(vals[c].mean())
        i += ch
    return pd.DataFrame(newvals, index=idx, columns=df.columns)

class WaterbodyScaler:
    def __init__(self, df):
        scaler = StandardScaler()
        scaler.fit(df)
        self.cols = df.columns
        self.mu = pd.Series(scaler.mean_, index=df.columns)
        self.sigma = pd.Series(scaler.scale_, index=df.columns)
    def scale(self, df):
        df = df.copy()
        cols = [col for col in df.columns if col in self.cols]
        for c in cols:
            df[c] = (df[c] - self.mu[c]) / self.sigma[c]
        return df
    def unscale(self, df):
        df = df.copy()
        cols = [col for col in df.columns if col in self.cols]
        for c in cols:
            df[c] = df[c] * self.sigma[c] + self.mu[c]
        return df

class WaterbodyDataset:
    def __init__(self, name, type, df, y):
        self.name = name
        self.type = type
        self.targets = list(y)
        ignore = list(y) + ['Date']
        self.features = list(set(df.columns) - set(ignore))
        self.df = df
        self.X = df[self.features]
        self.y = df[self.targets]

class AquiferDataset(WaterbodyDataset):
    kind = 'aquifer'
    def __init__(self, name, df, y):
        super().__init__(name, self.kind, df, y)

class AuserDataset(AquiferDataset):
    name = 'auser'
    def __init__(self, df):
        super().__init__(self.name, df, (
            'Depth_to_Groundwater_SAL',
            'Depth_to_Groundwater_CoS',
            'Depth_to_Groundwater_LT2',
        ))

class DoganellaDataset(AquiferDataset):
    name = 'doganella'
    def __init__(self, df):
        super().__init__(self.name, df, (
            'Depth_to_Groundwater_Pozzo_1',
            'Depth_to_Groundwater_Pozzo_2',
            'Depth_to_Groundwater_Pozzo_3',
            'Depth_to_Groundwater_Pozzo_4',
            'Depth_to_Groundwater_Pozzo_5',
            'Depth_to_Groundwater_Pozzo_6',
            'Depth_to_Groundwater_Pozzo_7',
            'Depth_to_Groundwater_Pozzo_8',
            'Depth_to_Groundwater_Pozzo_9',
        ))

class LucoDataset(AquiferDataset):
    name = 'luco'
    def __init__(self, df):
        super().__init__(self.name, df, ('Depth_to_Groundwater_Podere_Casetta',))

class PetrignanoDataset(AquiferDataset):
    name = 'petrignano'
    def __init__(self, df):
        super().__init__(self.name, df, (
            'Depth_to_Groundwater_P24',
            'Depth_to_Groundwater_P25'
        ))

class WaterspringDataset(WaterbodyDataset):
    kind = 'waterspring'
    def __init__(self, name, df, y):
        super().__init__(name, self.kind, df, y)

class AmiataDataset(WaterspringDataset):
    name = 'amiata'
    def __init__(self, df):
        super().__init__(self.name, df, (
            'Flow_Rate_Bugnano',
            'Flow_Rate_Arbure',
            'Flow_Rate_Ermicciolo',
            'Flow_Rate_Galleria_Alta',
        ))

class LupaDataset(WaterspringDataset):
    name = 'lupa'
    def __init__(self, df):
        super().__init__(self.name, df, ('Flow_Rate_Lupa',))

class MadonnaDataset(WaterspringDataset):
    name = 'madonna_di_canneto'
    def __init__(self, df):
        super().__init__(self.name, df, ('Flow_Rate_Madonna_di_Canneto',))

class RiverDataset(WaterbodyDataset):
    kind = 'river'
    def __init__(self, name, df, y):
        super().__init__(name, self.kind, df, y)

class ArnoDataset(RiverDataset):
    name = 'arno'
    def __init__(self, df):
        super().__init__(self.name, df, ('Hydrometry_Nave_di_Rosano',))

class LakeDataset(WaterbodyDataset):
    kind = 'lake'
    def __init__(self, name, df, y):
        super().__init__(name, self.kind, df, y)

class BilancinoDataset(LakeDataset):
    name = 'bilancino'
    def __init__(self, df):
        super().__init__(self.name, df, ('Lake_Level', 'Flow_Rate',))
