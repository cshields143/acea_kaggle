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

class WaterbodyDataset:

    def __init__(self, cleaner, name, type, chunks, path, X, y):
        self.name = name
        self.type = type
        self.targets = y
        self.raw = pd.read_csv(path)
        self.clean = cleaner(self.raw, X, y)
        self.chunk = chunk_days(self.clean, chunks)
        scaler = StandardScaler()
        scaler.fit(self.chunk)
        self.stand = pd.DataFrame(
            scaler.transform(self.chunk),
            index=self.chunk.index,
            columns=self.chunk.columns
        )
        self.mu = pd.Series(scaler.mean_, index=self.chunk.columns)
        self.sigma = pd.Series(scaler.scale_, index=self.chunk.columns)

    def unscale(self, df):
        df = df[[c for c in df.columns if c in self.stand.columns]]
        for c in df.columns:
            df[c] = df[c] * self.sigma[c] + self.mu[c]
        return df

    def getX(self):
        return self.stand[list(set(self.stand.columns) - set(self.targets))]

    def gety(self):
        return self.stand[self.targets]

class AquiferDataset(WaterbodyDataset):

    def __init__(self, name, chunks, path, X, y):
        super().__init__(aquifer_pipe, name, 'aquifer', chunks, path, X, y)

class AuserDataset(AquiferDataset):

    def __init__(self, path, ch):
        super().__init__(
            'auser',
            ch,
            path,
            [
                'Rainfall_Gallicano',
                'Rainfall_Pontetetto',
                'Rainfall_Monte_Serra',
                'Rainfall_Orentano',
                'Rainfall_Borgo_a_Mozzano',
                'Rainfall_Piaggione',
                'Rainfall_Calavorno',
                'Rainfall_Croce_Arcana',
                'Rainfall_Tereglio_Coreglia_Antelminelli',
                'Rainfall_Fabbriche_di_Vallico',
                'Depth_to_Groundwater_PAG',
                'Depth_to_Groundwater_DIEC',
                'Temperature_Orentano',
                'Temperature_Monte_Serra',
                'Temperature_Ponte_a_Moriano',
                'Temperature_Lucca_Orto_Botanico',
                'Volume_POL',
                'Volume_CC1',
                'Volume_CC2',
                'Volume_CSA',
                'Volume_CSAL',
                'Hydrometry_Monte_S_Quirico',
                'Hydrometry_Piaggione'
            ],
            [
                'Depth_to_Groundwater_SAL',
                'Depth_to_Groundwater_CoS',
                'Depth_to_Groundwater_LT2'
            ]
        )

class DoganellaDataset(AquiferDataset):

    def __init__(self, path, ch):
        super().__init__(
            'doganella',
            ch,
            path,
            [
                'Rainfall_Monteporzio',
                'Rainfall_Velletri',
                'Volume_Pozzo_1',
                'Volume_Pozzo_2',
                'Volume_Pozzo_3',
                'Volume_Pozzo_4',
                'Volume_Pozzo_5+6',
                'Volume_Pozzo_7',
                'Volume_Pozzo_8',
                'Volume_Pozzo_9',
                'Temperature_Monteporzio',
                'Temperature_Velletri'
            ],
            [
                'Depth_to_Groundwater_Pozzo_1',
                'Depth_to_Groundwater_Pozzo_2',
                'Depth_to_Groundwater_Pozzo_3',
                'Depth_to_Groundwater_Pozzo_4',
                'Depth_to_Groundwater_Pozzo_5',
                'Depth_to_Groundwater_Pozzo_6',
                'Depth_to_Groundwater_Pozzo_7',
                'Depth_to_Groundwater_Pozzo_8',
                'Depth_to_Groundwater_Pozzo_9'
            ]
        )

class LucoDataset(AquiferDataset):

    def __init__(self, path, ch):
        super().__init__(
            'luco',
            ch,
            path,
            [
                'Rainfall_Simignano',
                'Rainfall_Siena_Poggio_al_Vento',
                'Rainfall_Mensano',
                'Rainfall_Montalcinello',
                'Rainfall_Monticiano_la_Pineta',
                'Rainfall_Sovicille',
                'Rainfall_Ponte_Orgia',
                'Rainfall_Scorgiano',
                'Rainfall_Pentolina',
                'Rainfall_Monteroni_Arbia_Biena',
                'Depth_to_Groundwater_Pozzo_1',
                'Depth_to_Groundwater_Pozzo_3',
                'Depth_to_Groundwater_Pozzo_4',
                'Temperature_Siena_Poggio_al_Vento',
                'Temperature_Mensano',
                'Temperature_Pentolina',
                'Temperature_Monteroni_Arbia_Biena',
                'Volume_Pozzo_1',
                'Volume_Pozzo_3',
                'Volume_Pozzo_4'
            ],
            [
                'Depth_to_Groundwater_Podere_Casetta'
            ]
        )

class PetrignanoDataset(AquiferDataset):

    def __init__(self, path, ch):
        super().__init__(
            'petrignano',
            ch,
            path,
            [
                'Rainfall_Bastia_Umbra',
                'Temperature_Bastia_Umbra',
                'Temperature_Petrignano',
                'Volume_C10_Petrignano',
                'Hydrometry_Fiume_Chiascio_Petrignano'
            ],
            [
                'Depth_to_Groundwater_P24',
                'Depth_to_Groundwater_P25'
            ]
        )

class WaterspringDataset(WaterbodyDataset):
    def __init__(self, name, chunks, path, X, y):
        super().__init__(waterspring_pipe, name, 'waterspring', chunks, path, X, y)

class AmiataDataset(WaterspringDataset):
    def __init__(self, path, ch):
        super().__init__(
            'amiata',
            ch,
            path,
            [
                'Rainfall_Castel_del_Piano',
                'Rainfall_Abbadia_S_Salvatore',
                'Rainfall_S_Fiora',
                'Rainfall_Laghetto_Verde',
                'Rainfall_Vetta_Amiata',
                'Depth_to_Groundwater_S_Fiora_8',
                'Depth_to_Groundwater_S_Fiora_11bis',
                'Depth_to_Groundwater_David_Lazzaretti',
                'Temperature_Abbadia_S_Salvatore',
                'Temperature_S_Fiora',
                'Temperature_Laghetto_Verde'
            ],
            [
                'Flow_Rate_Bugnano',
                'Flow_Rate_Arbure',
                'Flow_Rate_Ermicciolo',
                'Flow_Rate_Galleria_Alta'
            ]
        )

class LupaDataset(WaterspringDataset):
    def __init__(self, path, ch):
        super().__init__('lupa', ch, path, ['Rainfall_Terni'], ['Flow_Rate_Lupa'])

class MadonnaDataset(WaterspringDataset):
    def __init__(self, path, ch):
        super().__init__('madonna di canneto', ch, path,
            ['Rainfall_Settefrati', 'Temperature_Settefrati'],
            ['Flow_Rate_Madonna_di_Canneto'])

class RiverDataset(WaterbodyDataset):
    def __init__(self, name, chunks, path, X, y):
        super().__init__(river_pipe, name, 'river', chunks, path, X, y)

class ArnoDataset(RiverDataset):
    def __init__(self, path, ch):
        super().__init__(
            'arno',
            ch,
            path,
            [
                'Rainfall_Le_Croci',
                'Rainfall_Cavallina',
                'Rainfall_S_Agata',
                'Rainfall_Mangona',
                'Rainfall_S_Piero',
                'Rainfall_Vernio',
                'Rainfall_Stia',
                'Rainfall_Consuma',
                'Rainfall_Incisa',
                'Rainfall_Montevarchi',
                'Rainfall_S_Savino',
                'Rainfall_Laterina',
                'Rainfall_Bibbiena',
                'Rainfall_Camaldoli',
                'Temperature_Firenze'
            ],
            [
                'Hydrometry_Nave_di_Rosano'
            ]
        )

class LakeDataset(WaterbodyDataset):
    def __init__(self, name, chunks, path, X, y):
        super().__init__(lake_pipe, name, 'lake', chunks, path, X, y)

class BilancinoDataset(LakeDataset):
    def __init__(self, path, ch):
        super().__init__('bilancino', ch, path, [
            'Rainfall_S_Piero',
            'Rainfall_Mangona',
            'Rainfall_S_Agata',
            'Rainfall_Cavallina',
            'Rainfall_Le_Croci',
            'Temperature_Le_Croci'
        ], ['Lake_Level', 'Flow_Rate'])
