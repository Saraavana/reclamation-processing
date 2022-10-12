import torch
from torch.utils.data import Dataset
import pandas as pd
from sklearn.preprocessing import LabelEncoder


class Intellizenz(Dataset):
    def __init__(self, path):
        self.path = path


        # 1. Load the data 
        data_df = pd.read_parquet(path)

        sfdfs = ['vg_state_count', 'vg_state_mean', 
        'vg_state_std', 'vg_state_min', 'vg_state_5%', 	
        'vg_state_10%', 'vg_state_15%', 'vg_state_20%', 'vg_state_25%', 'vg_state_30%', 	
        'vg_state_35%', 'vg_state_40%', 'vg_state_45%', 'vg_state_50%', 'vg_state_55%', 	
        'vg_state_60%', 'vg_state_65%', 'vg_state_70%', 'vg_state_75%', 'vg_state_80%', 	
        'vg_state_85%', 'vg_state_90%', 'vg_state_95%', 'vg_state_max', 	
        'band_count', 'band_mean', 'band_std', 'band_min', 'band_5%', 'band_10%', 'band_15%',	
        'band_20%', 'band_25%', 'band_30%', 'band_35%', 'band_40%', 'band_45%', 'band_50%', 
        'band_55%', 'band_60%', 'band_65%', 'band_70%', 'band_75%', 'band_80%', 'band_85%', 	
        'band_90%', 'band_95%', 'band_max', 
        'promoter_count', 'promoter_mean', 'promoter_std', 'promoter_min', 'promoter_5%', 
        'promoter_10%', 'promoter_15%', 'promoter_20%', 'promoter_25%', 'promoter_30%', 
        'promoter_35%', 'promoter_40%', 'promoter_45%', 'promoter_50%', 'promoter_55%', 
        'promoter_60%', 'promoter_65%', 'promoter_70%', 'promoter_75%', 'promoter_80%', 
        'promoter_85%', 'promoter_90%', 'promoter_95%', 
        #'promoter_max', 
        # 'place_kirche', 'place_hotel', 'place_cafe', 'place_theater', 'place_club', 'place_halle', 
        # 'place_gaststaette', 'place_festhalle', 'place_kulturzentrum', 'place_festzelt', 'place_schloss', 
        # 'place_pub', 'place_stadthalle', 'place_park', 'place_gasthof', 'place_kabarett', 'place_arena', 
        # 'place_schlachthof', 'place_wandelhalle', 'place_turnhalle', 'place_buergerhaus', 'place_museum', 
        # 'place_rathaus', 'place_staatsbad', 'place_zelt', 'place_jazz', 'place_forum', 'place_gymnasium', 
        # 'place_schule', 'place_sporthalle', 
        # 'band_kurorchester bad wil', 'band_musikverein harmonie', 
        # 'band_kasalla', 'band_cat ballou', 'band_roncalli royal orch', 'band_jugendblasorchester', 'band_kurorchester bad pyr', 
        # 'band_hoehner', 'band_paveier', 'band_domstuermer', 
        'band_kluengelkoepp', 'band_alleinunterhalter', 'band_the gregorian voices', 
        'band_brings', 'band_musica hungarica', 'band_concerto', 'band_bad salzuflen orches', 'band_musikverein stadtkap', 'band_salonorchester hunga', 
        'band_miljoe', 'band_raeuber', 'band_kabarett leipziger f', 'band_marita koellner', 'band_salon-orchester hung', 'band_blaeck foeoess', 
        'band_schuelerinnen und sc', 'band_romain vicente', 'band_staatliche kurkapell', 'band_musikzug der freiwil', 'band_funky marys', 
        'state_bavaria', 'state_rhineland-palatinate', 'state_baden-wuerttemberg', 'state_north rhine-westphalia', 
        # 'state_thuringia', 'state_hesse', 'state_brandenburg', 'state_schleswig-holstein', 'state_berlin', 'state_mecklenburg-western pomerania', 
        # 'state_lower saxony', 'state_hamburg', 'state_saarland', 'state_saxony-anhalt', 'state_saxony', 'state_bremen', 
        #'vg_datum_year', 
        'vg_datum_month', 'vg_datum_day_of_week', 'vg_datum_season', 'veranst_segment', 'vg_inkasso']

        X = data_df.loc[:,~data_df.columns.isin(['veranst_segment','vg_inkasso'])] # 152 features
        # X = data_df.loc[:,~data_df.columns.isin(sfdfs)] # 152 features
        y = data_df['veranst_segment']

        # Encode categorical labels
        l_enc = LabelEncoder()
        X['vg_datum_year'] = l_enc.fit_transform(X['vg_datum_year'])

        def fillmissing(df, feature):
            df[feature] = df[feature].fillna(0)
                

        features_missing= X.columns[X.isna().any()]
        for feature in features_missing:
            fillmissing(X, feature= feature)
        
        X.info()
        print('##########################################')
        y = l_enc.fit_transform(y)


        self.X = torch.Tensor(X.values) #dimension: [n, 152]
        self.y = torch.Tensor(y) #dimension: [n]

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        dataTensor = self.X[index]
        query = ":- not event(p1,{}). ".format(int(self.y[index]))
        return {'t1':dataTensor}, query


class Intellizenz_Data(Dataset):
    def __init__(self, path):
        self.path = path

        # 1. Load the data 
        data_df = pd.read_parquet(path)

        X = data_df.loc[:,~data_df.columns.isin(['veranst_segment','vg_inkasso'])] # 152 features
        y = data_df['veranst_segment']

        # Encode categorical labels
        l_enc = LabelEncoder()
        X['vg_datum_year'] = l_enc.fit_transform(X['vg_datum_year'])

        y = l_enc.fit_transform(y)

        self.X = torch.Tensor(X.values) #dimension: [n, 152]
        self.y = torch.Tensor(y) #dimension: [n]
                    
    def __getitem__(self, index):
        return self.X[index], int(self.y[index])
        
       
    def __len__(self):
        return len(self.y)
