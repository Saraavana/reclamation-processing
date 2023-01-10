from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

from tqdm.auto import tqdm

tqdm.pandas()

### Transform the Promoter column to remove repetitive company name
# Remove the repeating organisation(company types) substring from Organizer/Promoter
def transform_promoter(x, substrings):
    str_value = x
    str_value = str_value.replace('K. D. OE. R','K.D.OE.R')

    for subs in substrings:
        if str_value.count(subs) == 2:
            str_value = str_value.replace(subs,'X',1) # replace 1st occurance of the string with X
            str_value = str_value.replace(subs,'').strip() # replace 2st occurance of the string with empty
            str_value = str_value.replace('X', subs) # replace X with substring value
            return str_value
        else:
            return str_value

def get_train_test_df():
    df = pd.read_parquet('C:/Users/sgopalakrish/Downloads/intellizenz-model-training/data/export_features_2016_2020_v3.parquet.gzip')

    df=df.rename(columns = {'band_x':'band', 'vg_state_x':'vg_state','vg_raum_wo_stopwords':'venue'})

    orgs = ['GMBH & CO. KG', 'E.V', 'GMBH', 'GBR', 'K.D.OE.R', 'OHG']

    df['promoter_transform'] = df.apply(lambda x: transform_promoter(x['veranst_name'], substrings=[orgs[0]]), axis=1)
    df['promoter_transform'] = df.apply(lambda x: transform_promoter(x['promoter_transform'], substrings=[orgs[1]]), axis=1)
    df['promoter_transform'] = df.apply(lambda x: transform_promoter(x['promoter_transform'], substrings=[orgs[2]]), axis=1)
    df['promoter_transform'] = df.apply(lambda x: transform_promoter(x['promoter_transform'], substrings=[orgs[3]]), axis=1)
    df['promoter_transform'] = df.apply(lambda x: transform_promoter(x['promoter_transform'], substrings=[orgs[4]]), axis=1)
    df['promoter_transform'] = df.apply(lambda x: transform_promoter(x['promoter_transform'], substrings=[orgs[5]]), axis=1)

    df_train, df_test = train_test_split(df, test_size = 0.20, random_state=1)

    print(df_train.shape)
    print(df_test.shape)

    df_train_featurize = df_train[[
        'vg_inkasso', 'veranst_segment', 'venue', 'vg_state', 'band', 'promoter_transform',
        'vg_datum_year', 'vg_datum_month', 'vg_datum_day_of_week', 'vg_datum_season',
        'tarif_bez'
    ]].copy()

    df_test_featurize = df_test[[
        'vg_inkasso', 'veranst_segment', 'venue', 'vg_state', 'band', 'promoter_transform',
        'vg_datum_year', 'vg_datum_month', 'vg_datum_day_of_week', 'vg_datum_season',
        'tarif_bez'
    ]].copy()

    return df_train_featurize, df_test_featurize

def get_df():
    df = pd.read_parquet('C:/Users/sgopalakrish/Downloads/intellizenz-model-training/data/export_features_2016_2020_v3.parquet.gzip')

    df=df.rename(columns = {'band_x':'band', 'vg_state_x':'vg_state','vg_raum_wo_stopwords':'venue'})

    orgs = ['GMBH & CO. KG', 'E.V', 'GMBH', 'GBR', 'K.D.OE.R', 'OHG']

    df['promoter_transform'] = df.apply(lambda x: transform_promoter(x['veranst_name'], substrings=[orgs[0]]), axis=1)
    df['promoter_transform'] = df.apply(lambda x: transform_promoter(x['promoter_transform'], substrings=[orgs[1]]), axis=1)
    df['promoter_transform'] = df.apply(lambda x: transform_promoter(x['promoter_transform'], substrings=[orgs[2]]), axis=1)
    df['promoter_transform'] = df.apply(lambda x: transform_promoter(x['promoter_transform'], substrings=[orgs[3]]), axis=1)
    df['promoter_transform'] = df.apply(lambda x: transform_promoter(x['promoter_transform'], substrings=[orgs[4]]), axis=1)
    df['promoter_transform'] = df.apply(lambda x: transform_promoter(x['promoter_transform'], substrings=[orgs[5]]), axis=1)

    print(df.shape)

    df_featurize = df[[
        'vg_inkasso', 'veranst_segment', 'venue', 'vg_state', 'band', 'promoter_transform',
        'vg_datum_year', 'vg_datum_month', 'vg_datum_day_of_week', 'vg_datum_season',
        'tarif_bez'
    ]].copy()

    df_featurize=df_featurize.rename(columns = {'promoter_transform':'promoter'})

    return df_featurize

def extract_leave_one_out_features(df, stat_var, save_path):
    
    df_stat = {}

    def get_descr_stat(row):
        def descr_stat(row, var):
            result = pd.Series(dtype='float64')

            if not pd.isnull(row[var]):
                inkasso = df_stat[var].loc[row[var]].copy()
                if len(inkasso) > 1:
                    inkasso.remove(row['vg_inkasso'])
                    result = pd.Series(inkasso).describe(percentiles=percentiles)
                    result = result.add_prefix('{}_'.format(var))

            result.name = row.name
            return result

        descr_stat_result = pd.Series(dtype='float64')
        for v in stat_var:
            # descr_stat_result = descr_stat_result.append(descr_stat(row, v))
            descr_stat_result = pd.concat([descr_stat_result, descr_stat(row, v)])

        return descr_stat_result

    for v in tqdm(stat_var):
        df_stat[v] = df.groupby(v)['vg_inkasso'].apply(list)

    percentiles = [round(x, 2) for x in np.linspace(0, 1, 21)[1:-1].tolist()]

    df_descr_stat = df.progress_apply(get_descr_stat, axis=1)

    df_descr_stat.to_parquet(save_path,compression='gzip')



# train_df, test_df = get_train_test_df()

# train_band_save_path = './data/export_train_band_descr_stats_2016_2020_v2.parquet.gzip'
# train_promoter_save_path = './data/export_train_promoter_descr_stats_2016_2020_v2.parquet.gzip'
# train_venue_save_path = './data/export_train_venue_descr_stats_2016_2020_v2.parquet.gzip'
# train_tarif_save_path = './data/export_train_tarif_descr_stats_2016_2020_v2.parquet.gzip'

# test_band_save_path = './data/export_test_band_descr_stats_2016_2020_v2.parquet.gzip'
# test_promoter_save_path = './data/export_test_promoter_descr_stats_2016_2020_v2.parquet.gzip'
# test_venue_save_path = './data/export_test_venue_descr_stats_2016_2020_v2.parquet.gzip'
# test_tarif_save_path = './data/export_test_tarif_descr_stats_2016_2020_v2.parquet.gzip'


# ### Featurize testing dataframe
# extract_leave_one_out_features(test_df, stat_var=['band'], save_path=test_band_save_path)
# extract_leave_one_out_features(test_df, stat_var=['promoter_transform'], save_path=test_promoter_save_path)
# extract_leave_one_out_features(test_df, stat_var=['venue'], save_path=test_venue_save_path)
# # extract_leave_one_out_features(test_df, stat_var=['tarif_bez'], save_path=test_tarif_save_path)

# ### Featurize training dataframe
# extract_leave_one_out_features(train_df, stat_var=['band'], save_path=train_band_save_path)
# extract_leave_one_out_features(train_df, stat_var=['promoter_transform'], save_path=train_promoter_save_path)
# extract_leave_one_out_features(train_df, stat_var=['venue'], save_path=train_venue_save_path)
# # extract_leave_one_out_features(train_df, stat_var=['tarif_bez'], save_path=train_tarif_save_path)


df = get_df()

band_save_path = './data/export_band_descr_stats_2016_2020_v3.parquet.gzip'
promoter_save_path = './data/export_promoter_descr_stats_2016_2020_v3.parquet.gzip'
venue_save_path = './data/export_venue_descr_stats_2016_2020_v3.parquet.gzip'
tarif_save_path = './data/export_tarif_descr_stats_2016_2020_v3.parquet.gzip'

extract_leave_one_out_features(df, stat_var=['band'], save_path=band_save_path)
extract_leave_one_out_features(df, stat_var=['promoter'], save_path=promoter_save_path)
extract_leave_one_out_features(df, stat_var=['venue'], save_path=venue_save_path)
# extract_leave_one_out_features(df, stat_var=['tarif_bez'], save_path=tarif_save_path)