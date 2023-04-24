data_path_2016_2020_v3 = 'C:/Users/sgopalakrish/Downloads/intellizenz-model-training/data/export_features_2016_2020_v3.parquet.gzip'

anony_data_path_2016_2020_v1 = '/Users/saravana/Documents/Work/Master-Thesis/reclamation-processing/data/export_anonymized_features_2016_2020_v1.parquet.gzip'

train_data_path = 'C:/Users/sgopalakrish/Downloads/intellizenz-model-training/data/export_training_features_2016_2020_v2.parquet.gzip'
test_data_path = 'C:/Users/sgopalakrish/Downloads/intellizenz-model-training/data/export_testing_features_2016_2020_v2.parquet.gzip'

# label encoded tarif-bez - with leave-one-hot- target encoding features 
data_path_2016_2020_v4 = 'C:/Users/sgopalakrish/Downloads/intellizenz-model-training/data/export_features_2016_2020_v4.parquet.gzip'
# label unencoded tarif-bez - with leave-one-hot- target encoding features
data_path_2016_2020_v5 = '/Users/saravana/Documents/Work/Master-Thesis/reclamation-processing/data/export_features_2016_2020_v5.parquet.gzip'

# 30 venue, 30 tarif, 30 band, 30 promoter , 16 state, 4 seasonality
anonymized_features_v1 = [
#  4 seasonality
'vg_datum_season',	'vg_datum_month', 'vg_datum_day_of_week', 'vg_datum_year',

# 30 venues
'place_kirche', 'place_hotel', 'place_cafe',
'place_theater', 'place_club', 'place_halle',	
'place_kulturzentrum', 'place_gaststaette', 'place_buergerhaus',	
'place_festhalle', 'place_stadthalle', 'place_festzelt', 
'place_schloss', 'place_pub', 'place_restaurant', 
'place_gasthaus', 'place_bar', 'place_kurhaus', 
'place_kulturhaus', 'place_kabarett', 'place_rathaus', 
'place_arena', 'place_gasthof', 'place_park', 
'place_wandelhalle', 'place_schlachthof', 'place_turnhalle', 
'place_staatsbad', 'place_zelt', 'place_mehrzweckhalle', 

# 30 tarifs
'tarif_u-v ii. 1 (+ii 2)', 'tarif_u-k (musiker)', 'tarif_u-k (musiker) mindestverguetung', 
'tarif_u-st i. (musiker)', 'tarif_u-k i (+ ii 2b) - mit sonstigem geldwerten vorteil', 
'tarif_u-st i (musiker) nl', 'tarif_u-v iii. 1', 'tarif_u-v iii. 2', 
'tarif_u-k iii. 2d) (musiker) - vor geladenen gaesten', 'tarif_variete i (musiker)', 
'tarif_u-v vi. b', 'tarif_e (musiker)', 'tarif_p-k i. (u-musik)', 
'tarif_u-k ii. (musiker) bis 50 min', 'tarif_u-k ii. (musiker) bis 20 min', 'tarif_e-p', 
'tarif_u-k ii. (musiker) bis 25 min', 'tarif_u-k ii. (musiker) bis 15 min',	
'tarif_u-k ii. (musiker) bis 30 min', 'tarif_u-k ii. (musiker) bis 10 min', 'tarif_u-st (musiker)', 
'tarif_chorverband konzert u-musik', 'tarif_vk i 3 zirkusunternehmen (musiker)', 
'tarif_u-k ii. (musiker) bis 5 min', 'tarif_u-k ii. (musiker) bis 35 min', 'tarif_u-k ii. (musiker) bis 40 min',
'tarif_p-k i. (e-musik)', 'tarif_u-k ii (musiker) mindestverguetung', 
'tarif_u-k ii. (musiker) bis 45 min', 'tarif_u-v vi. b mindestverguetung', 

# 16 states
'state_bavaria', 'state_baden-wuerttemberg', 'state_north rhine-westphalia', 
'state_hesse', 'state_lower saxony', 'state_berlin', 
'state_rhineland-palatinate', 'state_saxony', 'state_hamburg', 
'state_thuringia', 'state_schleswig-holstein', 'state_brandenburg', 
'state_saxony-anhalt', 'state_mecklenburg-western pomerania', 'state_saarland', 
'state_bremen', 

# 30 bands
'band_hofmann ortmann ag', 'band_christoph löffler ag', 'band_margraf döring gbr', 
'band_huhn trüb ag & co. ohg', 'band_finke haering ag', 'band_pergande neuschäfer ag', 
'band_hornig täsche ohg mbh', 'band_ziegert scheel ag', 'band_adolph heser gmbh & co. ohg', 
'band_hübel eckbauer ag', 'band_kensy gute gmbh & co. kgaa', 'band_ring flantz stiftung & co. kg', 
'band_wende scheuermann gmbh & co. kg', 'band_pärtzelt bolnbach kg', 'band_bachmann ernst kg', 
'band_hamann hethur stiftung & co. kgaa', 'band_nohlmans roskoth e.v.', 'band_schönland koch ii ag', 
'band_hermighausen süßebier gmbh', 'band_juncken briemer stiftung & co. kg', 
'band_lachmann heinrich gmbh & co. ohg', 'band_hendriks döring ag', 'band_bähr weihmann ag', 
'band_beer reising ag', 'band_stumpf ritter gmbh & co. kgaa', 'band_thies stiebitz gbr', 
'band_bärer jungfer gmbh & co. kg', 'band_trubin butte e.g.', 'band_römer steckel gmbh & co. kg', 
'band_wiek schmiedt gmbh & co. ohg', 

# 30 promoters
'promoter_schmidtke reichmann gmbh & co. kg', 'promoter_heintze ag & co. kg', 'promoter_drubin gmbh', 
'promoter_textor kg', 'promoter_schleich stiftung & co. kg', 'promoter_fiebig weinhage ag & co. kgaa', 
'promoter_zobel carsten gbr', 'promoter_finke hethur e.g.', 'promoter_fröhlich sölzer gmbh', 
'promoter_seip scholtz gmbh', 'promoter_margraf dörr ag & co. kg', 'promoter_bolzmann caspar gmbh & co. ohg', 
'promoter_franke franke ag', 'promoter_sontag gmbh & co. ohg', 'promoter_wesack riehl gmbh & co. kg', 
'promoter_röhrdanz', 'promoter_juncken eckbauer e.v.', 'promoter_gutknecht fechner ag', 
'promoter_knappe ag', 'promoter_flantz kensy gmbh & co. ohg', 'promoter_kohl kabus ag & co. kg', 
'promoter_gunpf hertrampf ohg mbh', 'promoter_förster van der dussen ag & co. kg', 'promoter_mude eckbauer gmbh', 
'promoter_metz ohg mbh', 'promoter_franke langern stiftung & co. kg', 'promoter_biggen hermann ag & co. ohg', 
'promoter_wähner haase gmbh & co. kg', 'promoter_dippel koch kg', 'promoter_kitzmann ag', 
]

# 30 venue, 30 band, 16 state, 4 seasonality, 2 target features
features_v1 = [
            'place_kirche', 'place_hotel', 'place_cafe',
            'place_theater', 'place_club', 'place_halle',
            'place_gaststaette', 'place_festhalle', 'place_kulturzentrum',
            'place_festzelt', 'place_schloss', 'place_pub',
            'place_stadthalle', 'place_park', 'place_gasthof',
            'place_kabarett', 'place_arena', 'place_schlachthof',
            'place_wandelhalle', 'place_turnhalle', 'place_buergerhaus',
            'place_museum', 'place_rathaus', 'place_staatsbad',
            'place_zelt', 'place_jazz', 'place_forum',
            'place_gymnasium', 'place_schule', 'place_sporthalle', 

            #30 bands

            'state_bavaria','state_rhineland-palatinate',
            'state_baden-wuerttemberg',	'state_north rhine-westphalia',	
            'state_thuringia','state_hesse',	
            'state_brandenburg', 'state_schleswig-holstein',	
            'state_berlin',	'state_mecklenburg-western pomerania',	
            'state_lower saxony', 'state_hamburg',	
            'state_saarland', 'state_saxony-anhalt',	
            'state_saxony',	'state_bremen',

            'vg_datum_year','vg_datum_month','vg_datum_day_of_week','vg_datum_season', 

            'veranst_segment','vg_inkasso'
        ]

# 140 Feautes used - 30 common features('VG_RAUM_KEYWORDS', 'BAND', 'PROMOTER', 'tariffs') 'vg_state', 'VG_DATUM_VON'
# 30 venue, 30 tarif, 30 band, 30 promoter , 16 state, 4 seasonality
features_v2 = ['place_kirche', 'place_hotel', 'place_cafe',
 'place_theater', 'place_club', 'place_halle',
 'place_gaststaette', 'place_festhalle', 'place_kulturzentrum',
 'place_festzelt', 'place_schloss', 'place_pub',
 'place_stadthalle', 'place_park', 'place_gasthof',
 'place_kabarett', 'place_arena', 'place_schlachthof',
 'place_wandelhalle', 'place_turnhalle', 'place_buergerhaus',
 'place_museum', 'place_rathaus', 'place_staatsbad',
 'place_zelt', 'place_jazz', 'place_forum',
 'place_gymnasium', 'place_schule', 'place_sporthalle',

'tarif_u-v ii. 1 (+ii 2)', 'tarif_u-k (musiker)', 'tarif_u-k (musiker) mindestverguetung', 
'tarif_u-k i (+ ii 2b) - mit sonstigem geldwerten vorteil', 'tarif_u-st i (musiker) nl', 'tarif_u-v iii. 1', 
'tarif_u-v iii. 2', 'tarif_u-k iii. 2d) (musiker) - vor geladenen gaesten', 'tarif_variete i (musiker)', 
'tarif_u-v vi. b', 'tarif_e (musiker)', 'tarif_p-k i. (u-musik)', 
'tarif_u-k ii. (musiker) bis 50 min', 'tarif_u-k ii. (musiker) bis 20 min', 'tarif_e-p', 
'tarif_u-k ii. (musiker) bis 15 min', 'tarif_u-k ii. (musiker) bis 25 min', 'tarif_u-k ii. (musiker) bis 30 min', 
'tarif_u-st (musiker)', 'tarif_u-k ii. (musiker) bis 10 min', 'tarif_chorverband konzert u-musik',
'tarif_vk i 3 zirkusunternehmen (musiker)', 'tarif_u-k ii. (musiker) bis 5 min', 'tarif_u-k ii. (musiker) bis 35 min', 
'tarif_u-k ii. (musiker) bis 40 min', 'tarif_p-k i. (e-musik)', 'tarif_u-st i. (musiker)', 
'tarif_u-k ii (musiker) mindestverguetung', 'tarif_u-k ii. (musiker) bis 45 min', 'tarif_u-v vi. b mindestverguetung', 

# 30 bands


'state_bavaria','state_rhineland-palatinate',
'state_baden-wuerttemberg',	'state_north rhine-westphalia',	
'state_thuringia','state_hesse',	
'state_brandenburg', 'state_schleswig-holstein',	
'state_berlin',	'state_mecklenburg-western pomerania',	
'state_lower saxony', 'state_hamburg',	
'state_saarland', 'state_saxony-anhalt',	
'state_saxony',	'state_bremen',

'vg_datum_year','vg_datum_month','vg_datum_day_of_week','vg_datum_season',

# 30 promoters

]

# 142 Feautes used - 'VG_RAUM_KEYWORDS', 'VG_DATUM_VON', 'vg_state', 'BAND', 'PROMOTER', TARIF_BEZ
# 30 venue, 30 tarif, 30 band, 30 promoter , 16 state, 4 seasonality, 2 target featuers
features_v3 = ['place_kirche', 'place_hotel', 'place_cafe',
 'place_theater', 'place_club', 'place_halle',
 'place_gaststaette', 'place_festhalle', 'place_kulturzentrum',
 'place_festzelt', 'place_schloss', 'place_pub',
 'place_stadthalle', 'place_park', 'place_gasthof',
 'place_kabarett', 'place_arena', 'place_schlachthof',
 'place_wandelhalle', 'place_turnhalle', 'place_buergerhaus',
 'place_museum', 'place_rathaus', 'place_staatsbad',
 'place_zelt', 'place_jazz', 'place_forum',
 'place_gymnasium', 'place_schule', 'place_sporthalle',

'tarif_u-v ii. 1 (+ii 2)', 'tarif_u-k (musiker)', 'tarif_u-k (musiker) mindestverguetung', 
'tarif_u-k i (+ ii 2b) - mit sonstigem geldwerten vorteil', 'tarif_u-st i (musiker) nl', 'tarif_u-v iii. 1', 
'tarif_u-v iii. 2', 'tarif_u-k iii. 2d) (musiker) - vor geladenen gaesten', 'tarif_variete i (musiker)', 
'tarif_u-v vi. b', 'tarif_e (musiker)', 'tarif_p-k i. (u-musik)', 
'tarif_u-k ii. (musiker) bis 50 min', 'tarif_u-k ii. (musiker) bis 20 min', 'tarif_e-p', 
'tarif_u-k ii. (musiker) bis 15 min', 'tarif_u-k ii. (musiker) bis 25 min', 'tarif_u-k ii. (musiker) bis 30 min', 
'tarif_u-st (musiker)', 'tarif_u-k ii. (musiker) bis 10 min', 'tarif_chorverband konzert u-musik',
'tarif_vk i 3 zirkusunternehmen (musiker)', 'tarif_u-k ii. (musiker) bis 5 min', 'tarif_u-k ii. (musiker) bis 35 min', 
'tarif_u-k ii. (musiker) bis 40 min', 'tarif_p-k i. (e-musik)', 'tarif_u-st i. (musiker)', 
'tarif_u-k ii (musiker) mindestverguetung', 'tarif_u-k ii. (musiker) bis 45 min', 'tarif_u-v vi. b mindestverguetung', 

#30 band


'state_bavaria','state_rhineland-palatinate',
'state_baden-wuerttemberg',	'state_north rhine-westphalia',	
'state_thuringia','state_hesse',	
'state_brandenburg', 'state_schleswig-holstein',	
'state_berlin',	'state_mecklenburg-western pomerania',	
'state_lower saxony', 'state_hamburg',	
'state_saarland', 'state_saxony-anhalt',	
'state_saxony',	'state_bremen',

'vg_datum_year','vg_datum_month','vg_datum_day_of_week','vg_datum_season',

#30 promoters

'veranst_segment','vg_inkasso'
]

# 238 Feautes used - 'VG_RAUM_KEYWORDS', 'VG_DATUM_VON', 'vg_state', 'BAND', 'PROMOTER', TARIF_BEZ, vg_state_percentiles,
# promoter_percentiles, band_percentiles, tarif_percentiles
# 30 venue, 30 tarif, 30 band, 30 promoter , 16 state, 4 seasonality, each 24(band, state, promoter, tarif) percentiles, 2 target feature

features_v4 = ['place_kirche', 'place_hotel', 'place_cafe',
 'place_theater', 'place_club', 'place_halle',
 'place_gaststaette', 'place_festhalle', 'place_kulturzentrum',
 'place_festzelt', 'place_schloss', 'place_pub',
 'place_stadthalle', 'place_park', 'place_gasthof',
 'place_kabarett', 'place_arena', 'place_schlachthof',
 'place_wandelhalle', 'place_turnhalle', 'place_buergerhaus',
 'place_museum', 'place_rathaus', 'place_staatsbad',
 'place_zelt', 'place_jazz', 'place_forum',
 'place_gymnasium', 'place_schule', 'place_sporthalle',

'tarif_u-v ii. 1 (+ii 2)', 'tarif_u-k (musiker)', 'tarif_u-k (musiker) mindestverguetung', 
'tarif_u-k i (+ ii 2b) - mit sonstigem geldwerten vorteil', 'tarif_u-st i (musiker) nl', 'tarif_u-v iii. 1', 
'tarif_u-v iii. 2', 'tarif_u-k iii. 2d) (musiker) - vor geladenen gaesten', 'tarif_variete i (musiker)', 
'tarif_u-v vi. b', 'tarif_e (musiker)', 'tarif_p-k i. (u-musik)', 
'tarif_u-k ii. (musiker) bis 50 min', 'tarif_u-k ii. (musiker) bis 20 min', 'tarif_e-p', 
'tarif_u-k ii. (musiker) bis 15 min', 'tarif_u-k ii. (musiker) bis 25 min', 'tarif_u-k ii. (musiker) bis 30 min', 
'tarif_u-st (musiker)', 'tarif_u-k ii. (musiker) bis 10 min', 'tarif_chorverband konzert u-musik',
'tarif_vk i 3 zirkusunternehmen (musiker)', 'tarif_u-k ii. (musiker) bis 5 min', 'tarif_u-k ii. (musiker) bis 35 min', 
'tarif_u-k ii. (musiker) bis 40 min', 'tarif_p-k i. (e-musik)', 'tarif_u-st i. (musiker)', 
'tarif_u-k ii (musiker) mindestverguetung', 'tarif_u-k ii. (musiker) bis 45 min', 'tarif_u-v vi. b mindestverguetung', 

# 30 bands


'state_bavaria','state_rhineland-palatinate',
'state_baden-wuerttemberg',	'state_north rhine-westphalia',	
'state_thuringia','state_hesse',	
'state_brandenburg', 'state_schleswig-holstein',	
'state_berlin',	'state_mecklenburg-western pomerania',	
'state_lower saxony', 'state_hamburg',	
'state_saarland', 'state_saxony-anhalt',	
'state_saxony',	'state_bremen',

'vg_datum_year','vg_datum_month','vg_datum_day_of_week','vg_datum_season',

# 30 promoters

'vg_state_count', 'vg_state_mean', 'vg_state_std', 'vg_state_min', 
'vg_state_5%', 'vg_state_10%', 'vg_state_15%', 'vg_state_20%', 
'vg_state_25%', 'vg_state_30%', 'vg_state_35%', 'vg_state_40%', 
'vg_state_45%', 'vg_state_50%', 'vg_state_55%', 'vg_state_60%', 
'vg_state_65%', 'vg_state_70%', 'vg_state_75%', 'vg_state_80%', 
'vg_state_85%', 'vg_state_90%', 'vg_state_95%', 'vg_state_max',

'band_count', 'band_mean', 'band_std', 'band_min', 
'band_5%', 'band_10%', 'band_15%',	'band_20%', 
'band_25%', 'band_30%', 'band_35%', 'band_40%', 
'band_45%', 'band_50%', 'band_55%', 'band_60%', 
'band_65%', 'band_70%', 'band_75%', 'band_80%', 
'band_85%', 'band_90%', 'band_95%', 'band_max', 

'promoter_count', 'promoter_mean', 'promoter_std', 'promoter_min', 
'promoter_5%', 'promoter_10%', 'promoter_15%', 'promoter_20%', 
'promoter_25%', 'promoter_30%', 'promoter_35%', 'promoter_40%', 
'promoter_45%', 'promoter_50%', 'promoter_55%', 'promoter_60%', 
'promoter_65%', 'promoter_70%', 'promoter_75%', 'promoter_80%', 
'promoter_85%', 'promoter_90%', 'promoter_95%', 'promoter_max', 

'tarif_bez_count', 'tarif_bez_mean', 'tarif_bez_std', 'tarif_bez_min', 
'tarif_bez_5%', 'tarif_bez_10%', 'tarif_bez_15%', 'tarif_bez_20%', 
'tarif_bez_25%', 'tarif_bez_30%', 'tarif_bez_35%', 'tarif_bez_40%', 
'tarif_bez_45%', 'tarif_bez_50%', 'tarif_bez_55%', 'tarif_bez_60%', 
'tarif_bez_65%', 'tarif_bez_70%', 'tarif_bez_75%', 'tarif_bez_80%', 
'tarif_bez_85%', 'tarif_bez_90%', 'tarif_bez_95%', 'tarif_bez_max',

'veranst_segment','vg_inkasso'
]

# 143 Feautes used - 'VG_RAUM_KEYWORDS', 'VG_DATUM_VON', 'vg_state', 'BAND', 'PROMOTER', TARIF_BEZ
# 30 venue, 30 tarif, 30 band, 30 promoter , 16 state, 4 seasonality, 2 target features, tarif_bez
features_v5 = ['place_kirche', 'place_hotel', 'place_cafe',
 'place_theater', 'place_club', 'place_halle',
 'place_gaststaette', 'place_festhalle', 'place_kulturzentrum',
 'place_festzelt', 'place_schloss', 'place_pub',
 'place_stadthalle', 'place_park', 'place_gasthof',
 'place_kabarett', 'place_arena', 'place_schlachthof',
 'place_wandelhalle', 'place_turnhalle', 'place_buergerhaus',
 'place_museum', 'place_rathaus', 'place_staatsbad',
 'place_zelt', 'place_jazz', 'place_forum',
 'place_gymnasium', 'place_schule', 'place_sporthalle',

'tarif_u-v ii. 1 (+ii 2)', 'tarif_u-k (musiker)', 'tarif_u-k (musiker) mindestverguetung', 
'tarif_u-k i (+ ii 2b) - mit sonstigem geldwerten vorteil', 'tarif_u-st i (musiker) nl', 'tarif_u-v iii. 1', 
'tarif_u-v iii. 2', 'tarif_u-k iii. 2d) (musiker) - vor geladenen gaesten', 'tarif_variete i (musiker)', 
'tarif_u-v vi. b', 'tarif_e (musiker)', 'tarif_p-k i. (u-musik)', 
'tarif_u-k ii. (musiker) bis 50 min', 'tarif_u-k ii. (musiker) bis 20 min', 'tarif_e-p', 
'tarif_u-k ii. (musiker) bis 15 min', 'tarif_u-k ii. (musiker) bis 25 min', 'tarif_u-k ii. (musiker) bis 30 min', 
'tarif_u-st (musiker)', 'tarif_u-k ii. (musiker) bis 10 min', 'tarif_chorverband konzert u-musik',
'tarif_vk i 3 zirkusunternehmen (musiker)', 'tarif_u-k ii. (musiker) bis 5 min', 'tarif_u-k ii. (musiker) bis 35 min', 
'tarif_u-k ii. (musiker) bis 40 min', 'tarif_p-k i. (e-musik)', 'tarif_u-st i. (musiker)', 
'tarif_u-k ii (musiker) mindestverguetung', 'tarif_u-k ii. (musiker) bis 45 min', 'tarif_u-v vi. b mindestverguetung', 

# 30 bands


'state_bavaria','state_rhineland-palatinate',
'state_baden-wuerttemberg',	'state_north rhine-westphalia',	
'state_thuringia','state_hesse',	
'state_brandenburg', 'state_schleswig-holstein',	
'state_berlin',	'state_mecklenburg-western pomerania',	
'state_lower saxony', 'state_hamburg',	
'state_saarland', 'state_saxony-anhalt',	
'state_saxony',	'state_bremen',

'vg_datum_year','vg_datum_month','vg_datum_day_of_week','vg_datum_season',

# 30 promoters

'veranst_segment','vg_inkasso', 'tarif_bez'
]

# 80 Feautes used - Leave-one-out-target-encoded 'band', 'promoter', 'venue' &
# 'vg_inkasso', 'veranst_segment', 'vg_state', 'vg_datum_year', 
#  'vg_datum_month', 'vg_datum_day_of_week', 'vg_datum_season', 'tarif_bez'
features_v6 = [
    'venue_10%', 'venue_15%', 'venue_20%', 'venue_25%', 'venue_30%', 
    'venue_35%', 'venue_40%', 'venue_45%', 'venue_5%', 'venue_50%', 
    'venue_55%', 'venue_60%', 'venue_65%', 'venue_70%', 'venue_75%', 
    'venue_80%', 'venue_85%', 'venue_90%', 'venue_95%', 'venue_count', 
    'venue_max', 'venue_mean', 'venue_min', 'venue_std', 
    
    'band_10%', 'band_15%', 'band_20%', 'band_25%', 'band_30%', 
    'band_35%', 'band_40%', 'band_45%', 'band_5%', 'band_50%', 
    'band_55%', 'band_60%', 'band_65%', 'band_70%', 'band_75%', 
    'band_80%', 'band_85%', 'band_90%', 'band_95%', 'band_count', 
    'band_max', 'band_mean', 'band_min', 'band_std', 

    'promoter_10%', 'promoter_15%', 'promoter_20%', 'promoter_25%', 'promoter_30%', 
    'promoter_35%', 'promoter_40%', 'promoter_45%', 'promoter_5%', 'promoter_50%', 
    'promoter_55%', 'promoter_60%', 'promoter_65%', 'promoter_70%', 'promoter_75%', 
    'promoter_80%', 'promoter_85%', 'promoter_90%', 'promoter_95%', 'promoter_count', 
    'promoter_max', 'promoter_mean', 'promoter_min', 'promoter_std',
    
    'vg_inkasso', 'veranst_segment', 'vg_state', 'vg_datum_year', 
    'vg_datum_month', 'vg_datum_day_of_week', 'vg_datum_season', 'tarif_bez'
    ]

# 77 Feautes used - Leave-one-out-target-encoded 'band', 'promoter', 'venue' &
#  'vg_state', 'vg_datum_year', 
#  'vg_datum_month', 'vg_datum_day_of_week', 'vg_datum_season'
features_v7 = [
    'venue_10%', 'venue_15%', 'venue_20%', 'venue_25%', 'venue_30%', 
    'venue_35%', 'venue_40%', 'venue_45%', 'venue_5%', 'venue_50%', 
    'venue_55%', 'venue_60%', 'venue_65%', 'venue_70%', 'venue_75%', 
    'venue_80%', 'venue_85%', 'venue_90%', 'venue_95%', 'venue_count', 
    'venue_max', 'venue_mean', 'venue_min', 'venue_std', 
    
    'band_10%', 'band_15%', 'band_20%', 'band_25%', 'band_30%', 
    'band_35%', 'band_40%', 'band_45%', 'band_5%', 'band_50%', 
    'band_55%', 'band_60%', 'band_65%', 'band_70%', 'band_75%', 
    'band_80%', 'band_85%', 'band_90%', 'band_95%', 'band_count', 
    'band_max', 'band_mean', 'band_min', 'band_std', 

    'promoter_10%', 'promoter_15%', 'promoter_20%', 'promoter_25%', 'promoter_30%', 
    'promoter_35%', 'promoter_40%', 'promoter_45%', 'promoter_5%', 'promoter_50%', 
    'promoter_55%', 'promoter_60%', 'promoter_65%', 'promoter_70%', 'promoter_75%', 
    'promoter_80%', 'promoter_85%', 'promoter_90%', 'promoter_95%', 'promoter_count', 
    'promoter_max', 'promoter_mean', 'promoter_min', 'promoter_std',
    
    'vg_state', 'vg_datum_year', 
    'vg_datum_month', 'vg_datum_day_of_week', 'vg_datum_season'
    ]

    # 78 Feautes used - Leave-one-out-target-encoded 'band', 'promoter', 'venue' &
#  'vg_state', 'vg_datum_year', 
#  'vg_datum_month', 'vg_datum_day_of_week', 'vg_datum_season', 'tarif_bez'
features_v8 = [
    'venue_10%', 'venue_15%', 'venue_20%', 'venue_25%', 'venue_30%', 
    'venue_35%', 'venue_40%', 'venue_45%', 'venue_5%', 'venue_50%', 
    'venue_55%', 'venue_60%', 'venue_65%', 'venue_70%', 'venue_75%', 
    'venue_80%', 'venue_85%', 'venue_90%', 'venue_95%', 'venue_count', 
    'venue_max', 'venue_mean', 'venue_min', 'venue_std'
    , 
    
    'band_10%', 'band_15%', 'band_20%', 'band_25%', 'band_30%', 
    'band_35%', 'band_40%', 'band_45%', 'band_5%', 'band_50%', 
    'band_55%', 'band_60%', 'band_65%', 'band_70%', 'band_75%', 
    'band_80%', 'band_85%', 'band_90%', 'band_95%', 'band_count', 
    'band_max', 'band_mean', 'band_min', 'band_std'
    , 

    'promoter_10%', 'promoter_15%', 'promoter_20%', 'promoter_25%', 'promoter_30%', 
    'promoter_35%', 'promoter_40%', 'promoter_45%', 'promoter_5%', 'promoter_50%', 
    'promoter_55%', 'promoter_60%', 'promoter_65%', 'promoter_70%', 'promoter_75%', 
    'promoter_80%', 'promoter_85%', 'promoter_90%', 'promoter_95%', 'promoter_count', 
    'promoter_max', 'promoter_mean', 'promoter_min', 'promoter_std'
    ,
    
    'vg_state', 'vg_datum_year', 
    'vg_datum_month', 'vg_datum_day_of_week', 'vg_datum_season', 'tarif_bez'
    ]

#9 features
#mean
features_v9 = [ 
    'venue_mean', 
    
    'band_mean', 

    'promoter_mean',
    
    'vg_state', 'vg_datum_year', 
    'vg_datum_month', 'vg_datum_day_of_week', 'vg_datum_season', 'tarif_bez'
    ]

#21 features
#mean, std, min, max, count
features_v10 = [ 
    'venue_mean', 'venue_min', 'venue_std', 'venue_max', 'venue_count', 
    
    'band_mean','band_min', 'band_std', 'band_max', 'band_count',

    'promoter_mean', 'promoter_min', 'promoter_std', 'promoter_max', 'promoter_count',
    
    'vg_state', 'vg_datum_year', 
    'vg_datum_month', 'vg_datum_day_of_week', 'vg_datum_season', 'tarif_bez'
    ]