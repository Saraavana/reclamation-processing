data_path_2016_2020_v3 = 'C:/Users/sgopalakrish/Downloads/intellizenz-model-training/data/export_features_2016_2020_v3.parquet.gzip'
train_data_path = 'C:/Users/sgopalakrish/Downloads/intellizenz-model-training/data/export_training_features_2016_2020_v2.parquet.gzip'
test_data_path = 'C:/Users/sgopalakrish/Downloads/intellizenz-model-training/data/export_testing_features_2016_2020_v2.parquet.gzip'

data_path_2016_2020_v4 = 'C:/Users/sgopalakrish/Downloads/intellizenz-model-training/data/export_features_2016_2020_v4.parquet.gzip'
data_path_2016_2020_v5 = 'C:/Users/sgopalakrish/Downloads/intellizenz-model-training/data/export_features_2016_2020_v5.parquet.gzip'


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

            'band_kurorchester bad wil', 'band_musikverein harmonie', 'band_kasalla',
            'band_cat ballou', 'band_roncalli  royal orch', 'band_jugendblasorchester',
            'band_kurorchester bad pyr', 'band_hoehner', 'band_paveier',
            'band_domstuermer', 'band_kluengelkoepp', 'band_alleinunterhalter',
            'band_the gregorian voices', 'band_brings', 'band_musica hungarica',
            'band_concerto', 'band_bad salzuflen orches', 'band_musikverein stadtkap',
            'band_salonorchester hunga', 'band_miljoe', 'band_raeuber',
            'band_kabarett leipziger f', 'band_marita koellner', 'band_salon-orchester hung',
            'band_blaeck foeoess', 'band_schuelerinnen und sc', 'band_romain vicente',
            'band_staatliche kurkapell', 'band_musikzug der freiwil', 'band_funky marys',

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

features_v2 = [

]

# 142 Feautes used - 'VG_RAUM_KEYWORDS', 'VG_DATUM_VON', 'vg_state', 'BAND', 'PROMOTER', TARIF_BEZ
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


'band_kurorchester bad wil', 'band_musikverein harmonie', 'band_kasalla',
 'band_cat ballou', 'band_roncalli  royal orch', 'band_jugendblasorchester',
 'band_kurorchester bad pyr', 'band_hoehner', 'band_paveier',
 'band_domstuermer', 'band_kluengelkoepp', 'band_alleinunterhalter',
 'band_the gregorian voices', 'band_brings', 'band_musica hungarica',
 'band_concerto', 'band_bad salzuflen orches', 'band_musikverein stadtkap',
 'band_salonorchester hunga', 'band_miljoe', 'band_raeuber',
 'band_kabarett leipziger f', 'band_marita koellner', 'band_salon-orchester hung',
 'band_blaeck foeoess', 'band_schuelerinnen und sc', 'band_romain vicente',
 'band_staatliche kurkapell', 'band_musikzug der freiwil', 'band_funky marys',


'state_bavaria','state_rhineland-palatinate',
'state_baden-wuerttemberg',	'state_north rhine-westphalia',	
'state_thuringia','state_hesse',	
'state_brandenburg', 'state_schleswig-holstein',	
'state_berlin',	'state_mecklenburg-western pomerania',	
'state_lower saxony', 'state_hamburg',	
'state_saarland', 'state_saxony-anhalt',	
'state_saxony',	'state_bremen',

'vg_datum_year','vg_datum_month','vg_datum_day_of_week','vg_datum_season',

'promoter_live nation gmbh',
'promoter_fkp scorpio konzertproduktionen gmbh', 
'promoter_trinity music gmbh', 
'promoter_karsten jahnke konzertdirektion gmbh', 
'promoter_prime entertainment gmbh', 
'promoter_semmel concerts entertainment gmbh', 
'promoter_chorverband nrw e.v', 
'promoter_schwaebischer chorverband e.v', 
'promoter_backstage concerts gmbh', 
'promoter_kulturzentrum schlachthof wiesbaden e.v', 
'promoter_irish pubs gaststaetten gmbh', 
'promoter_fraenkischer saengerbund e.v', 
'promoter_frankfurter kulturzentrum e.v', 
'promoter_circus roncalli gmbh', 
'promoter_paul daly und paul fleming gbr', 
'promoter_foerderkreis jazz und malerei muenchen e.v', 
'promoter_hessischer saengerbund e.v', 
'promoter_gastro event gmbh', 
'promoter_berninger musik & gastronomie gmbh', 
'promoter_feierwerk e.v',
'promoter_bayerisches staatsbad bad steben gmbh', 
'promoter_europa-park gmbh & co mack kg', 
'promoter_badischer chorverband 1862 e.v', 
'promoter_bayerisches wirtshaus berlin gmbh', 
'promoter_tollwood gmbh', 
'promoter_graeflicher park gmbh & co. kg',
'promoter_konzertbuero schoneberg gmbh',  	
'promoter_staatsbad salzuflen gmbh', 
'promoter_kurverwaltung bad mergentheim gmbh', 
'promoter_gisbert hiller',

'veranst_segment','vg_inkasso'
]

# 238 Feautes used - 'VG_RAUM_KEYWORDS', 'VG_DATUM_VON', 'vg_state', 'BAND', 'PROMOTER', TARIF_BEZ, vg_state_percentiles,
# promoter_percentiles, band_percentiles, tarif_percentiles
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


'band_kurorchester bad wil', 'band_musikverein harmonie', 'band_kasalla',
 'band_cat ballou', 'band_roncalli  royal orch', 'band_jugendblasorchester',
 'band_kurorchester bad pyr', 'band_hoehner', 'band_paveier',
 'band_domstuermer', 'band_kluengelkoepp', 'band_alleinunterhalter',
 'band_the gregorian voices', 'band_brings', 'band_musica hungarica',
 'band_concerto', 'band_bad salzuflen orches', 'band_musikverein stadtkap',
 'band_salonorchester hunga', 'band_miljoe', 'band_raeuber',
 'band_kabarett leipziger f', 'band_marita koellner', 'band_salon-orchester hung',
 'band_blaeck foeoess', 'band_schuelerinnen und sc', 'band_romain vicente',
 'band_staatliche kurkapell', 'band_musikzug der freiwil', 'band_funky marys',


'state_bavaria','state_rhineland-palatinate',
'state_baden-wuerttemberg',	'state_north rhine-westphalia',	
'state_thuringia','state_hesse',	
'state_brandenburg', 'state_schleswig-holstein',	
'state_berlin',	'state_mecklenburg-western pomerania',	
'state_lower saxony', 'state_hamburg',	
'state_saarland', 'state_saxony-anhalt',	
'state_saxony',	'state_bremen',

'vg_datum_year','vg_datum_month','vg_datum_day_of_week','vg_datum_season',

'promoter_live nation gmbh',
'promoter_fkp scorpio konzertproduktionen gmbh', 
'promoter_trinity music gmbh', 
'promoter_karsten jahnke konzertdirektion gmbh', 
'promoter_prime entertainment gmbh', 
'promoter_semmel concerts entertainment gmbh', 
'promoter_chorverband nrw e.v', 
'promoter_schwaebischer chorverband e.v', 
'promoter_backstage concerts gmbh', 
'promoter_kulturzentrum schlachthof wiesbaden e.v', 
'promoter_irish pubs gaststaetten gmbh', 
'promoter_fraenkischer saengerbund e.v', 
'promoter_frankfurter kulturzentrum e.v', 
'promoter_circus roncalli gmbh', 
'promoter_paul daly und paul fleming gbr', 
'promoter_foerderkreis jazz und malerei muenchen e.v', 
'promoter_hessischer saengerbund e.v', 
'promoter_gastro event gmbh', 
'promoter_berninger musik & gastronomie gmbh', 
'promoter_feierwerk e.v',
'promoter_bayerisches staatsbad bad steben gmbh', 
'promoter_europa-park gmbh & co mack kg', 
'promoter_badischer chorverband 1862 e.v', 
'promoter_bayerisches wirtshaus berlin gmbh', 
'promoter_tollwood gmbh', 
'promoter_graeflicher park gmbh & co. kg',
'promoter_konzertbuero schoneberg gmbh',  	
'promoter_staatsbad salzuflen gmbh', 
'promoter_kurverwaltung bad mergentheim gmbh', 
'promoter_gisbert hiller',

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


'band_kurorchester bad wil', 'band_musikverein harmonie', 'band_kasalla',
 'band_cat ballou', 'band_roncalli  royal orch', 'band_jugendblasorchester',
 'band_kurorchester bad pyr', 'band_hoehner', 'band_paveier',
 'band_domstuermer', 'band_kluengelkoepp', 'band_alleinunterhalter',
 'band_the gregorian voices', 'band_brings', 'band_musica hungarica',
 'band_concerto', 'band_bad salzuflen orches', 'band_musikverein stadtkap',
 'band_salonorchester hunga', 'band_miljoe', 'band_raeuber',
 'band_kabarett leipziger f', 'band_marita koellner', 'band_salon-orchester hung',
 'band_blaeck foeoess', 'band_schuelerinnen und sc', 'band_romain vicente',
 'band_staatliche kurkapell', 'band_musikzug der freiwil', 'band_funky marys',


'state_bavaria','state_rhineland-palatinate',
'state_baden-wuerttemberg',	'state_north rhine-westphalia',	
'state_thuringia','state_hesse',	
'state_brandenburg', 'state_schleswig-holstein',	
'state_berlin',	'state_mecklenburg-western pomerania',	
'state_lower saxony', 'state_hamburg',	
'state_saarland', 'state_saxony-anhalt',	
'state_saxony',	'state_bremen',

'vg_datum_year','vg_datum_month','vg_datum_day_of_week','vg_datum_season',

'promoter_live nation gmbh',
'promoter_fkp scorpio konzertproduktionen gmbh', 
'promoter_trinity music gmbh', 
'promoter_karsten jahnke konzertdirektion gmbh', 
'promoter_prime entertainment gmbh', 
'promoter_semmel concerts entertainment gmbh', 
'promoter_chorverband nrw e.v', 
'promoter_schwaebischer chorverband e.v', 
'promoter_backstage concerts gmbh', 
'promoter_kulturzentrum schlachthof wiesbaden e.v', 
'promoter_irish pubs gaststaetten gmbh', 
'promoter_fraenkischer saengerbund e.v', 
'promoter_frankfurter kulturzentrum e.v', 
'promoter_circus roncalli gmbh', 
'promoter_paul daly und paul fleming gbr', 
'promoter_foerderkreis jazz und malerei muenchen e.v', 
'promoter_hessischer saengerbund e.v', 
'promoter_gastro event gmbh', 
'promoter_berninger musik & gastronomie gmbh', 
'promoter_feierwerk e.v',
'promoter_bayerisches staatsbad bad steben gmbh', 
'promoter_europa-park gmbh & co mack kg', 
'promoter_badischer chorverband 1862 e.v', 
'promoter_bayerisches wirtshaus berlin gmbh', 
'promoter_tollwood gmbh', 
'promoter_graeflicher park gmbh & co. kg',
'promoter_konzertbuero schoneberg gmbh',  	
'promoter_staatsbad salzuflen gmbh', 
'promoter_kurverwaltung bad mergentheim gmbh', 
'promoter_gisbert hiller',

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