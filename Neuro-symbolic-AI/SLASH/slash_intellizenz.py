
import datetime
import train

date_string = datetime.datetime.today().strftime('%d-%m-%Y')

experiments = {'intellizenz_bt_20_5_bs_512_s_20_k_20': 
                   {'structure': 'binary-trees', 'num_repetitions': 20, 'depth': 5,
                    'num_sums':20, 'k':20,
                    'lr': 0.01, 'bs':4096, 'epochs':200, 'use_em':False,
                    'start_date':date_string, 'resume':True, 'p_num':6, 'credentials':'DO',
              'explanation': """Intellizenz experiment using WMC for a simple classification."""}}

experiments = {'slash-with-nn': 
                   {'lr': 0.001, 'bs':64, 'epochs':200, 'use_em':False,
                    'start_date':date_string, 'resume':True, 'p_num':6, 'credentials':'SNN'}}

experiments = {'simple-nn': 
                   {'lr': 0.001, 'bs':64, 'epochs':200, 'use_em':False,
                    'start_date':date_string, 'resume':True, 'p_num':6, 'credentials':'NN'}}

experiments = {'tabnet': 
                   {'lr': 0.001, 'bs':64, 'epochs':200, 'use_em':False,
                    'start_date':date_string, 'resume':True, 'p_num':6, 'credentials':'TN'}}

experiments = {'slash-with-tabnet': 
                   {'lr': 0.001, 'bs':64, 'epochs':200, 'use_em':False,
                    'start_date':date_string, 'resume':True, 'p_num':6, 'credentials':'STN'}}




# train the network
for exp_name in experiments:
    print(exp_name)
    train.slash_intellizenz(exp_name, experiments[exp_name])