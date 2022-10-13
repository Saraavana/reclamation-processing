
import datetime
import train

date_string = datetime.datetime.today().strftime('%d-%m-%Y')

experiments = {'intellizenz_bt_20_5_bs_512_s_20_k_20': 
                   {'structure': 'binary-trees', 'num_repetitions': 20, 'depth': 5,
                    'num_sums':20, 'k':20,
                    'lr': 0.01, 'bs':4096, 'epochs':10, 'use_em':False,
                    'start_date':date_string, 'resume':False, 'p_num':6, 'credentials':'DO',
              'explanation': """Intellizenz experiment using WMC for a simple classification."""}}






# train the network
for exp_name in experiments:
    print(exp_name)
    train.slash_intellizenz(exp_name, experiments[exp_name])