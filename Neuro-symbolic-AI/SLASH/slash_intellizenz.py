
import datetime
import train

date_string = datetime.datetime.today().strftime('%d-%m-%Y')

# experiments = {'slash-with-nn_intellizenz_bt_20_5_bs_512_s_20_k_20': 
#                    {'lr': 0.01, 'bs':4096, 'epochs':200, 'use_em':False,
#                     'start_date':date_string, 'resume':True, 'p_num':6, 'credentials':'SNN',
#               'explanation': """Intellizenz experiment using WMC for a simple classification."""}}

experiments = {'slash-with-nn': 
                   {'lr': 0.00001, 'bs':64, 'epochs':20, 'use_em':False,
                    'start_date':date_string, 'resume':False, 'p_num':6, 'credentials':'SNN'}}

# experiments = {'simple-nn': 
#                    {'lr': 0.00001, 'bs':128, 'epochs':500, 'use_em':False,
#                     'start_date':date_string, 'resume':True, 'p_num':6, 'credentials':'NN'}}

# experiments = {'tabnet': 
#                    {'lr': 0.001, 'bs':64, 'epochs':200, 'use_em':False,
#                     'start_date':date_string, 'resume':True, 'p_num':6, 'credentials':'TN'}}

# experiments = {'slash-with-tabnet': 
#                    {'lr': 0.001, 'bs':64, 'epochs':200, 'use_em':False,
#                     'start_date':date_string, 'resume':True, 'p_num':6, 'credentials':'STN'}}

# experiments = {'predictive-whittle-network-einsum': 
#                    {
#                     'lr': 0.00001, 'bs':64, 'epochs':10, 'use_em':False,
#                     'start_date':date_string, 'resume':False, 'p_num':6, 'credentials':'PWN_ES'}}




# train the network
for exp_name in experiments:
    print(exp_name)
    train.slash_intellizenz(exp_name, experiments[exp_name])