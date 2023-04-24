import datetime
import slash_tabnet_train

date_string = datetime.datetime.today().strftime('%d-%m-%Y')


# experiments = {'tabnet': 
#                    {'lr': 0.001, 'bs':64, 'epochs':200, 'use_em':False,
#                     'start_date':date_string, 'resume':True, 'p_num':6, 'credentials':'TN'}}

# experiments = {'slash-with-tabnet': 
#                    {'lr': 0.001, 'bs':64, 'epochs':200, 'use_em':False,
#                     'start_date':date_string, 'resume':False, 'p_num':1, 'credentials':'STN'}}

experiments = {'slash-with-tabnet': 
                   {'lr': 0.02, 'bs':16384, 'epochs':5, 'use_em':False,
                    'start_date':date_string, 'resume':False, 'p_num':6, 'credentials':'STN'}}

# experiments = {'slash-with-tabnet': 
#                    {'lr': 0.00002, 'bs':16384, 'epochs':30, 'use_em':False,
#                     'start_date':date_string, 'resume':False, 'p_num':6, 'credentials':'STN'}}

# experiments = {'slash-with-tabnet': 
#                    {'lr': 0.02, 'bs':500, 'epochs':200, 'use_em':False,
#                     'start_date':date_string, 'resume':False, 'p_num':8, 'credentials':'STN'}}

# experiments = {'slash-with-tabnet': 
#                    {'lr': 0.02, 'bs':7000, 'epochs':1, 'use_em':False,
#                     'start_date':date_string, 'resume':False, 'p_num':1, 'credentials':'STN'}}

                    


# Train the network
for exp_name in experiments:
    print(exp_name)
    slash_tabnet_train.slash_tabnet(exp_name=exp_name, exp_dict=experiments[exp_name])