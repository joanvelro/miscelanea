#!/usr/bin/env/python
"""
    This script retrieves the road traffic data grouped by section with 15-min time resolution from the database,
    Prepare the input data set for the recursive prediction module, make the prediction by section at 15, 60 and 120 min
    and finally save resutls in database

    External dependencies:
        dl-ia-cla_create_config_file.py
        dl_ia_cla_settings.py
        query_read_agg_road_traffic_data.sql
        config.ini
        clustering_sections_all.csv
        predictive_model.sav

      @ Jose Angel Velasco (joseangel.velasco@yahoo.es)
    (C) Universidad Carlos III de Madrid
"""
import pyodbc
import pandas as pd
import numpy as np
import datetime
pd.options.mode.chained_assignment = None  # default='warn'
import joblib
import time
from configparser import ConfigParser
import pprint
import logging
import sys

### Load required functions
from dl_ia_cla_utils import dl_ia_cla_utils_initialize_engine
from dl_ia_cla_utils import dl_ia_cla_utils_create_timestamp
from dl_ia_cla_utils import dl_ia_cla_utils_create_datetime

### initialize error
error = 0

### Start time
start = time.time()

""" Define road (or get as input argument)
"""
#road = 'C7'
road = str(sys.argv[1]) # uncoment this to read the input argument


""" Set logging file
"""
logger = logging.getLogger('dl-ia-cla-predictive')
logger.setLevel(logging.DEBUG)
fh = logging.FileHandler('dl-ia-cla-predictive.log')
fh.setLevel(logging.DEBUG)
logger.addHandler(fh)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fh.setFormatter(formatter)
logger.addHandler(fh)
logging.getLogger().addHandler(logging.StreamHandler()) # to display in console message
#logger.debug('mensaje debug')
#logger.info('mensaje info')
#logger.warning('mensaje warning')
#logger.error('mensaje error')
#logger.critical('mensaje critical')




""" Get config file info
"""
### Choose database to use
db_pre = True # To use the postgreSQL database
db_ser = False # To use the SQL server database

### Read config.ini file
config_object = ConfigParser()
config_object.read("config.ini")
#print(config_object._sections)

### get paths from config file
data_path =  config_object._sections["paths"]['data_path']
models_path = config_object._sections["paths"]['models_path']
predictive_table = config_object._sections["database_tables"]['predictive_table']

if db_pre:
    db_settings =  config_object._sections["DB"]
    engine = dl_ia_cla_utils_initialize_engine(db_settings)
if db_ser:
    db_settings =  config_object._sections["DB_SER"]

### databasse credentials
driver = db_settings['driver']
server = db_settings['server']
database = db_settings['database']
user = db_settings['user']
password = db_settings['pass']
schema = db_settings['schema']
port = db_settings['port']



#pp.pprint(db_settings_PRE)
#for a in db_settings_PRO.keys():
#   print(a,':', db_settings_PRO[a])


logger.info('{} Starting prediction'.format('-'*20))


""" query to get last 6 hours of road traffic data from SQL server
"""
if db_ser:
    logger.info('{} Query to retrieve last six hourns of data'.format('-'*20))
    logger.info('{} Loading input data from server:[{}] database:[{}] schema:[{}] '.format('-' * 20, server, database, schema))
    #print('{} Loading input data from server:[{}] database:[{}] schema:[{}] '.format('-' * 20, server, database, schema))
    sql_conn_str = 'DRIVER={};SERVER={},{};DATABASE={};UID={};PWD={}'.format(driver, server, port, database, user, password)
    #sql_conn_str = 'DRIVER={SQL Server};SERVER=' + server + ';DATABASE=' + database + ';UID=' + user + ';PWD=' + password + 'Trusted_Connection=yes'

    query = """select logic_code, section_alias, fecha, intensidad from [F_Prediction_Section] (7, '[]')"""

    try:
        sql_conn = pyodbc.connect(sql_conn_str)
        df_input = pd.read_sql(query, sql_conn)
        sql_conn.close()

        if df_input.empty:
            logger.error('{} (!) Error: No data retrieved from database'.format('-' * 20))
            error = 1

    except Exception as exception_msg:
        logger.error('{} (!) Error in query database sqlserver: '.format('-'*20) + str(exception_msg))
        error=1





""" query to get last 6 hours of road traffic data from postgreSQL
"""
if db_pre:
    logger.info('{} Query to retrieve last six hours of data'.format('-'*20))
    logger.info('{} Loading input data from server:[{}] database:[{}] schema:[{}] '.format('-'*20, server, database, schema))

    f = open('query_read_agg_road_traffic_data.sql', "r")
    query = f.read()

    try:
        df_input = pd.read_sql_query(query, engine)

        if df_input.empty:
            logger.error('{} (!) Error: No data retrieved from database'.format('-' * 20))
            error = 1

        else:
            df_input['timestamp'] = pd.to_datetime(df_input['fecha'], format='%Y-%m-%d %H:%M:%S')
            df_input['logic_code'] = df_input['sec_logic_code']
            df_input.rename(columns={'intensity': 'intensidad'}, inplace=True)

    except Exception as exception_msg:
        logger.error('{} (!) Error in query database postgreSQL: '.format('-'*20) + str(exception_msg))
        error=1







""" select only the sections of the road 
"""
if error==0:
    logger.info('{} Merge sections dict'.format('-'*20))
    #df_sections = pd.read_csv(data_path + '\\sections_dict_all.csv')
    df_sections = pd.read_csv('references/sections_dict_all.csv')
    df_input['logic_code'] = df_input['logic_code'].astype(int)
    df_input = pd.merge(df_sections[['ID_SECTION', 'logic_code']], df_input, on=['logic_code'], how='inner')

""" Check if the dataframe is empty
"""
if error==0:
    if df_input.empty:
        logger.error('{} (!) Error: No sections data merged'.format('-'*20))
        error=1

""" Prepare the input data for prediction
"""
if error==0:
    logger.info('{} Prepare the input data for prediction'.format('-'*20))
    df_input['VOLUME'] = df_input['intensidad']/4
    df_input['VOLUME'] = df_input['VOLUME'].astype(int)
    df_input['TIMESTAMP'] = pd.to_datetime(df_input['fecha'], format='%Y-%m-%d %H:%M:%S')
    logger.info('{} most updated data: {} (UTC TIME)'.format('-'*20, df_input['TIMESTAMP'].max()))
    df_input['MONTH'] = df_input['TIMESTAMP'].dt.month
    df_input['HOUR'] = df_input['TIMESTAMP'].dt.hour
    df_input['MINUTE'] = df_input['TIMESTAMP'].dt.minute
    df_input['WEEKDAY'] = df_input['TIMESTAMP'].dt.weekday
    df_input.sort_values(by=['ID_SECTION', 'HOUR', 'MINUTE'], inplace=True, ascending=True)
    df_input.reset_index(inplace=True, drop=True)



""" create lag variables of volume, iter over all sections
"""
if error==0:
    logger.info('{} create lag variables'.format('-'*20))
    df_input_final = pd.DataFrame([])
    for section in df_input['ID_SECTION'].unique():
        df_aux = df_input[df_input['ID_SECTION']==section]

        for i in range(1, 25):
            if i == 1:
                df_aux['VOLUME_{}'.format(str(i))] = df_aux['VOLUME'].shift(1)
            elif i > 1:
                df_aux['VOLUME_{}'.format(str(i))] = df_aux['VOLUME_{}'.format(str(i - 1))].shift(1)

        df_aux.fillna(0, inplace=True)
        df_aux = df_aux[df_aux['TIMESTAMP'] == df_aux['TIMESTAMP'].max()].reset_index()
        df_input_final = pd.concat([df_input_final, df_aux], sort=True)



""" Check if the dataframe is empty
"""
if error==0:
    if df_input_final.empty:
        logger.error('{} (!) Data frame empty after creating lag variables'.format('-'*20))
        error=1


""" change to integet data type 
"""
if error==0:
    for i in np.arange(1, 25):
        df_input_final['VOLUME_{}'.format(str(i))] = df_input_final['VOLUME_{}'.format(str(i))].astype(int)


""" Add the columns cluster
"""
if error==0:
    logger.info('{} Merge clustering data'.format('-'*20))
    #df_cluster = pd.read_csv(data_path + '\\clustering_sections_all.csv')
    df_cluster = pd.read_csv('references/clustering_sections_all.csv') # for deployment

    if df_cluster.empty:
        logger.error('(!) Error: No clustering data')

    df_input_final = pd.merge(df_input_final, df_cluster, on=['ID_SECTION'], how='inner')
    if df_input_final.empty:
        logger.error('(!) Error: No clustering data merged')

    df_input_final['ID_CLUSTER'] = df_input_final['ID_CLUSTER'].astype(int)

""" Check if the dataframe is empty
"""
if error==0:
    if df_input_final.empty:
        logger.error('{} (!) Data frame empty'.format('-'*20))
        error=1


""" Define final dataset
"""
if error==0:
    logger.info('{} Define final dataset'.format('-'*20))
    df_input_final = df_input_final[['ID_CLUSTER', 'ID_SECTION', 'TIMESTAMP', 'MONTH', 'WEEKDAY', 'HOUR', 'MINUTE', 'VOLUME'] + ['VOLUME_{}'.format(i) for i in np.arange(1,25)]]
    df_input_final.reset_index(inplace=True, drop=True)





""" Create features list
"""
if error==0:
    logger.info('{} Create features list and splitting data set'.format('-' * 20))
    lagged_features = []
    for i in np.arange(1, 25):
        lagged_features.append('VOLUME_{}'.format(i))
    features = ['ID_CLUSTER', 'MONTH', 'WEEKDAY', 'HOUR', 'MINUTE'] + lagged_features
    target = ['VOLUME']


""" Load predictive model
"""
if error==0:
    logger.info('{} Load predictive model'.format('-'*20))
    #model = joblib.load(models_path + '\\predictive_model.sav')
    model = joblib.load('predictive_model.sav')


""" Get predictions in recursive way foe earch segment
"""
if error==0:
    logger.info('{} Making recursive predictions'.format('-'*20))
    # initialise a results dataframe
    df_results = pd.DataFrame(data=[])
    for section in df_input_final['ID_SECTION'].unique():
        logger.info('{} Section: {}'.format('-' * 20, section))

        # slice the input features of the current segment
        test_data = df_input_final.loc[df_input_final['ID_SECTION'] == section]  # clustering function
        X = test_data.loc[:, features]

        # make first prediction horizon 15 min
        y_pred_15 = model.predict(X)

        # initialise a vector of prediction in such a way that prediction horizon 60 is 3 position, and horizon 120 is 7
        horizon = [15, 30, 45, 60, 75, 90, 105, 120]
        y_preds = [y_pred_15[0], 0, 0, 0, 0, 0, 0, 0, 0]

        # continue for horizon 30 min and the following up to 120 min i.e.2h (time step of 15 min)
        time_step = 0
        forecast_window = int(120 / 15)  # 8 steps
        while time_step < forecast_window:
            # print('{} prediction time step {} horizon:{} min'.format('-' * 20, time_step, horizon[time_step]))
            if X['MINUTE'].values[0] < 45:
                X['MINUTE'] = X['MINUTE'] + 15
            if X['MINUTE'].values[0] == 45:
                X['MINUTE'] = int(0)
                X['HOUR'] = X['HOUR'] + 1

            # move to the left the lagged columns
            Xm = X.iloc[:, 5:].shift(1, axis=1)
            X = pd.concat([X[['ID_CLUSTER', 'MONTH', 'WEEKDAY', 'HOUR', 'MINUTE']], Xm], axis=1)

            # update lagged volumes with the prediction
            for i in range(0, time_step + 1):
                # print('VOLUME_{} = {}'.format(i + 1, i))
                X['VOLUME_{}'.format(i + 1)] = int(y_preds[i])

            # make the new prediction
            time_step = time_step + 1
            y_preds[time_step] = model.predict(X)[0]

        # create an auxiliar dataframe for results
        results0 = test_data.loc[:, ['ID_CLUSTER', 'MONTH', 'WEEKDAY', 'HOUR', 'MINUTE']]
        results0['Prediccion_int_15'] = int(y_preds[1])
        results0['Prediccion_int_60'] = int(y_preds[3])
        results0['Prediccion_int_120'] = int(y_preds[7])
        results0['ID_SECTION'] = section

        # add the auxiliar dataframe of results to the final resulta dataframe
        df_results = pd.concat([df_results, results0])

""" Create accidents variables
"""
if error==0:
    logger.info('{} Create accidents variables'.format('-'*20))
    df_results['Ind_accidente_15'] = 0
    df_results['Ind_accidente_60'] = 0
    df_results['Ind_accidente_120'] = 0

""" Create timestamp column
"""
if error==0:
    logger.info(' {} Create timestamp column'.format('-'*20))
    df_results['YEAR'] = df_input['TIMESTAMP'].dt.year
    df_results['MONTH'] = df_input['TIMESTAMP'].dt.month
    df_results['DAY'] = df_input['TIMESTAMP'].dt.day
    df_results['HOUR'] = df_input['TIMESTAMP'].dt.hour
    df_results['MINUTE'] = df_input['TIMESTAMP'].dt.minute



    df_results['Fecha'] = df_results.apply(dl_ia_cla_utils_create_timestamp, axis=1)



""" Add logic code
"""
if error==0:
    logger.info('{} Add logic code'.format('-'*20))
    df_results = pd.merge(df_results, df_sections, on=['ID_SECTION'], how='inner')


""" Prepare final data frame
"""
if error==0:
    logger.info('{} Prepare final data frame'.format('-'*20))
    df_results.drop(columns=['YEAR', 'MONTH','DAY','WEEKDAY', 'HOUR', 'MINUTE', 'ID_SECTION'], inplace=True)
    df_results.rename(columns={'section_alias': 'Section_Alias'}, inplace=True)
    final_cols = ['logic_code','Fecha', 'Section_Alias', 'Prediccion_int_15', 'Prediccion_int_60', 'Prediccion_int_120', 'Ind_accidente_15', 'Ind_accidente_60', 'Ind_accidente_120']
    df_results = df_results[final_cols]
    df_results.drop_duplicates(inplace=True)





""" Check primary key : logic_code, day_type, hour, minute
"""
if error==0:
   df_results.rename(columns={'Section_Alias':'section_alias',
                                   'Fecha':'fecha',
                                   'Prediccion_int_15':'prediccion_int_15',
                                   'Prediccion_int_60':'prediccion_int_60',
                                   'Prediccion_int_120':'prediccion_int_120',
                                   'Ind_accidente_15':'ind_accidente_15',
                                   'Ind_accidente_60':'ind_accidente_60',
                                   'Ind_accidente_120':'ind_accidente_120'}, inplace=True)
   df_results = df_results[['logic_code',
                             'section_alias',
                             'fecha',
                             'prediccion_int_15',
                             'prediccion_int_60',
                             'prediccion_int_120',
                             'ind_accidente_15',
                             'ind_accidente_60',
                             'ind_accidente_120']]

""" Check primary key : logic_code, day_type, hour, minute
"""
if error==0:
    index = df_results[['logic_code', 'fecha', ]].duplicated()
    if index.sum()>0:
        logger.warning('{} Primary key duplicated'.format('-' * 20))
        df_results = df_results[np.invert(index)]


""" Store prediction results in SQLServer 
"""
if error==0:
    logger.info('{} Saving prediction results in database'.format('-'*20))
    if db_ser:
        logger.info('Server:[{}] database:[{}] schema:[{}] '.format(server, database, schema))
        #sql_conn = pyodbc.connect('DRIVER={SQL Server};SERVER=' + server + ';DATABASE=' + database + ';UID=' + user + ';PWD=' + password)
        sql_conn_str = 'DRIVER={};SERVER={},{};DATABASE={};UID={};PWD={}'.format(driver, server, port, database, user, password)
        # sql_conn_str = 'DRIVER={SQL Server};SERVER=' + server + ';DATABASE=' + database + ';UID=' + user + ';PWD=' + password + 'Trusted_Connection=yes'

        try:
            sql_conn = pyodbc.connect(sql_conn_str)
            cursor = sql_conn.cursor()
            for index, row in df_results.iterrows():
                # print(row)
                cursor.execute(
                    "INSERT INTO H_PREDICTIVE_SECTION([logic_code],[Fecha],[Section_Alias],[Prediccion_int_15],[Prediccion_int_60],[Prediccion_int_120],[Ind_accidente_15],[Ind_accidente_60],[Ind_accidente_120]) values(?,?,?,?,?,?,?,?,?)",
                    row['logic_code'],
                    row['Fecha'],
                    row['Section_Alias'],
                    row['Prediccion_int_15'],
                    row['Prediccion_int_60'],
                    row['Prediccion_int_120'],
                    row['Ind_accidente_15'],
                    row['Ind_accidente_60'],
                    row['Ind_accidente_120'])

            sql_conn.commit()
            cursor.close()
            sql_conn.close()
        except Exception as exception_msg:
            logger.error('{} (!) Error in loading prediction in database sqlserver: '.format('-' * 20) + str(exception_msg))
            error = 1


    """ Store prediction results in postgreSQL
    """
    if db_pre:
        logger.info('{} Server:[{}] database:[{}] schema:[{}] '.format('-'*20, server, database, schema))



        try:
            df_results.to_sql(name=predictive_table,
                              con=engine,
                              schema=schema,
                              if_exists='append',
                              chunksize=10000,
                              index=False)
        except Exception as exception_msg:
            logger.error('{} (!) Error in loading prediction in database postgreSQL: '.format('-' * 20) + str(exception_msg))
            error = 1





if error==0:
    logger.info('{} Prediction successfully'.format('-'*20, server, database, schema))
    end = time.time()
    logger.info('{} total time required: {} min'.format('-' * 20, (end - start) / 60))
else:
    logger.error('{} (!) Something went wrong'.format('-'*20, server, database, schema))

