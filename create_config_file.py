#!/usr/bin/env/python
"""
This script creates the config file of the project

    @ Jose Angel Velasco (joseangel.velasco@yahoo.es)
    (C) Universidad Carlos III de Madrid
"""

from configparser import ConfigParser

#Get the configparser object
config_object = ConfigParser()


# Paths
config_object["paths"] = {
    "data_path":'D:\\data\\clarence',
    'models_path':'D:\\models\\clarence',
    'results_path':'D:\\results\\clarence'
}

# Paths
config_object["database_tables"] = {
    "descriptive_table":'h_descriptive_section',
    'predictive_table':'h_predictive_section'
}

# DB postgresql
config_object["DB"] = {
    "driver": "postgresql",
    "user": "conf_horus",
    "pass": "CONF_HORUS",
    "server": "10.72.1.16",
    "database": "horus",
     "port": "5432",
    "schema": "hist_horus",
}

# DB SQL server testing
config_object["DB_SER"] = {
    "driver": "{SQL Server}", # ODBC Driver 13 for SQL Server # SQL Server Native Client RDA 11.0
    "user": "sa",
    "pass": "sa$2019",
    "server": "10.72.1.11",
    "database": "HIST_HORUS",
     "port": "5432",
    "schema": "dbo",
}



#Write the above sections to config.ini file
with open('config.ini', 'w') as conf:
    config_object.write(conf)