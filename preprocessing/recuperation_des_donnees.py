#!/usr/bin/env python
# coding: utf-8
    
import os
import sys
import time
import datetime
from babel.dates import format_date, format_datetime, format_time

import numpy as np
import pandas as pd
import geopandas as gpd
import pymssql
import reglementation_sql, permits_sql
from lapin.transactions import utils
from lapin.transactions import constants

# Sauvegarde des fichiers
OUTPUT = './'

# Configuration de connection
CPTR_CONNECTOR = {
    'server':'prisqlbiprod01',
    'database':'CPTR_Station'
}
AXES_CONNECTOR = {
    'server':'prisqlbiprod01',
    'database':'Axes'
}

def main():

    # Recupération des données de capteurs
    con_cptr = pymssql.connect(**CPTR_CONNECTOR)
    sql_capt = f"""
    SELECT 
        SK_D_Place,
        CASE WHEN F.No_Place LIKE 'COMMUNAUTO-%' THEN R.No_Place
             ELSE F.No_Place
        END AS No_Place,
        Type_Observation,
        Valeur_Observee,
        Unite_Observation,
        DH_Date_Observation
    FROM dbo.F_ActiviteCapteursPilotThings F
    LEFT JOIN dbo.F_rel_Place_VLS R ON R.No_Place_vls = F.No_Place
    WHERE Type_Observation = 'Stationnement'
    """
    try:
        capteurs = pd.read_csv('./donnees_capteurs_22jui22.csv')
        capteurs.DH_Date_Observation = pd.to_datetime(capteurs.DH_Date_Observation)
    except FileNotFoundError:
        capteurs = pd.read_sql(con=con_cptr, sql=sql_capt)

    date_min = capteurs.DH_Date_Observation.min()
    date_max = capteurs.DH_Date_Observation.max()
    print('Données de capteurs récupérées.')

    # Récupération des places de stationnements
    con_axes = pymssql.connect(**AXES_CONNECTOR)
    sql_places = f"""
    SELECT DISTINCT
          [No_Place],
          [Latitude],
          [Longitude]
    FROM [dbo].[D_Place]
    WHERE No_Place IN {tuple(list(capteurs.No_Place.unique()))}
    """
    try:
        places = pd.read_csv('donnees_places_22jui22.csv')
        places = places.groupby('No_Place').first().reset_index()
    except FileNotFoundError:
        places = pd.read_sql(con=con_axes, sql=sql_places)
        places = places.groupby('No_Place').first().reset_index()
    print('Données de places récupérées.')

    # Récupération des données de transactions
    sql_trans = f"""
    SELECT
        [SK_D_Place],
        [SK_D_Terrain],
        [No_Place],
        [No_Trans_Src],
        [DH_Debut_Prise_Place],
        [DH_Fin_Prise_Place]
    FROM [dbo].[F_TransactionPayezPartez]
    WHERE DH_Fin_Prise_Place > '{date_min}' AND DH_Debut_Prise_Place < '{date_max}'
    AND No_Place IN {tuple(places.No_Place.to_list())}
    """
    try:
        trans = pd.read_csv('donnees_transac_22jui22.csv')
        trans = trans.drop_duplicates()
    except FileNotFoundError:
        trans = pd.read_sql(con=con_cptr, sql=sql_trans)
        trans = trans.drop_duplicates()
    print('Données de transactions récupérées.')

    # Récupération des périodes réglementaires et permis ODP des places tarifées
    periods = [{'from':date_min.date(), 'to':date_max.date()}]
    reglements = []
    odp_permits = []

    for period in periods:
        start_date = period['from']
        end_date = period['to']
        delta = datetime.timedelta(days=1)
        
        while start_date <= end_date:
            sys.stdout.write(f"\r {start_date} - {end_date}")
            sys.stdout.flush()
            
            ## ODP
            odp = utils.get_data(AXES_CONNECTOR, permits_sql.SQL_ODP, permits_sql.PLACE_COND_ODP,
                                 permits_sql.DATE_COND_ODP, tuple(places.No_Place.to_list()), start_date)
            odp = odp.set_index('No_Place')
            
            ## Reglementation
            reg = utils.get_data(AXES_CONNECTOR, reglementation_sql.SQL_REGLEMENT,
                                 reglementation_sql.PLACE_SQL_REG, "",
                                 tuple(places.No_Place.to_list()), start_date)
            reg = reg.set_index('No_Place')
            
            # append transactions with regulations
            reglements.append(reg)
            odp_permits.append(odp)
            
            start_date += delta
    print('\n')

    reglements = pd.concat(reglements).reset_index()
    odp_permits = pd.concat(odp_permits).reset_index()
    
    # Sauvegarde
    day = [dt[:3] if i==1 else dt[-2:] for i,dt in enumerate(format_date(datetime.datetime.today(), format='medium', locale='fr_CA').split(' ')) if not dt.startswith('juil')]
    day = ''.join(day) if len(day) > 2 else 'jul'.join(day)

    capteurs.to_csv(os.path.join(OUTPUT, f'donnees_capteurs_{day}.csv'), index=False)
    places.to_csv(os.path.join(OUTPUT, f'donnees_places_{day}.csv'), index=False)
    trans.to_csv(os.path.join(OUTPUT, f'donnees_transac_{day}.csv'), index=False)
    reglements.to_csv(os.path.join(OUTPUT, f'donnees_reglements_{day}.csv'), index=False)
    odp_permits.to_csv(os.path.join(OUTPUT, f'donnees_odp_permits_{day}.csv'), index=False)

    print('Sauvegarde effectuée.')

if __name__ == '__main__':
    main()
