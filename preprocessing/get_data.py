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

def get_all_sensors_data():
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
        capteurs = pd.read_csv('./output/donnees_capteurs_7jul22.csv')
        capteurs.DH_Date_Observation = pd.to_datetime(capteurs.DH_Date_Observation)
    except FileNotFoundError:
        capteurs = pd.read_sql(con=con_cptr, sql=sql_capt)

    date_min = capteurs.DH_Date_Observation.min()
    date_max = capteurs.DH_Date_Observation.max()

    return capteurs, date_min, date_max

def get_places(no_place_list=[], where_cond=""):
    con_axes = pymssql.connect(**AXES_CONNECTOR)

    sql_places = f"""
    SELECT DISTINCT
          [No_Place],
          [Latitude],
          [Longitude]
    FROM [dbo].[D_Place]
    WHERE 1=1
    """
    sql_places += f"""
    AND No_Place IN {tuple(no_place_list)}
    """ if no_place_list else ""

    sql_places += f""""
    AND {where_cond}
    """ if where_cond else ""

    places = pd.read_sql(con=con_axes, sql=sql_places)
    places = places.groupby('No_Place').first().reset_index()

    return places


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
        capteurs = pd.read_csv('./output/donnees_capteurs_7jul22.csv')
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
        places = pd.read_csv('./output/donnees_places_7jul22.csv')
        places = places.groupby('No_Place').first().reset_index()
    except FileNotFoundError:
        places = pd.read_sql(con=con_axes, sql=sql_places)
        places = places.groupby('No_Place').first().reset_index()
    print('Données de places récupérées.')

    # Récupération des données de transactions
    sql_trans = f"""
    SELECT
       T.[SK_D_Place],
       T.[SK_D_Terrain],
       T.[No_Place],
       T.[No_Trans_Src],
       D.[Date],
       CASE WHEN CAST(T.[DH_Debut_Prise_Place] AS DATE) < D.Date 
            THEN CAST(D.Date as datetime)
            ELSE T.[DH_Debut_Prise_Place]
       END AS [DH_Debut_Prise_Place],
       CASE WHEN CAST(T.[DH_Fin_Prise_Place] AS DATE) > D.Date 
            THEN DATEDIFF(dd, 0, D.Date) + CONVERT(DATETIME,'23:59:59.000')
            ELSE T.[DH_Fin_Prise_Place]
       END AS [DH_Fin_Prise_Place]
    FROM [Axes].[dbo].[D_Date] D
    INNER JOIN [dbo].[F_TransactionPayezPartez] T ON D.Date BETWEEN DH_Debut_Prise_Place AND DH_Fin_Prise_Place
    WHERE DH_Fin_Prise_Place > '{date_min}' AND DH_Debut_Prise_Place < '{date_max}'
    AND No_Place IN {tuple(places.No_Place.to_list())}
    """
    try:
        trans = pd.read_csv('./output/donnees_transac_7jul22.csv')
        trans = trans.drop_duplicates()
    except FileNotFoundError:
        trans = pd.read_sql(con=con_cptr, sql=sql_trans)
        trans = trans.drop_duplicates()
    print('Données de transactions récupérées.')

    sql_regl = f"""
    SELECT 
       [No_Place_Terrain],
       [Code_Regl],
       [Priorite_Regl],
       LEFT([DT_Deb_Regl], 2) AS Deb_Jours_Regl,
	   RIGHT([DT_Deb_Regl], 2) AS Deb_Mois_Regl,
	   LEFT([DT_Fin_Regl], 2) AS Fin_Jours_Regl,
	   RIGHT([DT_Fin_Regl], 2) AS Fin_Mois_Regl,
       [Hr_Deb_Regl],
       [Hr_Fin_Regl],
       [Ind_Interdiction],
       [Ind_Lun],
       [Ind_Mar],
       [Ind_Mer],
       [Ind_Jeu],
       [Ind_Ven],
       [Ind_Sam],
       [Ind_Dim],
	   [MD_Dt_Effectif],
	   [MD_Dt_Expir]
    FROM [Axes].[dbo].[D_Reglement]
    WHERE TYPE_REGL NOT IN ('P', 'Q', 'M', 'D', 'V', 'G')
    AND No_Place_Terrain IN {tuple(places.No_Place.to_list())}
    AND MD_Dt_Expir >= '{date_min.date()}' AND MD_Dt_Effectif <= '{date_max.date()}'
    ORDER BY No_Place_Terrain
    """

    try:
        reglements = pd.read_csv('./output/donnees_reglements_bis_3oct22.csv')
        reglements = reglements.drop_duplicates()
    except FileNotFoundError:
        reglements = pd.read_sql(con=con_axes, sql=sql_regl)
        reglements = reglements.drop_duplicates()
    print('Données de réglements récupérées.')

    # Récupération des périodes réglementaires et permis ODP des places tarifées
    try:
        odp_permits = pd.read_csv('./output/donnees_odp_permits_7jul22.csv')
    except FileNotFoundError:
        periods = [{'from':date_min.date(), 'to':date_max.date()}]
        #reglements = []
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
                # reg = utils.get_data(AXES_CONNECTOR, reglementation_sql.SQL_REGLEMENT,
                                    # reglementation_sql.PLACE_SQL_REG, "",
                                    # tuple(places.No_Place.to_list()), start_date)
                # reg = reg.set_index('No_Place')
                
                # append transactions with regulations
                # reglements.append(reg)
                odp_permits.append(odp)
                
                start_date += delta
        print('\n')

        #reglements = pd.concat(reglements).reset_index()
        odp_permits = pd.concat(odp_permits).reset_index()
    print('Données ODP récupérées.')
    
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
