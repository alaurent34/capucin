# -*- coding: utf-8 -*-
"""
Created on Tue Jun 21 13:07:12 2022

@author: lgauthier
"""

import time

import datetime
import numpy as np
import pandas as pd
from copy import deepcopy

from exceptions import ContinuousStatusExceptedFormatError

TIME_DELTA_BUFFER = 3900 # 65min

class ContinuousStatus:
    '''
    Taxonomie gérée par la classe: 
        1  : occupé
        0  : libre --> n'est pas gardé à la fin du processus car redondant. 
        -2 : inconnu

    On stock les différents types d'observations de la manière suivante:

        self._occ_obs = {
            1  : [[t1, t2), [t4, t7), ..., [tn-1, tn)],
            -2 : [[t7, t8), [t12, t14)]
        }

    La notation [t1, t2) indique un intervalle fermé à t1 et ouvert à t2.

    L'occupation entre deux timestamps t_deb et t_fin pour une journée sera calculée en 
    trouvant tous les états qui surviennent durant cette période.

        Occ_tdeb_tend = Intersect('Occupé', [t_deb, t_fin]) / 
                        ((t_deb - t_fin) - Intersect('Inconu', [t_deb, t_fin]))

    Il est aussi facile de supprimer les observations que l'on juge trop courte.
    '''
    def __init__(self, year: int, month: int, day: int):
        self.yy = year
        self.mm = month
        self.dd = day
        self._occ_obs  = {} # capteurs

    def add_obs(self, t_start: datetime.datetime, t_end: datetime.datetime, status: int):
        try:
            self._occ_obs[status].append((t_start.total_seconds(), t_end.total_seconds()))
        except KeyError:
            self._occ_obs[status] = [(t_start.total_seconds(), t_end.total_seconds())]

    def process_observations(self) -> None:
        pass

    def get_obs(self, hr_start: int = 0, hr_end: int = 24) -> pd.DataFrame:
        occ_obs = self._occ_obs
        data = pd.DataFrame(np.concatenate([np.insert(occ_obs[x], 2, x, axis=1) for x in occ_obs.keys() if occ_obs[x]]), columns=['start', 'end', 'observation'])

        start = hr_start*3600
        end = hr_end*3600
        return data[(data.start < end) & (data.end > start)]

    def get_occ_h(self, hr_start: int = 0, hr_end: int = 24) -> pd.DataFrame:

        occ_h = pd.DataFrame(index=np.arange(hr_start, hr_end, 1), columns=['occ'])

        for hr in range(hr_start, hr_end, 1):
            occ_h.loc[hr, 'occ'] = self.get_occ(hr, hr+1)

        return occ_h

    def get_occ(self, hr_start: int = 0, hr_end: int = 24) -> float:
        # get observation in this period
        obs = self.get_obs(hr_start, hr_end)  

        # set default occupancy divisor and dividend
        avail_time = hr_end*3600 - hr_start*3600
        occup_time = 0

        # remove unknown time
        for row in obs.loc[obs.observation == -2, ['start', 'end']].itertuples():
            avail_time -= min(hr_end*3600, row[2]) - max(hr_start*3600, row[1])

        # add occupancy value
        for row in obs.loc[obs.observation == 1, ['start', 'end']].itertuples():
            occup_time += min(hr_end*3600, row[2]) - max(hr_start*3600, row[1])

        return occup_time / avail_time if avail_time > 0 else 0


class PingStatus(ContinuousStatus):
    ''' Classe qui traite les données de type catpeurs.

    Hérite de ContinuousStatus.

    Permet de gérer les observations de capteurs dans un format d'observations continue.

    '''

    def __init__(self, year: int, month: int, day: int):
        super().__init__(year, month, day)

    def add_obs(self, timestamp: datetime.datetime, status: int):
        self._occ_obs[timestamp] = status

    def _create_unknown_status(self):

        items = list(self._occ_obs.items())

        start = datetime.datetime(self.yy, self.mm, self.dd, 0, 0, 0)
        end = datetime.datetime(self.yy, self.mm, self.dd, 23, 59, 59)

        #add the end point so we can treat between last seen and end of day
        items.append((end, 'PLACEHOLDER'))

        for pos in range(1, len(items)):
            if (items[pos][0] - items[pos-1][0]).total_seconds() > TIME_DELTA_BUFFER: # 65min
                items.insert(pos, (datetime.timedelta(seconds=TIME_DELTA_BUFFER) + items[pos-1][0], -2))

        #find the status of start
        if (items[0][0] - start).total_seconds()  > TIME_DELTA_BUFFER:
            items.insert(0, (start, -2))
        else:
            items.insert(0, (start, items[0][1]))

        #adjust the status of end
        items[-1] = (items[-1][0], items[-2][1])

        self._occ_obs = pd.DataFrame(items, columns=['Timestamp','Status'])

    def _remove_duplicate_status(self) -> None:
        """ Drop all the duplicates status that are consecutive in self._occ_obs.
        Keep the first occurance of the status.

        Raise
        -----
        ContinuousStatusExceptedFormatError

        Example
        -------
        Given a duplicated status observation 

        >>> cs = PingStatus(2022, 7, 6)
        >>> cs.add_obs(datetime.datetime(2022, 7, 6, 0, 1, 0), 'Occupe')
        >>> cs.add_obs(datetime.datetime(2022, 7, 6, 0, 2, 0), 'Occupe')
        >>> cs.add_obs(datetime.datetime(2022, 7, 6, 0, 2, 30), 'Libre')
        >>> cs._occ_obs
        {datetime.datetime(2022, 7, 6, 0, 1): 'Occupe',
         datetime.datetime(2022, 7, 6, 0, 2): 'Occupe',
         datetime.datetime(2022, 7, 6, 0, 2, 30): 'Libre'}

        Call the method to clean the data
        
        >>> cs.remove_duplicate_status()
        >>> cs._occ_obs
        {datetime.datetime(2022, 7, 6, 0, 1): 'Occupe',
         datetime.datetime(2022, 7, 6, 0, 2, 30): 'Libre'}
        """
        
        if not isinstance(self._occ_obs, pd.DataFrame):
            raise ContinuousStatusExceptedFormatError('Dataframe excpected in self._occ_obs')

        self._occ_obs = self._occ_obs[self._occ_obs.Status != self._occ_obs.shift().Status]

    def _transform_observation(self) -> None:
        """ TODO 
        """
        obs = {}
        # recover all parking observations 
        actual_obs = self._occ_obs.copy() # TODO : raise excpetion for non DataFrame
        actual_obs[['NextTimestamp', 'NextStatus']] = actual_obs.shift(-1)[['Timestamp', 'Status']]
        # transform datetime to second in the day
        actual_obs['Timestamp'] = actual_obs['Timestamp'].dt.hour * 3600 +\
                                  actual_obs['Timestamp'].dt.minute * 60 +\
                                  actual_obs['Timestamp'].dt.second 
        actual_obs['NextTimestamp'] = actual_obs['NextTimestamp'].dt.hour * 3600 +\
                                      actual_obs['NextTimestamp'].dt.minute * 60 +\
                                      actual_obs['NextTimestamp'].dt.second 
        # convert to array
        actual_obs = actual_obs[['Timestamp', 'Status', 'NextTimestamp', 'NextStatus']].values

        # fill the obs dict
        for i in range(actual_obs.shape[0]):
            # we care only for parking events
            if actual_obs[i, 1] not in [1, -2]:
                continue
            start = actual_obs[i, 0]
            end = actual_obs[i, 2]
            if actual_obs[i, 3] == -2: # Next observation is unknown
                end -= 3600
            try:
                obs[actual_obs[i, 1]].append((start, end))
            except KeyError:
                obs[actual_obs[i, 1]] = [(start, end)]

        self._occ_obs = obs

    def _remove_noize_status(self, threshold=60) -> None:
        """ Remove noize observation data.

        Paramaters
        ----------
        threshold: int (Default: 60).
            time delta in second where a parking observation is removed.

        Raise
        -----
        ContinuousStatusExceptedFormatError
        """

        try:
            tmp = deepcopy(self._occ_obs[1])
            self._occ_obs[1] = [(s, e) for s,e in tmp if e - s >= threshold]
        except KeyError:
            pass
        
    def process_observations(self, threshold=60) -> None:
        ''' Complete all the stages requiered to transform occupancy data to 
        PingStatus data model.

        This should be done after the addition of all the observation for the 
        day.

        Parameters
        ----------
        threshold: int (Default: 60).
            time delta in second where a parking observation is removed.

        Raise
        -----
        ContinuousStatusExceptedFormatError: Exception
        '''
        self._create_unknown_status()
        self._remove_duplicate_status()
        self._transform_observation()

        try:
            self._remove_noize_status(threshold=threshold)
        except ContinuousStatusExceptedFormatError as e:
            raise e

class TransactionsStatus(ContinuousStatus):
    pass

class LPRStatus(ContinuousStatus):
    pass


class SingleParkingSpot:
    ''' TODO: chaque type de données doit avoir un days. Ou alors on fait juste autant de
    ParkingCollection que de type de données. Je penses que c'est le mieux d'ailleurs.
    '''

    def __init__(self, _id: str):

        self._id = _id
        self.days = {}

    def add_ping_obs(self, timestamp: datetime.datetime, status: str):
        date = timestamp.date()
        if date not in list(self.days.keys()):
            self.days[date] = PingStatus(timestamp.year, timestamp.month, timestamp.day)
        self.days[date].add_obs(timestamp, status)

    def finalise_ping_import(self):
        for k in self.days.keys():
            self.days[k].process_observations(threshold=60)

    def get_obs(self, day_start: str ='1970-01-01', day_end: str = '2099-12-31', hr_start: int = 0, hr_end: int = 24):
        day_start = datetime.datetime.strptime(day_start, "%Y-%m-%d").date()
        day_end = datetime.datetime.strptime(day_end, "%Y-%m-%d").date()

        assert day_start <= day_end, 'day_start value cannot be bigger than day_end value'

        if min(self.days.keys()) > day_end or max(self.days.keys()) < day_start:
            return pd.DataFrame()
        
        day_start = max(day_start, min(self.days.keys()))
        day_end = min(day_start, max(self.days.keys()))
        delta = datetime.timedelta(days=1)

        obs = []
        while day_start <= day_end:
            try:
                day_obs = self.days[day_start].get_obs(hr_start, hr_end)
            except KeyError:
                day_start += delta
                continue

            day_obs['date'] = day_start
            obs.append(day_obs)

            day_start += delta

        return pd.concat(obs).reindex(columns=['date', 'start', 'end', 'observation'])

    def get_occ(self, day_start: str ='1970-01-01', day_end: str = '2099-12-31', hr_start: int = 0, hr_end: int = 24):
        day_start = datetime.datetime.strptime(day_start, "%Y-%m-%d").date()
        day_end = datetime.datetime.strptime(day_end, "%Y-%m-%d").date()

        assert day_start <= day_end, 'day_start value cannot be bigger than day_end value'

        if min(self.days.keys()) > day_end or max(self.days.keys()) < day_start:
            return pd.DataFrame()

        day_start = max(day_start, min(self.days.keys()))
        day_end = min(day_start, max(self.days.keys()))
        delta = datetime.timedelta(days=1)

        occ = []
        while day_start <= day_end:
            try:
                day_occ = self.days[day_start].get_occ(hr_start, hr_end)
            except KeyError:
                day_start += delta
                continue

            occ.append([day_start, day_occ])

            day_start += delta

        return pd.DataFrame(occ, columns=['date', 'occ'])
        

    def get_occ_h(self, day_start: str ='1970-01-01', day_end: str = '2099-12-31', hr_start: int = 0, hr_end: int = 24):
        day_start = datetime.datetime.strptime(day_start, "%Y-%m-%d").date()
        day_end = datetime.datetime.strptime(day_end, "%Y-%m-%d").date()

        assert day_start <= day_end, 'day_start value cannot be bigger than day_end value'

        if min(self.days.keys()) > day_end or max(self.days.keys()) < day_start:
            return pd.DataFrame()

        day_start = max(day_start, min(self.days.keys()))
        day_end = min(day_start, max(self.days.keys()))
        delta = datetime.timedelta(days=1)

        occ_h = []
        while day_start <= day_end:
            try:
                day_occ_h = self.days[day_start].get_occ_h(hr_start, hr_end).reset_index().rename(columns={'index':'hour'})
            except KeyError:
                day_start += delta
                continue

            day_occ_h['date'] = day_start
            occ_h.append(day_occ_h)

            day_start += delta

        return pd.concat(occ_h).reindex(columns=['date', 'hour', 'occ'])


class ParkingSpotCollection:

    def __init__(self):
        self.spots = {}

    def read_ping_data(self, dframe: pd.DataFrame, columns_conf: dict = {'DH_Date_Observation': 'timestamp', 'No_Place':'spot_id', 'Valeur_Observee':'obs'},
            obs_conf: dict = {'Occupé':1, 'Libre':0, 'Inconnu':-2}):

        fsorted = dframe.sort_values(by='DH_Date_Observation')
        fsorted.rename(columns=columns_conf, inplace=True)
        fsorted.obs = fsorted.obs.map(obs_conf)

        for _,row in fsorted.iterrows():
            self.populate_ping(row.spot_id, row.timestamp, row.obs)

        self._finalise_occ_import()

    def _finalise_occ_import(self):
        for k in self.spots:
            self.spots[k].finalise_ping_import()

    def populate_ping(self, spot_id: str, timestamp: datetime.datetime, status: str):
        if spot_id not in self.spots:
            self.spots[spot_id] = self.create_spot(spot_id)

        self.spots[spot_id].add_ping_obs(timestamp, status)

    def create_spot(self, spot_id):
        return SingleParkingSpot(spot_id)

    @property
    def days(self):
        return np.unique(np.asarray([spot.days.keys() for k,spot in self.spots.items()]))

    #@overload(get_day_rawdata)

    def get_day_obs(self, day: datetime.datetime.date) -> pd.DataFrame:
        raw = []
        for spot_name, spot in self.spots.items():
            spot_obs = spot.get_obs(day_start=day.strftime('%Y-%m-%d'), day_end=day.strftime('%Y-%m-%d'))
            spot_obs['spot'] = spot_name
            raw.append(spot_obs)


        return pd.concat(raw).reindex(columns=['spot', 'date', 'start', 'end', 'observation'])

    def get_day_occ(self, day: datetime.datetime.date) -> pd.DataFrame:
        raw = []
        for spot_name, spot in self.spots.items():
            spot_occ = spot.get_occ(day_start=day.strftime('%Y-%m-%d'), day_end=day.strftime('%Y-%m-%d'))
            spot_occ['spot'] = spot_name
            raw.append(spot_occ)


        return pd.concat(raw).reindex(columns=['spot', 'date', 'occ'])

    def get_day_occ_h(self, day: datetime.datetime.date) -> pd.DataFrame:
        raw = []
        for spot_name, spot in self.spots.items():
            spot_occ_h = spot.get_occ_h(day_start=day.strftime('%Y-%m-%d'), day_end=day.strftime('%Y-%m-%d'))
            spot_occ_h['spot'] = spot_name
            raw.append(spot_occ_h)


        return pd.concat(raw).reindex(columns=['spot', 'date', 'hour', 'occ'])



if __name__ == '__main__':
    #TODO: CHARGEMENT ET TRAITEMENT DES TRANSACTIONS

    #import raw data
    capteurs = pd.read_csv(r'./02_Intrants/Données de stationnement/donnees_capteurs_22jui22.csv')
    cols_capteurs = {'DH_Date_Observation': 'timestamp', 'No_Place':'spot_id', 'Valeur_Observee':'obs'}
    capteurs_value = {'Occupé':1, 'Libre':0, 'Inconnu':-2}

    #transform the dates
    capteurs.loc[:,'DH_Date_Observation'] = pd.to_datetime(capteurs.DH_Date_Observation)
    capteurs = capteurs.sort_values('DH_Date_Observation')

    parkings = ParkingSpotCollection()
    parkings.read_ping_data(capteurs.iloc[:5000], columns_conf=cols_capteurs, obs_conf=capteurs_value)

    capteurs_obs = parkings.get_day_obs(day=datetime.datetime(2021, 6, 17).date())
    print('Observations :\n', capteurs_obs.head(10))

    capteurs_occ_moy = parkings.get_day_occ(day=datetime.datetime(2021, 6, 17).date()) 
    print('Occupation moyenne :\n', capteurs_occ_moy.head(10))
    
    capteurs_occ_h = parkings.get_day_occ_h(day=datetime.datetime(2021, 6, 17).date()) 
    print('Ocupation horraire :\n', capteurs_occ_h.head(10))