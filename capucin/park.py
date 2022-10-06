# -*- coding: utf-8 -*-
"""
Created on Tue Jun 21 13:07:12 2022

@author: lgauthier
"""

import itertools

import datetime
from re import search
from time import time
from turtle import st
import numpy as np
import pandas as pd
import pandasql as ps
from copy import deepcopy

from .exceptions import ContinuousStatusExceptedFormatError

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

        if not occ_obs:
            return pd.DataFrame(columns=['start', 'end', 'observation'])

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
        duplicates_rm = self._occ_obs[self._occ_obs.Status != self._occ_obs.shift().Status]
        #duplicates_rm = pd.concat([duplicates_rm, self._occ_obs.tail(1)])
        self._occ_obs = duplicates_rm

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
            if actual_obs[i, 1] not in [0, 1, -2]:
                continue
            start = actual_obs[i, 0]
            end = actual_obs[i, 2]
            if actual_obs[i, 3] == -2: # Next observation is unknown
                end -= 3600
                actual_obs[i+1, 0] -= 3600
            if pd.isna(actual_obs[i, 2]): # end of the day
                end = 23 * 3600 + 59 * 60 + 59
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
        """

        df = self._occ_obs.copy()

        df[['NextTimestamp', 'NextStatus']] = df.shift(-1)
        df.NextTimestamp = df.NextTimestamp.fillna(pd.to_datetime(df.NextTimestamp.iloc[0].date())+datetime.timedelta(hours=23, minutes=59, seconds=59))
        df.loc[df.NextStatus.isna(), 'NextStatus'] = df[df.NextStatus.isna()].Status

        df_clean = df.copy().reset_index(drop=True)
        for i, row in df.reset_index(drop=True).iterrows():
            if row.Status == 0:
                continue
            if (row.NextTimestamp - row.Timestamp).total_seconds() <= 60:
                df_clean.drop(i, inplace=True)
        df_clean = df_clean[['Timestamp', 'Status']]

        self._occ_obs = df_clean
        self._remove_duplicate_status()
        

    def _remove_noize_status_old(self, threshold=60) -> None:
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

    def _clean_empty_data(self) -> None:
        occ_obs = self._occ_obs
        
        occ_obs_cleaned = {k: v for k, v in occ_obs.items() if v}

        self._occ_obs = occ_obs_cleaned
        
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
        self._remove_noize_status(threshold=threshold)
        self._transform_observation()
        self._clean_empty_data()

class TransactionsStatus(ContinuousStatus):
    def __init__(self, year: int, month: int, day: int):
        super().__init__(year, month, day)
        self._id_trsc = []

    def add_obs(self, t_start: datetime.datetime, t_end: datetime.datetime, id_trsc: np.int64 = -2) -> None:
        st = t_start.hour * 3600 + t_start.minute * 60 + t_start.second
        en = t_end.hour * 3600 + t_end.minute * 60 + t_end.second
        try:
            self._occ_obs[1].append((st, en))
        except KeyError:
            self._occ_obs[1] = [(st, en)]
        self._id_trsc.append(id_trsc)

    def get_obs(self, hr_start: int = 0, hr_end: int = 24) -> pd.DataFrame:
        occ_obs = self._occ_obs

        if not occ_obs:
            return pd.DataFrame(columns=['start', 'end', 'observation', 'id_trsc'])

        data = pd.DataFrame(np.concatenate([np.insert(occ_obs[x], 2, x, axis=1) for x in occ_obs.keys() if occ_obs[x]]), columns=['start', 'end', 'observation'])
        data['id_trsc'] = self._id_trsc

        start = hr_start*3600
        end = hr_end*3600
        return data[(data.start < end) & (data.end > start)]

    def get_event_paid(self, t_start: datetime.datetime, t_end: datetime.datetime, buffer: int=300) -> int:
        hr_st = t_start.hour
        hr_en = t_end.hour

        trsc = self.get_obs(hr_start=hr_st-1, hr_end=hr_en+1)
        if trsc.empty:
            return [], False, 0

        t_st = hr_st * 3600 + t_start.minute * 60 + t_start.second
        t_en = hr_en * 3600 + t_end.minute * 60 + t_end.second
        trsc = trsc[(trsc.start + buffer > t_st) & (trsc.start < t_en) & (trsc.end > t_st)].copy()
        trsc.sort_values('start', inplace=True)
        has_paid = True if not trsc.empty else False

        paid_time=0
        for row in trsc.loc[:, ['start', 'end']].itertuples():
            paid_time += row[2] - row[1]

        return trsc.id_trsc.to_list(), has_paid, paid_time

    def search_next_state(self, t_start: datetime.datetime, t_end: datetime.datetime):
        hr_st = t_start.hour
        hr_en = t_end.hour + 1

        data = self.get_obs(hr_start=hr_st, hr_end=hr_en)
        data.sort_values(['start'], inplace=True)

        trsc_concu = []
        for _,trsc in data.iterrows():
            # Several time representation of regulations
            trsc_hr_from  = int(trsc.start // 3600)
            trsc_min_from = int(trsc.start % 3600 // 60)
            trsc_sec_from = int(trsc.start % 3600 % 60)

            trsc_hr_to  = int(trsc.end // 3600)
            trsc_min_to = int(trsc.end % 3600 // 60)
            trsc_sec_to = int(trsc.end % 3600 % 60)
 
            trsc_ts_from = pd.to_datetime(f"{t_start.year}-{t_start.month}-{t_start.day} {trsc_hr_from:02d}:{trsc_min_from:02d}:{trsc_sec_from:02d}")
            trsc_ts_to = pd.to_datetime(f"{t_start.year}-{t_start.month}-{t_start.day} {trsc_hr_to:02d}:{trsc_min_to:02d}:{trsc_sec_to:02d}")

            if t_start < trsc_ts_to and t_end > trsc_ts_from:
                trsc_concu.append([max(t_start, trsc_ts_from), min(t_end, trsc_ts_to), [trsc.id_trsc]])

        if not trsc_concu:
            return t_start, t_end, t_start, t_end, -2 # No informations

        result = trsc_concu[0]
        for t in trsc_concu[1:]:
            # if they begin at the same time, then end_time is min of both end time
            if result[0] == t[0]:
                result[1] = min(result[1], t[1])
                result[2].extend(t[2]) # transactions overlap
            # begining of transaction t is before the end saved, then beg_time is end time
            elif result[1] > t[0]:
                result[1] = t[0]

        return t_start, t_end, result[0], result[1], result[2]
                
    def get_all_states(self, t_start: datetime.datetime, t_end: datetime.datetime):
        result = []

        while t_start < t_end:
            p_s, p_e, r_s, r_e, id_trsc = self.search_next_state(t_start, t_end)
            if p_s < r_s:
                result.append({'from':p_s, 'to':r_s, 'has_been_paid':False, 'id_trsc':-2})
            result.append({'from':r_s, 'to':r_e, 'has_been_paid':True if id_trsc != -2 else False, 'id_trsc': id_trsc}) 
            t_start = r_e

        return result

class PermisSatus(object):
    def __init__(self, year: int, month: int, day: int):
        self.yy = year
        self.mm = month
        self.dd = day
        self._permits = [] 

    def add_permits(self, t_start: datetime.datetime, t_end: datetime.datetime):
        st = t_start.hour * 3600 + t_start.minute * 60 + t_start.second
        en = t_end.hour * 3600 + t_end.minute * 60 + t_end.second
        self._permits.append((st, en))

    def get_start_state(self, timestamp: datetime.datetime):
        tt = timestamp.hour * 3600 + timestamp.minute * 60 + timestamp.second
        for perm in self._permits:
            if perm[0] <= tt <= perm[1]:
                return True
        return False

    def search_next_state(self, t_start: datetime.datetime, t_end: datetime.datetime):

        data = pd.DataFrame(self._permits, columns=['start', 'end'])
        data.sort_values(['start'], inplace=True)

        for _,perm in data.iterrows():
            # Several time representation of regulations
            perm_hr_from  = int(perm.start // 3600)
            perm_min_from = int(perm.start % 3600 // 60)
            perm_sec_from = int(perm.start % 3600 % 60)

            perm_hr_to  = int(perm.end // 3600)
            perm_min_to = int(perm.end % 3600 // 60)
            perm_sec_to = int(perm.end % 3600 % 60)
 
            perm_ts_from = pd.to_datetime(f"{t_start.year}-{t_start.month}-{t_start.day} {perm_hr_from:02d}:{perm_min_from:02d}:{perm_sec_from:02d}")
            perm_ts_to = pd.to_datetime(f"{t_start.year}-{t_start.month}-{t_start.day} {perm_hr_to:02d}:{perm_min_to:02d}:{perm_sec_to:02d}")

            if t_start < perm_ts_to and t_end > perm_ts_from:
                return t_start, t_end, max(t_start, perm_ts_from), min(t_end, perm_ts_to), True
        return t_start, t_end, t_start, t_end, False # No informations
                
    def get_all_states(self, t_start: datetime.datetime, t_end: datetime.datetime):
        result = []

        while t_start < t_end:
            p_s, p_e, r_s, r_e, is_trsc = self.search_next_state(t_start, t_end)
            if p_s < r_s:
                result.append({'from':p_s, 'to':r_s, 'permit':False})
            result.append({'from':r_s, 'to':r_e, 'permit': is_trsc}) 
            t_start = r_e

        return result



class ReglementationStatus(object):
    def __init__(self):
        self._regls = {} # 1 : permis 2: interdit -2: Unkown state

    def add_regl(self, regl_type: int, regl: dict):
        try:
            self._regls[regl_type].append(regl)
        except KeyError:
            self._regls[regl_type] = [regl]

    def get_state(self, timestamp: datetime.datetime):
        result = {1:False, 2:False, -2:True}
        for regl_type, regls in self._regls.items():
            for regl in regls:
                time_from = int(regl['start_hr'][:2]) * 3600 + int(regl['start_hr'][3:5]) * 60 + int(regl['start_hr'][6:])
                time_to   = int(regl['end_hr'][:2]) * 3600 + int(regl['end_hr'][3:5]) * 60 + int(regl['end_hr'][6:])
                time_asked = timestamp.hour * 3600 + timestamp.minute * 60 + timestamp.second

                dt_from = pd.to_datetime(f"1970-{regl['start_month']}-{regl['start_day']}")
                dt_to = pd.to_datetime(f"1970-{regl['end_month']}-{regl['end_day']}")
                dt_asked = pd.to_datetime(f"1970-{timestamp.month}-{timestamp.day}")

                if regl['valid_from'] <= timestamp <= regl['valid_to'] \
                        and regl[timestamp.day_name()[:3].lower()] \
                        and time_from <= time_asked <= time_to \
                        and dt_from <= dt_asked <= dt_to:
                    result[regl_type] = True
                    result[-2] = False
                    break
                else:
                    continue

        return result

    def search_next_state(self, t_start: datetime.datetime, t_end: datetime.datetime):
        data = pd.DataFrame.from_dict(list(itertools.chain.from_iterable(self._regls.values())))
        data.sort_values(['priority', 'start_hr'], inplace=True)
        for _,regl in data.iterrows():
            # Several time representation of regulations
            reg_hr_from = int(regl['start_hr'][:2])
            reg_min_from = int(regl['start_hr'][3:5])
            reg_sec_from = int(regl['start_hr'][6:])

            reg_hr_to = int(regl['end_hr'][:2])
            reg_min_to = int(regl['end_hr'][3:5])
            reg_sec_to = int(regl['end_hr'][6:])

            reg_ts_from = pd.to_datetime(f"{t_start.year}-{t_start.month}-{t_start.day} {reg_hr_from:02d}:{reg_min_from:02d}:{reg_sec_from:02d}")
            reg_ts_to = pd.to_datetime(f"{t_start.year}-{t_start.month}-{t_start.day} {reg_hr_to:02d}:{reg_min_to:02d}:{reg_sec_to:02d}")

            dt_from = pd.to_datetime(f"{t_start.year}-{regl['start_month']}-{regl['start_day']}")
            dt_to = pd.to_datetime(f"{t_start.year}-{regl['end_month']}-{regl['end_day']}")

            if regl[t_start.day_name()[:3].lower()] \
                    and dt_from.date() <= t_start.date() <= dt_to.date() \
                    and regl['valid_from'].date() <= t_start.date() <= regl['valid_to'].date() \
                    and t_start < reg_ts_to and t_end > reg_ts_from:

                return t_start, t_end, max(t_start, reg_ts_from), min(t_end, reg_ts_to), regl['regl_type']
        return t_start, t_end, t_start, t_end, -2 # No informations
                
    def get_all_states(self, t_start: datetime.datetime, t_end: datetime.datetime):
        result = []

        while t_start < t_end:
            p_s, p_e, r_s, r_e, reg_type = self.search_next_state(t_start, t_end)
            if p_s < r_s:
                result.append({'from':p_s, 'to':r_s, 'regl_type':-2})
            result.append({'from':r_s, 'to':r_e, 'regl_type': reg_type}) 
            t_start = r_e

        return result

class LPRStatus(ContinuousStatus):
    pass


class SingleParkingSpot:
    ''' TODO: chaque type de données doit avoir un days. Ou alors on fait juste autant de
    ParkingCollection que de type de données. Je penses que c'est le mieux d'ailleurs.
    '''

    def __init__(self, _id: str):

        self._id = _id
        self.occ = {}
        self.trsc = {}
        self.regls = ReglementationStatus()
        self.permits = {}

    def add_ping_obs(self, timestamp: datetime.datetime, status: str):
        date = timestamp.date()
        if date not in list(self.occ.keys()):
            self.occ[date] = PingStatus(timestamp.year, timestamp.month, timestamp.day)
        self.occ[date].add_obs(timestamp, status)

    def finalise_ping_import(self):
        for k in self.occ.keys():
            self.occ[k].process_observations(threshold=60)

    def add_trsc_obs(self, start: datetime.datetime, end: datetime.datetime, id_trsc: np.int64 = -2):
        date = start.date()
        if date not in list(self.trsc.keys()):
            self.trsc[date] = TransactionsStatus(date.year, date.month, date.day)
        self.trsc[date].add_obs(start, end, id_trsc)

    def add_permits(self, start: datetime.datetime, end: datetime.datetime):
        date = start.date()
        if date not in list(self.permits.keys()):
            self.permits[date] = PermisSatus(date.year, date.month, date.day)
        self.permits[date].add_permits(start, end)

    def add_regl(self, regl_type: int, regl: dict):
        self.regls.add_regl(regl_type, regl)

    def get_matched_obs(
        self, 
        day_start: datetime.datetime.date = datetime.datetime(1970,1,1).date(),
        day_end: datetime.datetime.date = datetime.datetime(2099,12,31).date(),
        hr_start: int = 0,
        hr_end: int = 24
    ) -> pd.DataFrame:
        """TODO
        """
        obs = self.get_obs(day_start, day_end, hr_start, hr_end)
        obs = obs.reset_index(drop=True).reset_index().rename(columns={'index':'event_id'})
        trsc = self.get_trsc(day_start, day_end, hr_start, hr_end)
        if obs.empty:
            return pd.DataFrame()

        obs.sort_values(['date', 'start'], inplace=True)
        obs_old = obs.copy()

        matched = []
        # Has paid
        for _, event in obs.iterrows():
            # date start
            hr_st = event['start'] // 3600
            min_st = (event['start'] % 3600) // 60
            sec_st = (event['start'] % 3600) % 60
            date_st = pd.to_datetime(event['date']) + datetime.timedelta(hours=hr_st, minutes=min_st, seconds=sec_st)

            # date end
            hr_end = event['end'] // 3600
            min_end = (event['end'] % 3600) // 60
            sec_end = (event['end'] % 3600) % 60
            date_end = pd.to_datetime(event['date']) + datetime.timedelta(hours=hr_end, minutes=min_end, seconds=sec_end)

            if date_st.date() in list(self.trsc.keys()) and event['observation'] == 1:
                event_cp = event.copy()

                ids_trscs, has_paid, time_paid = self.trsc[date_st.date()].get_event_paid(date_st, date_end)
                event_cp['first_trsc'] = ids_trscs[0] if ids_trscs else -2
                event_cp['has_paid'] = has_paid
                event_cp['sec_paid'] = time_paid
                matched.append(event_cp)
            else:
                event_cp = event.copy()
                event_cp['first_trsc'] = -2
                event_cp['has_paid'] = False
                event_cp['sec_paid'] = 0
                matched.append(event_cp)

        obs = pd.DataFrame(matched)

        matched = []
        # ODP
        for _, event in obs.iterrows():
            # date start
            hr_st = event['start'] // 3600
            min_st = (event['start'] % 3600) // 60
            sec_st = (event['start'] % 3600) % 60
            date_st = pd.to_datetime(event['date']) + datetime.timedelta(hours=hr_st, minutes=min_st, seconds=sec_st)

            # date end
            hr_end = event['end'] // 3600
            min_end = (event['end'] % 3600) // 60
            sec_end = (event['end'] % 3600) % 60
            date_end = pd.to_datetime(event['date']) + datetime.timedelta(hours=hr_end, minutes=min_end, seconds=sec_end)

            if date_st.date() in list(self.permits.keys()):
                permits = self.permits[date_st.date()].get_all_states(date_st, date_end)
                for perm in permits:
                    event_cp = event.copy()

                    event_cp['start'] = perm['from'].hour * 3600 + perm['from'].minute * 60 + perm['from'].second
                    event_cp['end'] = perm['to'].hour * 3600 + perm['to'].minute * 60 + perm['to'].second
                    event_cp['has_permits'] = perm['permit']
                    matched.append(event_cp)
            else:
                event_cp = event.copy()
                event_cp['has_permits'] = False
                matched.append(event_cp)

        obs = pd.DataFrame(matched)

        matched = []
        # Regulations
        for _, event in obs.iterrows():
            # date start
            hr_st = event['start'] // 3600
            min_st = (event['start'] % 3600) // 60
            sec_st = (event['start'] % 3600) % 60
            date_st = pd.to_datetime(event['date']) + datetime.timedelta(hours=hr_st, minutes=min_st, seconds=sec_st)

            # date end
            hr_end = event['end'] // 3600
            min_end = (event['end'] % 3600) // 60
            sec_end = (event['end'] % 3600) % 60
            date_end = pd.to_datetime(event['date']) + datetime.timedelta(hours=hr_end, minutes=min_end, seconds=sec_end)


            regulations = self.regls.get_all_states(date_st, date_end)
            for reg in regulations:
                event_cp = event.copy()

                event_cp['start'] = reg['from'].hour * 3600 + reg['from'].minute * 60 + reg['from'].second 
                event_cp['end'] = reg['to'].hour * 3600 + reg['to'].minute * 60 + reg['to'].second
                event_cp['regulated'] = True if reg['regl_type'] == 1 else False
                event_cp['prohibited'] = True if reg['regl_type'] == 2 else False
                event_cp['unknown'] = True if reg['regl_type'] == -2 else False
                matched.append(event_cp)

        obs = pd.DataFrame(matched)

        matched = []
        # Transactions
        for _, event in obs.iterrows():
            # date start
            hr_st = event['start'] // 3600
            min_st = (event['start'] % 3600) // 60
            sec_st = (event['start'] % 3600) % 60
            date_st = pd.to_datetime(event['date']) + datetime.timedelta(hours=hr_st, minutes=min_st, seconds=sec_st)

            # date end
            hr_end = event['end'] // 3600
            min_end = (event['end'] % 3600) // 60
            sec_end = (event['end'] % 3600) % 60
            date_end = pd.to_datetime(event['date']) + datetime.timedelta(hours=hr_end, minutes=min_end, seconds=sec_end)

            if date_st.date() in list(self.trsc.keys()):
                trscs = self.trsc[date_st.date()].get_all_states(date_st, date_end)
                for trsc in trscs:
                    event_cp = event.copy()

                    event_cp['start'] = trsc['from'].hour * 3600 + trsc['from'].minute * 60 + trsc['from'].second
                    event_cp['end'] = trsc['to'].hour * 3600 + trsc['to'].minute * 60 + trsc['to'].second
                    event_cp['has_been_paid'] = trsc['has_been_paid']
                    event_cp['id_trsc'] = trsc['id_trsc']
                    matched.append(event_cp)
            else:
                event_cp = event.copy()
                event_cp['has_been_paid'] = False
                event_cp['id_trsc'] = -2
                matched.append(event_cp)
        
        df = pd.DataFrame(matched)#, obs_old, obs_permits, obs_regulation
        df = df[["event_id", "date", "start", "end", "observation",	"has_permits",
                 "regulated", "prohibited", "unknown", "has_been_paid", "id_trsc",
                 "has_paid", "first_trsc", "sec_paid"]] 

        return df

    def get_obs(
        self, 
        day_start: datetime.datetime.date = datetime.datetime(1970,1,1).date(),
        day_end: datetime.datetime.date = datetime.datetime(2099,12,31).date(),
        hr_start: int = 0,
        hr_end: int = 24
    ):
        if not self.occ:
            print('No occupation data.')
            return pd.DataFrame()

        assert day_start <= day_end, 'day_start value cannot be bigger than day_end value'

        if min(self.occ.keys()) > day_end or max(self.occ.keys()) < day_start:
            return pd.DataFrame()
        
        day_start = max(day_start, min(self.occ.keys()))
        day_end = min(day_end, max(self.occ.keys()))
        delta = datetime.timedelta(days=1)

        obs = []
        while day_start <= day_end:
            try:
                day_obs = self.occ[day_start].get_obs(hr_start, hr_end)
            except KeyError:
                day_start += delta
                continue

            day_obs['date'] = day_start
            obs.append(day_obs)

            day_start += delta

        return pd.concat(obs).reindex(columns=['date', 'start', 'end', 'observation'])

    def get_trsc(
        self, 
        day_start: datetime.datetime.date = datetime.datetime(1970,1,1).date(),
        day_end: datetime.datetime.date = datetime.datetime(2099,12,31).date(),
        hr_start: int = 0,
        hr_end: int = 24
    ):
        assert day_start <= day_end, 'day_start value cannot be bigger than day_end value'

        if (not self.trsc) or (min(self.trsc.keys()) > day_end or max(self.trsc.keys()) < day_start):
            return pd.DataFrame()
        
        day_start = max(day_start, min(self.trsc.keys()))
        day_end = min(day_end, max(self.trsc.keys()))
        delta = datetime.timedelta(days=1)

        obs = []
        while day_start <= day_end:
            try:
                day_obs = self.trsc[day_start].get_obs(hr_start, hr_end)
            except KeyError:
                day_start += delta
                continue

            day_obs['date'] = day_start
            obs.append(day_obs)

            day_start += delta

        return pd.concat(obs).reindex(columns=['date', 'start', 'end', 'observation', 'id_trsc'])

    def get_occ(
        self, 
        day_start: datetime.datetime.date = datetime.datetime(1970,1,1).date(),
        day_end: datetime.datetime.date = datetime.datetime(2099,12,31).date(),
        hr_start: int = 0,
        hr_end: int = 24
    ):
        assert day_start <= day_end, 'day_start value cannot be bigger than day_end value'

        if min(self.occ.keys()) > day_end or max(self.occ.keys()) < day_start:
            return pd.DataFrame()

        day_start = max(day_start, min(self.occ.keys()))
        day_end = min(day_end, max(self.occ.keys()))
        delta = datetime.timedelta(days=1)

        occ = []
        while day_start <= day_end:
            try:
                day_occ = self.occ[day_start].get_occ(hr_start, hr_end)
            except KeyError:
                day_start += delta
                continue

            occ.append([day_start, day_occ])

            day_start += delta

        return pd.DataFrame(occ, columns=['date', 'occ'])
        

    def get_occ_h(
        self, 
        day_start: datetime.datetime.date = datetime.datetime(1970,1,1).date(),
        day_end: datetime.datetime.date = datetime.datetime(2099,12,31).date(),
        hr_start: int = 0,
        hr_end: int = 24
    ):
        assert day_start <= day_end, 'day_start value cannot be bigger than day_end value'

        if min(self.occ.keys()) > day_end or max(self.occ.keys()) < day_start:
            return pd.DataFrame()

        day_start = max(day_start, min(self.occ.keys()))
        day_end = min(day_end, max(self.occ.keys()))
        delta = datetime.timedelta(days=1)

        occ_h = []
        while day_start <= day_end:
            try:
                day_occ_h = self.occ[day_start].get_occ_h(hr_start, hr_end).reset_index().rename(columns={'index':'hour'})
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

    def read_transac_data(self, trsc_df: pd.DataFrame,
        trsc_conf: dict = {'No_Place':'spot_id', 'DH_Debut_Prise_Place':'start', 'DH_Fin_Prise_Place':'end', 'No_Trans_Src':'id_trsc'}):
        """TODO
        """

        trsc_df = trsc_df.copy()
        trsc_df.rename(columns=trsc_conf, inplace=True)
        trsc_df.start = pd.to_datetime(trsc_df.start)
        trsc_df.end = pd.to_datetime(trsc_df.end)

        # cut transactions per days
        res = []
        for idx, row in trsc_df[trsc_df.end.dt.day != trsc_df.start.dt.day].iterrows():
            start_date = row.start
            end_date = row.end
            delta = datetime.timedelta(days=1)
            # first day to n-1 day
            while start_date.date() < end_date.date():
                row_cp = row.copy()
                row_cp['start'] = start_date
                row_cp['end'] = pd.to_datetime(start_date.date()) + datetime.timedelta(hours=23, minutes=59, seconds=59)
                res.append(row_cp)

                start_date = pd.to_datetime(start_date.date())
                start_date += delta 
            # last day 
            row['start'] = start_date
            res.append(row)

        adjusted_time = pd.DataFrame(res)
        trsc_df = pd.concat([trsc_df[trsc_df.end.dt.day == trsc_df.start.dt.day], adjusted_time])

        tsorted = trsc_df.sort_values(by='start')

        for _,row in tsorted.iterrows():
            self.populate_trsc(row.spot_id, row.start, row.end, row.id_trsc)

    def populate_trsc(self, spot_id: str, start: datetime.datetime, end: datetime.datetime, id_trsc: np.int64):
        if spot_id not in self.spots:
            self.spots[spot_id] = self.create_spot(spot_id)

        self.spots[spot_id].add_trsc_obs(start, end, id_trsc)

    def read_reglement_data(self, regl_df: pd.DataFrame,
        regl_conf: dict = {
            'No_Place_Terrain':'spot',
            'Code_Regl':'regl_id',
            'Priorite_Regl':'priority', 
            'Deb_Mois_Regl': 'start_month',
            'Deb_Jours_Regl': 'start_day', 
            'Fin_Mois_Regl': 'end_month', 
            'Fin_Jours_Regl': 'end_day', 
            'Hr_Deb_Regl': 'start_hr',
            'Hr_Fin_Regl': 'end_hr', 
            'Ind_Interdiction': 'regl_type', 
            'Ind_Lun': 'mon', 
            'Ind_Mar': 'tue', 
            'Ind_Mer': 'wed',
            'Ind_Jeu': 'thu', 
            'Ind_Ven': 'fri', 
            'Ind_Sam': 'sat', 
            'Ind_Dim': 'sun', 
            'MD_Dt_Effectif': 'valid_from',
            'MD_Dt_Expir': 'valid_to'
        },
        reg_type_conf: dict = {'Permis':1, 'Interdit':2, 'Oui':True, 'Non':False}):
        """TODO
        """
        
        regl_df = regl_df.copy()
        regl_df = regl_df.rename(columns=regl_conf)
        regl_df = regl_df.replace('2999-12-31', '2099-12-31')
        regl_df.valid_from = pd.to_datetime(regl_df.valid_from)
        regl_df.valid_to = pd.to_datetime(regl_df.valid_to)
        regl_df = regl_df.replace(reg_type_conf)

        rsorted = regl_df.sort_values(['priority'])
        for _, row in rsorted.iterrows():
            self.populate_regl(row['spot'], row['regl_type'], row.to_dict())

    def populate_regl(self, spot_id: str, regl_type: int, regl: dict):
        if spot_id not in self.spots:
            self.spots[spot_id] = self.create_spot(spot_id)

        self.spots[spot_id].add_regl(regl_type, regl)
        
    def read_permits_data(self, perm_df: pd.DataFrame,
        perm_conf: dict = {
            'No_Place': 'spot_id',
            'Dt_Permis': 'date',
            'DH_Deb_Permis': 'start',
            'DH_Fin_Permis': 'end'
        }):
        """TODO
        """
        
        perm_df = perm_df.copy()
        perm_df = perm_df.rename(columns=perm_conf)
        perm_df.start = pd.to_datetime(perm_df.start)
        perm_df.end = pd.to_datetime(perm_df.end)

        psorted = perm_df.sort_values(['start'])
        for _, row in psorted.iterrows():
            self.populate_perm(row['spot_id'], row['start'], row['end'])

    def populate_perm(self, spot_id: str, start: datetime.datetime, end: datetime.datetime):
        if spot_id not in self.spots:
            self.spots[spot_id] = self.create_spot(spot_id)

        self.spots[spot_id].add_permits(start, end)
        

    def create_spot(self, spot_id):
        return SingleParkingSpot(spot_id)

    @property
    def occ_days(self):
        return np.unique(list(itertools.chain(*[spot.occ.keys() for _, spot in self.spots.items()])))

    def get_match_obs(
        self,
        day_start: datetime.datetime.date = datetime.datetime(1970, 1, 1).date(),
        day_end: datetime.datetime.date = datetime.datetime(2099, 12, 31).date(),
        hr_start: int = 0,
        hr_end: int = 24,
        filter_spot: list = []
    ) -> pd.DataFrame:
        """TODO
        """
        raw = []
        for spot_name, spot in self.spots.items():
            # filter
            if filter_spot and spot_name not in filter_spot:
                continue

            matched_obs = spot.get_matched_obs(day_start=day_start, day_end=day_end,
                hr_start=hr_start, hr_end=hr_end)
            matched_obs['spot'] = spot_name
            raw.append(matched_obs)

        data = pd.concat(raw)
        data.date = data.date.astype(np.datetime64)
        data = data[["spot", "event_id", "date", "start", "end", "observation",	"has_permits",
                     "regulated", "prohibited", "unknown", "has_been_paid", "id_trsc",
                     "has_paid", "first_trsc", "sec_paid"]] 

        return data

    def get_obs(
        self,
        day_start: datetime.datetime.date = datetime.datetime(1970, 1, 1).date(),
        day_end: datetime.datetime.date = datetime.datetime(2099, 12, 31).date(),
        hr_start: int = 0,
        hr_end: int = 24,
        filter_spot: list = []
    ) -> pd.DataFrame:
        """ TODO
        """
        raw = []
        for spot_name, spot in self.spots.items():
            # filter
            if filter_spot and spot_name not in filter_spot:
                continue

            spot_obs = spot.get_obs(day_start=day_start, day_end=day_end,
                hr_start=hr_start, hr_end=hr_end)
            spot_obs['spot'] = spot_name
            raw.append(spot_obs)

        data = pd.concat(raw).reindex(columns=['spot', 'date', 'start', 'end', 'observation']) 
        data.date = data.date.astype(np.datetime64)

        return data

    def get_occ(
        self,
        day_start: datetime.datetime.date = datetime.datetime(1970, 1, 1).date(),
        day_end: datetime.datetime.date = datetime.datetime(2099, 12, 31).date(),
        hr_start: int = 0,
        hr_end: int = 24,
        filter_spot: list = []
    ) -> pd.DataFrame:
        """TODO
        """
        occ = []
        for spot_name, spot in self.spots.items():
            # filter
            if filter_spot and (spot_name not in filter_spot):
                continue

            spot_occ = spot.get_occ(day_start=day_start, day_end=day_end,
                hr_start=hr_start, hr_end=hr_end)
            spot_occ['spot'] = spot_name
            occ.append(spot_occ)

        data = pd.concat(occ).reindex(columns=['spot', 'date', 'occ'])
        data.date = data.date.astype(np.datetime64)

        return data

    def get_occ_h(
        self,
        day_start: datetime.datetime.date = datetime.datetime(1970, 1, 1).date(),
        day_end: datetime.datetime.date = datetime.datetime(2099, 12, 31).date(),
        hr_start: int = 0,
        hr_end: int = 24,
        filter_spot: list = []
    ) -> pd.DataFrame:
        """TODO: faire passer les heures dans la fonction : sinon toute la journée par défaut.
        """
        occ_h = []
        for spot_name, spot in self.spots.items():
            # filter
            if filter_spot and spot_name not in filter_spot:
                continue

            spot_occ = spot.get_occ_h(day_start=day_start, day_end=day_end,
                hr_start=hr_start, hr_end=hr_end)
            spot_occ['spot'] = spot_name
            occ_h.append(spot_occ)

        data = pd.concat(occ_h).reindex(columns=['spot', 'date', 'hour', 'occ'])
        data.date = data.date.astype(np.datetime64)

        return data

    def get_day_obs(
        self,
        day: datetime.datetime.date,
        hr_start: int = 0,
        hr_end: int = 24,
        filter_spot: list = []
    ) -> pd.DataFrame:
        """ TODO
        """
        return self.get_obs(day_start=day, day_end=day, filter_spot=filter_spot,
                            hr_start=hr_start, hr_end=hr_end)

    def get_day_occ(
        self,
        day: datetime.datetime.date,
        hr_start: int = 0,
        hr_end: int = 24,
        filter_spot: list = []
    ) -> pd.DataFrame:
        """ TODO
        """
        return self.get_occ(day_start=day, day_end=day, filter_spot=filter_spot,
                            hr_start=hr_start, hr_end=hr_end)

    def get_day_occ_h(
        self,
        day: datetime.datetime.date,
        hr_start: int = 0,
        hr_end: int = 24,
        filter_spot: list = []
    ) -> pd.DataFrame:
        """ TODO
        """
        return self.get_occ_h(day_start=day, day_end=day, filter_spot=filter_spot,
                              hr_start=hr_start, hr_end=hr_end)



if __name__ == '__main__':

    #import raw data
    capteurs = pd.read_csv('./preprocessing/output/donnees_capteurs_7jul22.csv')
    cols_capteurs = {'DH_Date_Observation': 'timestamp', 'No_Place':'spot_id', 'Valeur_Observee':'obs'}
    capteurs_value = {'Occupé':1, 'Libre':0, 'Inconnu':-2}
    trans = pd.read_csv('./preprocessing/output/donnees_transac_7jul22.csv')
    regls = pd.read_csv('./preprocessing/output/donnees_reglements_3oct22.csv')
    perms = pd.read_csv('./preprocessing/output/donnees_odp_permits_7jul22.csv')

    #transform the dates
    capteurs.loc[:,'DH_Date_Observation'] = pd.to_datetime(capteurs.DH_Date_Observation)
    capteurs = capteurs.sort_values('DH_Date_Observation')


    parkings = ParkingSpotCollection()
    parkings.read_ping_data(capteurs, columns_conf=cols_capteurs, obs_conf=capteurs_value)
    parkings.read_transac_data(trans)
    parkings.read_reglement_data(regls)
    parkings.read_permits_data(perms)

    matched_obs = parkings.get_match_obs()
    matched_obs.to_csv('matched_obs_v4.csv', index=False)

    capteurs_obs = parkings.get_day_obs(
        day=datetime.datetime(2021, 6, 16).date(),
        filter_spot=['RB383', 'RB351']
    )
    print('Observations :\n', capteurs_obs.head(5))

    capteurs_occ_moy = parkings.get_day_occ(
        day=datetime.datetime(2021, 6, 16).date(),
        filter_spot=['RB383', 'RB351']
    ) 
    print('Occupation moyenne :\n', capteurs_occ_moy.head(5))
    
    capteurs_occ_h = parkings.get_day_occ_h(
        day=datetime.datetime(2021, 6, 16).date(),
        filter_spot=['RB383', 'RB351']
    ) 
    print('Ocupation horraire :\n', capteurs_occ_h.head(5))