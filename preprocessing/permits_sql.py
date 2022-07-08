
###################################################################
# Retrieve all effective permits duration during a day (by hour). #
###################################################################

SQL_ODP = """
    drop table if exists ##ODPTmp; 
    drop table if exists ##ODPAgg; 

    -- Get active permits during a day.
    with PermisODP as (
        SELECT    
                 No_Place_ODP,
                 max(No_permis)    as DernierDebute
                 
        FROM     [dbo].[D_PermisODP] P
        
        WHERE    1=1
        AND            P.Statut_Permis   IN ('Actif', 'M')            
        AND            P.Type_Permis_ODP in ( 'Capuchonner', 'Enlever') 
        AND            P.SK_D_Permis_ODP > 0
        AND (
        '{date}' BETWEEN cast(P.DH_Deb_Permis as date) AND cast(P.DH_Fin_Permis as date)
        )
        
        GROUP BY No_Place_ODP
        )


    -- Transform start/end duration into hour bucket columns.

    SELECT    P.No_Place_ODP																																as No_Place
    ,         '{date}'																																		as Dt_Permis
    ,         CASE WHEN CAST(P.DH_Deb_Permis as DATE) < '{date}'  THEN CAST('{date}' as datetime) ELSE P.DH_Deb_Permis END                                  as DH_Deb_Permis
    ,         CASE WHEN CAST(P.DH_Fin_Permis as DATE) > '{date}'  THEN CAST(CONCAT('{date}',' 23:59:59') as datetime) ELSE P.DH_Fin_Permis END              as DH_Fin_Permis
    FROM        [dbo].[D_PermisODP]  P
    inner join    PermisODP             F on P.No_Place_ODP = F.No_Place_ODP and P.No_permis = F.DernierDebute     
    WHERE          1=1
    {whr_cond}

"""

PLACE_COND_ODP = """
    AND            P.Statut_Permis   IN ('Actif', 'M')            
    AND            P.Type_Permis_ODP in ( 'Capuchonner', 'Enlever') 
    AND            P.SK_D_Permis_ODP > 0
    AND            P.No_Place_ODP IN {place_filter}
"""

DATE_COND_ODP = """
     '{date}' BETWEEN cast(P.DH_Deb_Permis as date) AND cast(P.DH_Fin_Permis as date)
"""
