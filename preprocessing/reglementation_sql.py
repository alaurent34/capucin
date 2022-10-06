
##########################################################################################################
# Recovers all minutes where parking is allowed during the day (i.e. without active regulation) by hour. #
##########################################################################################################

SQL_REGLEMENT = '''
	drop table if exists ##EchantillonReglement; 
	drop table if exists ##ReglprioriteMin;  
	drop table if exists ##ReglementApplicable; 
    drop table if exists ##ReglTmp;

    declare @wday varchar(50)
    set @WDay = datepart(weekday, '{date}'); --Dimanche = 1, Lundi = 2, Mardi = 3, Mercredi = 4, Jeudi = 5, Vendredi = 6, Samedi = 7

    ----------------------------------------------------------------------------
    -- Création des règlementations actives
    ----------------------------------------------------------------------------
             
    With FilteredRegulation AS(
    SELECT * 
    FROM [dbo].[D_Reglement] I
    WHERE 1=1
    AND    '{date}'  between MD_Dt_Effectif and MD_Dt_Expir 
    AND    '{date}' between cast(cast(year('{date}') as Varchar(4)) +'-'+ right(DT_Deb_Regl, 2)+'-'+ left(DT_Deb_Regl, 2) as date) and cast(cast(year('{date}') as Varchar(4)) +'-'+ right(DT_Fin_Regl, 2)+'-'+ left(DT_Fin_Regl, 2) as date)
    AND Type_Regl NOT IN ('P', 'Q', 'M', 'D', 'V') -- Les types de règles qui sont des durées max
    AND case    when @WDay = 1 then Ind_Dim
                when @WDay = 2 then Ind_Lun
                when @WDay = 3 then Ind_Mar
                when @WDay = 4 then Ind_Mer
                when @WDay = 5 then Ind_Jeu
                when @WDay = 6 then Ind_Ven
                when @WDay = 7 then Ind_Sam
        end = 'Oui'    
    {whr_cond}
    ),
    Permis AS(           -- Tous les permis qui ne sont pas des durée maximum et sont actif pendant la journée en question
    SELECT    SK_D_Regl
    ,        Code_Regl
    ,        No_Place_Terrain
    ,        Type_Regl
    ,        Priorite_Regl
    ,        Ind_Interdiction
    ,        cast(cast('{date}' as Varchar(10)) + ' ' + cast(Hr_Deb_Regl as varchar(8)) as datetime) as DebutReglement
    ,        cast(cast('{date}' as Varchar(10)) + ' ' + cast(Hr_fin_Regl as varchar(8)) as datetime) as FinReglement
    ,         Ind_Lun
    ,        Ind_Mar
    ,        Ind_Mer
    ,        Ind_Jeu
    ,        Ind_Ven
    ,        Ind_Sam
    ,        Ind_Dim
    ,        Tarif_Hr
    ,        Mnt_Quot_Max
    ,        MD_Dt_Effectif
    ,        MD_Dt_Expir 
    FROM    FilteredRegulation
    WHERE 1=1
    AND Ind_Interdiction = 'Permis'
    ),
    Interdictions AS (    -- Toutes les interdictions qui recouvrent au moins une période d'un permis
    SELECT    I.SK_D_Regl
    ,        I.Code_Regl
    ,        I.No_Place_Terrain
    ,        I.Type_Regl
    ,        I.Priorite_Regl
    ,        I.Ind_Interdiction
    -- Les débuts et fin des interdictions sont tronquées à celle du permis
    ,        CASE WHEN cast(cast('{date}' as Varchar(10)) + ' ' + cast(I.Hr_Deb_Regl as varchar(8)) as datetime) < 
                        P.DebutReglement THEN dateadd(mi, datediff(mi, 0, dateadd(s, 30, P.DebutReglement)), 0)
                    ELSE dateadd(mi, datediff(mi, 0, dateadd(s, 30, cast(cast('{date}' as Varchar(10)) + ' ' + cast(I.Hr_Deb_Regl as varchar(8)) as datetime))), 0)
            END as DebutReglement
    ,        CASE WHEN cast(cast('{date}' as Varchar(10)) + ' ' + cast(I.Hr_fin_Regl as varchar(8)) as datetime) >=
                        P.FinReglement THEN  dateadd(mi, datediff(mi, 0, dateadd(s, 30, P.FinReglement)), 0)
                    ELSE dateadd(mi, datediff(mi, 0, dateadd(s, 30, cast(cast('{date}' as Varchar(10)) + ' ' + cast(I.Hr_fin_Regl as varchar(8)) as datetime))), 0)
            END as FinReglement
    ,         I.Ind_Lun
    ,        I.Ind_Mar
    ,        I.Ind_Mer
    ,        I.Ind_Jeu
    ,        I.Ind_Ven
    ,        I.Ind_Sam
    ,        I.Ind_Dim
    ,        I.Tarif_Hr
    ,        I.Mnt_Quot_Max
    ,        I.MD_Dt_Effectif
    ,        I.MD_Dt_Expir 
    FROM    FilteredRegulation  I
    LEFT JOIN Permis P ON P.No_Place_Terrain = I.No_Place_Terrain
    WHERE    I.Ind_Interdiction = 'Interdit'
        AND cast(cast('{date}' as Varchar(10)) + ' ' + cast(I.Hr_Deb_Regl as varchar(8)) as datetime) <
            P.FinReglement 
        AND P.DebutReglement <
            cast(cast('{date}' as Varchar(10)) + ' ' + cast(I.Hr_fin_Regl as varchar(8)) as datetime)
    ),
    Reglements as
    (
        SELECT    *
        from    Permis 
        UNION ALL
        SELECT    *
        from    Interdictions
    )
    select    *
    into    ##EchantillonReglement
    from    Reglements;
 
            
    print 'Reglement Echantillonage terminé :' + cast(getdate() as varchar(30));
    

    WITH MinPrioriteInterdit AS
    (
        SELECT No_Place_Terrain
        ,        MIN(Priorite_Regl) as Priorite_Regl_Min
        FROM    ##EchantillonReglement
        WHERE    Ind_Interdiction = 'Interdit'  
        group by 
                No_Place_Terrain
        
    )
    , MinPrioritePermis AS
    (
        SELECT No_Place_Terrain
        ,        DebutReglement 
        ,        FinReglement 
        ,        MIN(Priorite_Regl) as Priorite_Regl_Min
        FROM    ##EchantillonReglement
        WHERE    Ind_Interdiction = 'Permis'  
        group by 
                No_Place_Terrain
        ,        DebutReglement 
        ,        FinReglement 
        
    )
    , MinPriorite as
    (
        SELECT  DISTINCT No_Place_Terrain
        ,        Priorite_Regl_Min
        from    MinPrioriteInterdit 
        UNION ALL
        SELECT  DISTINCT  No_Place_Terrain
        ,        Priorite_Regl_Min
        from    MinPrioritePermis     
    )
    select    *
    into    ##ReglprioriteMin
    from    MinPriorite;

    --reglement terrain
    Select       R.No_Place_Terrain                                                                                 as No_Place
    ,            '{date}'                                                                                           as DateRegl
    ,            R.DebutReglement
    ,            R.FinReglement        
    ,            case when Ind_Interdiction = 'Permis' then 0 else 1 end                                            as Interdiction
    ,            'Terrain'                                                                                          as TypeReglement
    into        ##ReglementApplicable
    from        ##EchantillonReglement R
    inner join    ##ReglprioriteMin      P    on P.No_Place_Terrain = R.No_Place_Terrain and P.Priorite_Regl_Min = R.Priorite_Regl 
    where        R.No_Place_Terrain in (select No_place from dbo.D_Place where SK_D_Terrain > 0 and '{date}' between MD_Dt_Effectif AND MD_Dt_Expir);

    --reglement place
    insert into ##ReglementApplicable
    Select       R.No_Place_Terrain                                                                                         as No_Place
    ,            '{date}'                                                                                                   as DateRegl
    ,            R.DebutReglement
    ,            R.FinReglement        
    ,            case when Ind_Interdiction = 'Permis' then 0 else 1 end                                                    as Interdiction
    ,            'Place'                                                                                                    as TypeReglement
    from        ##EchantillonReglement R
    inner join    ##ReglprioriteMin      P    on P.No_Place_Terrain = R.No_Place_Terrain and P.Priorite_Regl_Min = R.Priorite_Regl 
    where        R.No_Place_Terrain not in (select No_place from dbo.D_Place where SK_D_Terrain > 0 and '{date}' between MD_Dt_Effectif AND MD_Dt_Expir);
            
                
select *
from ##ReglementApplicable;

	drop table if exists ##EchantillonReglement; 
	drop table if exists ##ReglprioriteMin;  
	drop table if exists ##ReglementApplicable; 
'''

PLACE_SQL_REG = '''
AND I.No_Place_Terrain in {place_filter}
'''