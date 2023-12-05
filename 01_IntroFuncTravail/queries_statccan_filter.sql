SELECT geo_code,
       theme,
       caracteristique,
       --note,
       total,
       --flag_total,
       sexe_masculin,
       --flag_sexe_masculin--,
       sexe_feminin--,
       --flag_sexe_feminin
  FROM StatCanadaPopulationData_csv
  WHERE
  length(trim(sexe_masculin)) >0
  AND length(trim(sexe_feminin)) >0
  --AND upper(theme) not like upper('%langue%')
  AND upper(theme) like upper('%selon%')
  and length(trim(caracteristique))>6 --eliminer 15 ans, 16 ans... 19 ans
  and geo_code = 'A0A' 
;
  
/*24*/
select distinct 
note
from 
StatCanadaPopulationData_csv;

/*12694*/
select distinct 
total
from 
StatCanadaPopulationData_csv;

/*3*/
select distinct 
flag_total
from 
StatCanadaPopulationData_csv;

/*5710*/
select distinct 
sexe_masculin
from 
StatCanadaPopulationData_csv;

/*10*/
select distinct 
theme
from 
StatCanadaPopulationData_csv;

/*252*/
select distinct 
caracteristique
from 
StatCanadaPopulationData_csv;

/*24*/
select distinct 
note
from 
StatCanadaPopulationData_csv;

/*12694*/
select distinct 
total
from 
StatCanadaPopulationData_csv;

/*5710*/
select distinct 
sexe_masculin
from 
StatCanadaPopulationData_csv;

/*5796*/
select distinct 
sexe_feminin
from 
StatCanadaPopulationData_csv;

/*4*/
select distinct 
flag_sexe_masculin
from 
StatCanadaPopulationData_csv;

/*4*/
select distinct 
flag_sexe_feminin
from 
StatCanadaPopulationData_csv;
