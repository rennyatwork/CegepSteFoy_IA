SELECT An_Debut_Invalidite
       ,Mois_Debut_Invalidite
       ,Duree_Delai_Attente
       ,FSA
       ,Sexe
       ,Annee_Naissance
       ,Code_Emploi
       ,Description_Invalidite
       ,Salaire_Annuel
       ,Duree_Invalidite
       , ( An_debut_invalidite - Annee_Naissance) as Age_debut_Invalid
       , case 
           when Mois_Debut_Invalidite <=3
        then 'hiver'
           when Mois_Debut_Invalidite <=6
        then 'printemps'
        when Mois_Debut_Invalidite <=9
            then 'ete'
        else
             'automne'
     end as saison_invalid
     
    , case 
     when   FSA like '_0_'
         then 1
     else
         0
     end as est_zone_rurale
     
    , case
    when salaire_annuel <50000
        then 'moins_50000' 
    when salaire_annuel>=50000 and salaire_annuel <75000
        then '50000_a_75000'
    when salaire_annuel>=75000 and salaire_annuel <100000
        then '75000_a_100000'
    when salaire_annuel>=100000 
        then 'plus_100000'
    end as categ_salaire
    , case 
    when duree_invalidite <180
        then 1
    else
        0
    end as est_invalid_courte 
    
    , case 
    when duree_delai_attente <14
        then 'moins_14'
    when duree_delai_attente >=14 and duree_delai_attente <30
        then '14_a_30'    
    when duree_delai_attente >=30  and duree_delai_attente <60    
        then '30_a_60'
    when duree_delai_attente >=60 and duree_delai_attente <90
        then '60_a_90'    
    when duree_delai_attente >=90 and duree_delai_attente <120
        then '90_a_120'
    when duree_delai_attente >=120
        then 'plus_120'
    end as categ_delai_attente  
    


    , case
    when upper(description_invalidite) like upper ('%cancer%')
        then 1
    else
        0
    end as est_cancer
    

   
    
, case   
    when upper(trim(description_invalidite)) like '%depression%'
        then 1
    else
        0
    end as est_depression
    
     , case   
    when upper(description_invalidite) like upper ('%major depressive disorder%')
        then 1
    else
        0
    end as est_major_depressive
    
    , case   
    when upper(description_invalidite) like upper ('%see file%')
        then 1
    else
        0
    end as est_see_file
    
     , case   
    when upper(description_invalidite) like upper ('%coronary%disease%')
        then 1
    else
        0
    end as est_coronary_disease
    
  , case   
    when upper(description_invalidite) like upper ('%back%pain%')
        then 1
    else
        0
    end as est_back_pain
    
, case   
    when upper(description_invalidite) like upper ('%arthritis%')
        then 1
    else
        0
    end as est_arthritis
    
, case   
    when upper(description_invalidite) like upper ('%disorder%')
        then 1
    else
        0
    end as est_disorder
      
  FROM assurance;
