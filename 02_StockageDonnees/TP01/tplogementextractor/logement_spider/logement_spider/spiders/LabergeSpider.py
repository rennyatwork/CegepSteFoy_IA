import scrapy
import pandas as pd
from pymongo import MongoClient
import json

class LabergeSpider(scrapy.Spider):
    name = 'Laberge'
    allowed_domains = ['laberge.qc.ca']    
    start_urls = ['https://laberge.qc.ca/recherche?page=1']

    def parse(self, response):        
        response = response.replace(body=response.body.replace(b'<br>', b'\n'))
        appartement = []                    
        for appartements in response.css('div.infos') :                                        
            rows = appartements.css('tr.clickable-row')
            data = []            
            for row in rows:                
                data.append([Data(row.css('td::text')[0].get().strip(), 
                             row.css('td::text')[1].get().strip(), 
                             row.css('td::text')[2].get().strip(), 
                             row.css('td::text')[3].get().strip(), 
                             "http://laberge.qc.ca" + row.attrib['data-href'].strip())])     
                
            appartement.append([Appartement(appartements.css('h3.mr-4::text').get(),
                                appartements.css('p::text').get().replace('\n','').strip().split("                    ")[0],
                                appartements.css('p::text').get().replace('\n','').strip().split("                    ")[1].split(',')[0],
                                appartements.css('p::text').get().replace('\n','').strip().split("                    ")[1].split(',')[1].strip(),
                                "http://laberge.qc.ca" + appartements.css('a.title-link').attrib['href'].strip(),
                                data)])            
            
        df = pd.DataFrame(appartement)
        self.sauvegarde_json(df)
            
            
    def sauvegarde_json(self,df):
        jsonFile = df.to_json (orient='records')
        with open('scrapping_Laberge.json', 'w', encoding='UTF-8') as outfile:
             json.dump(json.loads(jsonFile), outfile, ensure_ascii=False, indent=2)             
        self.sauvegarde_db()
             
    def sauvegarde_db(self):
             client = MongoClient('localhost', 27017)
             db = client['logement_db']
             collection_logement = db['laberge']

             with open('scrapping_Laberge.json', encoding='UTF-8') as f:
                file_data = json.load(f)
                
             collection_logement.insert_many(file_data)    
             client.close()
             
class Appartement:
    def __init__(self, nom, adresse, ville, codePostal, lien, appartements):
        self.nom, self.adresse, self.ville, self.codePostal, self.lien, self.appartements = nom, adresse, ville, codePostal, lien, appartements


class Data:
    def __init__(self, typeApp, prix, etage, disponibilite, lien):
        self.typeApp, self.prix, self.etage, self.disponibilite, self.lien = typeApp, prix, etage, disponibilite, lien