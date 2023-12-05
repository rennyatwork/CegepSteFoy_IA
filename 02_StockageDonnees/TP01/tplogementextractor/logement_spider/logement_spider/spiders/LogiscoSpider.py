import scrapy
import json
from pymongo import MongoClient

class LogiscoSpider(scrapy.Spider):
    name = 'Logisco'    
    start_urls = ['https://logisco.com/appartements-a-louer']
    headers = {
        "Accept": "application/json, text/plain, */*",
        "Accept-Encoding": "gzip, deflate, br",
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": "https://logisco.com/appartements-a-louer",
        "Sec-Fetch-Mode": "cors",
        "Sec-Fetch-Site": "same-origin",
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/102.0.0.0 Safari/537.36"
        }

    def parse(self, response):        
        url = 'https://logisco.com/api/projets?type=1'                      
        
        yield scrapy.Request(url, 
            callback=self.parse_api, 
            headers=self.headers)        
    
    def parse_api(self, response):
        raw_data = response.body
        data = json.loads(raw_data)
        self.sauvegarde_json(data)
      
        
    def sauvegarde_json(self,data):          
        with open('scrapping_Logisco.json', 'w', encoding='UTF-8') as outfile:
            json.dump(data, outfile, ensure_ascii=False, indent=2)
        self.sauvegarde_db()     
    
    def sauvegarde_db(self):
             client = MongoClient('localhost', 27017)
             db = client['logement_db']
             collection_logement = db['Logisco']

             with open('scrapping_Logisco.json', encoding='UTF-8') as f:
                file_data = json.load(f)
                
             collection_logement.insert_many(file_data)    
             client.close()
    