import sys
import json
from pymongo import MongoClient

client = MongoClient('localhost', 27017)
db = client['myDB']
var_collection = db['mycollection']

file_path="/home/hadoop/Documents/CegepSteFoy/IA/02-CollecteStockage/Cours-07/Partie-1/exercice-1-python-mongoDB/contacts.json"

with open(file_path) as f:
    #print("[f]: " + str(type(f)))
    file_data = json.load(f)
var_collection.insert_many(file_data)
client.close()