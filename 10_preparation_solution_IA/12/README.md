# TP 01

Les fichiers dans ce répertoire correspondent à une version adaptée de l'item 12 d'ici:
https://onedrive.live.com/?authkey=%21AGG5d98Pf53QsUw&id=7E57B262F9A1DD9A%2173208&cid=7E57B262F9A1DD9A

## Structure

### packages/regression_model
Toute la structure du modèle predictif

### packages/ml_api
L'infrastructure Flask pour fournin un API pour intéragir avec le modèle

## Étapes suivi

1) create venv
2) add __ini__.py
3) replace requirements.txt with https://github.com/rennyatwork/CegepSteFoy_IA/blob/main/10_preparation_solution_IA/10/packages/regression_model/requirements.txt
4) run create__init__.sh
5) run install_requirements.sh
6) go to ./packages/regression_model/regression_model
7) run tox (it should pass)

ML 
go to packages/ml_api
run 
python3 run.py


PACKAGING
go to regression_model dir and run:

$ python3 setup.py sdist
