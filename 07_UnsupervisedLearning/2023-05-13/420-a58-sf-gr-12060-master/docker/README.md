# Image Docker (Jupyter Notebook)

### Création de l'image à partir du Dockerfile
`docker build -t <nom_image> .`

#### Exemple:
`docker build -t 420-a58-sf .`

### Exécution sur Linux
`docker run --rm -it -p 8888:8888 -v $(pwd):/notebooks --name <nom_du_conteneur> <nom_image>`

### Exécution sur Powershell
`docker run --rm -it -p 8888:8888 -v ${PWD}:/notebooks --name <nom_du_conteneur> <nom_image>`
