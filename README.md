# PSIV AUTOENCODER


## INTRODUCCIÓ

La funció principal d'un a classificador és assignar etiquetes o categories a noves imatges en funció de les característiques que ha après durant l'entrenament.El procés d'entrenament d'un classificador d'imatges generalment implica alimentar el sistema amb un conjunt de dades d'imatges prèviament etiquetades. El model d'IA utilitza aquestes imatges per aprendre patrons i característiques que són distintives de cada classe o categoria. Un cop entrenat, el classificador pot fer prediccions sobre noves imatges, assignant-los una etiqueta o categoria basada en el seu coneixement previ. La classificació d’imatges pot quedar esbiaxat si  les classes no tenen un número de imatges balancejat. Fent que la red neuronal sigui incapaç d’aprendre o de classificar amb la precisió esperada. 

Per resoldre aquest problema clasificarem per contrast, és a dir que no aprendrem a classificar directament sino que d'una imatge farem una funció de contrast per imatge que en el nostre cas serà les diferencies entre la imatge original i la imatge generada per un autoencoder esbiaxat cap a una de les classes. Creant diferents embeddings pels tipus de classes  una classe la farà molt bé i l’altre no perque mai haurà rebut com entrenament una imatge d’aquella classe.  Mirant les diferències d’error i com han quedat les construccions de l’autoencoder podem classificar millor les classes obtingudes ja que una classe quedarà ven reconstruida i l'altre no. 

## OBJECTIU
L’objectiu d'aquesta práctica és la detecció del  Helicobacter pylori, en cél·lules de teixit humà. On per cada pacient hi ha una varietat de mostres de zones diferents del seu cos. Degut a que una detecció ràpida del Helicobacter pylori pot ser tractat fàcilment evitant ha llarg termini la aparició de càncer en el teixit. Com la majoria de pacients que tenen el Helicobacter pylori són asimptomàtics es detecta agafant mostres de teixit i mirant si hi ha presencia de Helicobacter pylori.

## DISTRIBUCIÓ
El GitHub l'hem distribuït de la següent manera:
  - directori modelos: anem guardant els diferents models del autoencoder que anem creant
  - autoencoder.py: arxiu on creem l'autoencoder i fem el seu entrenament.
  - segona_part.py: arxiu on es classifica si les imatges si tenen o no el elicobacter pylori.
  - imatges originals: carpeta que conté les imatges originals que tenen una construcció equivalent
  - imatges reconstruides: carpeta que conté less imatges reconstruides pel nostre autoencoder
  -  ... hay mas cosas mirar como quedara al final 

  
## BASE DE DADES
La base de dades consisteix en una carpeta on hi ha un munt de carpetes on cada carpeta conté imatges d'un mateix pacient, dins de cada carpeta de pacient hi ha imatges de les seves mostres de teixit. Aquestes imatges venen etiquetades per la densitat de  Helicobacter pylori amb les següents etiquetes de baixa, alta i negativa.  En una carpeta apart hi han les mostres de teixit que contenen el Helicobacter pylori seperat en carpetes de pacient.  

## PROCEDIMENT


## AUTOENCODER
Aquesta part consisteix en la creació de l'autoencoder que agafarà mostres de teixit de pacients sans per entrenar. La reconstrucció de teixit sa sigui bona i amb pocs errors i que quan el model recontriueixi una imatge en presencia del Helicobacter pylori, no la pogui recontruir bé. Amb aquest error de reconstrucció podem diferenciar les imatges amb el Helicobacter pylori i sense ell. La creació autoencoder segueix la següent de metodología, quedant la següent arquitectura del coder i decoder:


### Coder

Convolucional - 2D(3, 32, 3, stride=1, padding=1)
ReLu 
MaxPool:  (2, stride=2, padding=1)
Convolucional - 2D (32, 64, 3, stride=1, padding=1)
ReLu
MaxPool (2, stride=2, padding=1)
Convolucional - 2D (64, 128, 3, stride=1, padding=1)
ReLu
MaxPool (2, stride=2, padding=1)
Convolucional - 2D (128, 256, 3, stride=1, padding=1)
ReLu
MaxPool (2, stride=2, padding=1)


### Decoder

Convulocional Transposada (256, 128, 3, stride=2, padding=1) 
ReLu
Convulocional Transposada (128, 64, 3, stride=2, padding=1)
ReLu
Convulocional Transposada (64, 32, 3, stride=2, padding=1
ReLu
Convulocional Transposada (32, 3, 3, stride=2, padding=1)









On s’han anat probant diferents paràmetres en la xarxa modificant el seu número de neurones i learning rate, ajustant segons els resutltats de la loss.Calculant la loss amb el MSE i comparant la imatge generada del autoencoder amb la original. També s’ha anat el criterion i el número de époques per seguir perfilant l'obtenció d’una gràfica de loss adequada.
 


RESULTATS

CONCLUSIÓ





