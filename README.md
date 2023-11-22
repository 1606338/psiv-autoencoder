# Repte 3: Histologia Digital

## INTRODUCCIÓ
La funció principal d'un a classificador és assignar etiquetes o categories a noves imatges en funció de les característiques que ha après durant l'entrenament. El procés d'entrenament d'un classificador d'imatges generalment implica alimentar el sistema amb un conjunt de dades d'imatges prèviament etiquetades. El model d'IA utilitza aquestes imatges per aprendre patrons i característiques que són distintives de cada classe o categoria. Un cop entrenat, el classificador pot fer prediccions sobre noves imatges, assignant-los una etiqueta o categoria basada en el seu coneixement previ. La classificació d’imatges pot quedar esbiaxat si  les classes no tenen un número de imatges balancejat. Fent que la red neuronal sigui incapaç d’aprendre o de classificar amb la precisió esperada. 

Per resoldre aquest problema clasificarem per contrast, és a dir que no aprendrem a classificar directament sino que d'una imatge farem una funció de contrast per imatge que en el nostre cas serà les diferencies entre la imatge original i la imatge generada per un autoencoder esbiaxat cap a una de les classes. Creant diferents embeddings pels tipus de classes  una classe la farà molt bé i l’altre no perque mai haurà rebut com entrenament una imatge d’aquella classe.  Mirant les diferències d’error i com han quedat les construccions de l’autoencoder podem classificar millor les classes obtingudes ja que una classe quedarà ven reconstruida i l'altre no. 


## OBJECTIUS

L’objectiu d'aquesta práctica és la detecció del  Helicobacter pylori, en cél·lules de teixit humà. On per cada pacient hi ha una varietat de mostres de zones diferents del seu cos. Degut a que una detecció ràpida del Helicobacter pylori pot ser tractat fàcilment evitant ha llarg termini la aparició de càncer en el teixit. Com la majoria de pacients que tenen el Helicobacter pylori són asimptomàtics es detecta agafant mostres de teixit i mirant si hi ha presencia de Helicobacter pylori.

## DISTRIBUCIÓ

El GitHub l'hem distribuït de la següent manera:
  - directori modelos: anem guardant els diferents models del autoencoder que anem creant
  - autoencoder.py: arxiu on creem l'autoencoder i fem el seu entrenament.
  - segona_part.py: arxiu on es classifica si les imatges tenen o no el elicobacter pylori.
  - imatges_originals_train: carpeta que conté les imatges originals que utilitzem com a train que no tenen presència del elicobacter
  - imatges_reconstruides_train: carpeta que conté les imatges reconstruides del train pel nostre autoencoder
  - imatges_originals_test: carpeta que conté les imatges originals que utilitzem com a test que no tenen presència dle elicobacter
  - imatges_reconstruides_test: carpeta que conté les imatges reconstruides del train pel nostre autoencoder
  - Gràfiques: carpeta que conté les gràfiques de les losses tant del train com del test




## BASE DE DADES
La base de dades que utilitzem a estat una que se'ns ha prporcionat en el Campus Virtual. La base de dades està distruibuida de la següent manera:
Una crapeta que està fromada per moltes altres carpetes, on cadascuna fa referència a un pacient i conté les imatges de les mostres del teixit. Aquestes imatges venen etiquetades per la densitat de  Helicobacter pylori: baixa, alta i negativa.  
En el nostre cas hem decidit separar en train i test, per tal de poder comprovar que el autoencoder que fem funciona correctament i és robust. 
Pel train hem agafat __ carpetes i pel test hem agafat __ carpetes.

## PROCEDIMENT 

### PRIMERA PART: AUTOENCODER

Aquesta part consisteix en la creació de l'autoencoder que agafarà mostres de teixit de pacients sans per entrenar. La reconstrucció de teixit sans ha de ser bona i amb pocs errors, per tal que quan el model recontriueixi una imatge amb presència de l'Helicobacter pylori, no la pugui recontruir bé. Amb aquest error de reconstrucció a la segona part podrem diferenciar les imatges amb el Helicobacter pylori i sense ell i classificar-les. Però ara tornant a la primera part, el que hem fet per obtenir una xarxa neuronal correcte i robusta ha estat el següent:
Primer de tot disminuïr la mida de les imatges, és a dir, les hem convertit de 256x256x3 a 64x64x3.
Tot seguit amb molt de proba i error, hem modificat i probat molts paràmetres de la xarxa, com ara canviar el número de nuerones per cada capa, posar mé so menys capes, el learning rate també l'hem anat canviant, i ajustant segons els resultats de les losses que ens donaven, tant train com test, i ajustant també segons la loss amb el MSE i comparant la imatge generada del autoencoder amb la original. També s’ha anat canvinat el criterion i el número de époques per seguir perfilant l'obtenció d’una gràfica de loss adequada. Aquesta prova de parametres de la red s'ha anat combinant amb diferents criterion, epoques i learning rate. 


La creació autoencoder segueix la següent de metodología, quedant la següent arquitectura del coder i decoder:

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

### Parametres
- Adamax
- lr = 0.001
- epoques = 200

Entre les imatges generades per l'autoencoder i les orignals calculem la seva diferencia i amb MSE i aixì obtenim la nostre loss. Mostrem la loss del nostre train i del test veiem si el nostre autoencoder generalitza bé per generar parts de teixit sa. Creant el model sense la capacitat de generar el Helicobacter pylori. Ha train tenim 12 carpetes pel train 
acabar esto cuando acabe primera parte **********************************
 


### SEGONA PART: CLASSIFICACIÓ

Després de entrenar l'autoencoder, hem de fer el classificador per aquest procés hem decidit fer boxplot de les loss, per les dues classes per veure la diferencia que hi ha habia entre ells, ja que la loss de les que tenen el Helicobacter pylori ha de ser més gran que que les que no el tenen. Al visualitzar els boxplots veiem que no hi ha molta diferencia entre ells ja que la part del bacteri és molt reduida i no marca la diferencia amb les altres parts de color blau. Per tant per veure la diferencia només tractarem amb el canal vermell ja que el teixit sa no té presencia d'aquest color i el autoencoder no ha rebut cap imatge amb vermell per tan no el pot generar. Al calcular la loss només amb el canal vermell podem observar que el resultat dels boxplots és significatiu, ja que les parts sanen no tenen error i les parts infectades si. 
*** seguir cuando se acabe esta parte para eplicar que e hara i como e hara


## RESULTATS

## CONCLUSIONS



