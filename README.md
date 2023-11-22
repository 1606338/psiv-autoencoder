# Repte 3: Histologia Digital

## INTRODUCCIÓ
La funció principal d'un a classificador és assignar etiquetes o categories a noves imatges en funció de les característiques que ha après durant l'entrenament. El procés d'entrenament d'un classificador d'imatges generalment implica alimentar el sistema amb un conjunt de dades d'imatges prèviament etiquetades. El model d'IA utilitza aquestes imatges per aprendre patrons i característiques que són distintives de cada classe o categoria. Un cop entrenat, el classificador pot fer prediccions sobre noves imatges, assignant-los una etiqueta o categoria basada en el seu coneixement previ. La classificació d'imatges pot quedar esbiaixat si les classes no tenen un nombre d'imatges balancejat. Fent que la xarxa neuronal sigui incapaç d'aprendre o de classificar amb la precisió esperada.


Per resoldre aquest problema classificarem per contrast, és a dir que no aprendrem a classificar directament sinó que d'una imatge farem una funció de contrast per imatge que en el nostre cas serà les diferències entre la imatge original i la imatge generada per un autoencoder esbiaixat cap a una de les classes. Creant diferents embeddings pels tipus de classes una classe la farà molt bé i l'altre no perquè mai haurà rebut com entrenament una imatge d'aquella classe. Mirant les diferències d'error i com han quedat les construccions de l'autoencoder podem classificar millor les classes obtingudes, ja que una classe quedaran ben reconstruïda i l'altre no.


## OBJECTIUS

L'objectiu d'aquesta pràctica és la detecció del Helicobacter pylori, en cèl·lules de teixit humà. On per cada pacient hi ha una varietat de mostres de zones diferents del seu cos. A causa del fet que una detecció ràpida del Helicobacter pylori pot ser tractat fàcilment evitant a llarg termini l'aparició de càncer en el teixit. Com la majoria dels pacients que tenen el Helicobacter pylori són asimptomàtics es detecta agafant mostres de teixit i mirant si hi ha presència d'Helicobacter pylori.

## DISTRIBUCIÓ

El GitHub l'hem distribuït de la següent manera:
  - directori modelos: anem guardant els diferents models de l'autoencoder que anem creant
  - autoencoder.py: arxiu on creem l'autoencoder i fem el seu entrenament.
  - segona_part.py: arxiu on es classifica si les imatges tenen o no el Helicobacter pylori.
  - imatges_originals_train: carpeta que conté les imatges originals que utilitzem com a train que no tenen presència del elicobacter
  - imatges_reconstruides_train: carpeta que conté les imatges reconstruides del train pel nostre autoencoder
  - imatges_originals_test: carpeta que conté les imatges originals que utilitzem com a test que no tenen presència de l'Helicobacter
  - imatges_reconstruides_test: carpeta que conté les imatges reconstruïdes del train pel nostre autoencoder
  - Gràfiques: carpeta que conté les gràfiques de les losses tant del train com del test
  - Grafiques.py: fitxter python que conté el codi on es generen les gràfiques de loss del train i test.




## BASE DE DADES
La base de dades que utilitzem han estat una que se'ns ha proporcionat en el Campus Virtual. La base de dades està distribuïda de la següent manera:
Una carpeta que està formada per moltes altres carpetes, on cadascuna fa referència a un pacient i conté les imatges de les mostres del teixit. Aquestes imatges venen etiquetades per la densitat d'Helicobacter pylori: baixa, alta i negativa.
Per la primera part hem separat en train test tots els pacients que no tenen presència de l'Helicobacter pylori. En el nostre cas hem decidit separar en train i test, per tal de poder comprovar que l'autoencoder que fem funciona correctament i és robust.
Pel train hem agafat 10,20,30,50 carpetes, és a dir, fem  4 models amb diferents valors de carpetes, però amb un test constant de 5 carpetes sempre.


## PROCEDIMENT 

### PRIMERA PART: AUTOENCODER

Aquesta part consisteix en la creació de l'autoencoder que agafarà mostres de teixit de pacients sa per entrenar. La reconstrucció de teixit sans ha de ser bona i amb pocs errors, per tal que quan el model reconstrueixi  una imatge amb presència de l'Helicobacter pylori, no la pugui reconstruir  bé. Amb aquest error de reconstrucció a la segona part podrem diferenciar les imatges amb el Helicobacter pylori i sense ell i classificar-les. Però ara tornant a la primera part, el que hem fet per obtenir una xarxa neuronal correcta i robusta ha estat el següent:
Primer de tot disminuir la mida de les imatges, és a dir, les hem convertit de 256x256x3 a 64x64x3.

Tot seguit amb molta prova i error, hem modificat i provat molts paràmetres de la xarxa, com ara canviar el nombre de neurones per cada capa, posar més o menys capes, el learning rate també l'hem anat canviant, i ajustant segons els resultats de les losses que ens donaven, tant train com test, i ajustant també segons la loss amb el MSE i comparant la imatge generada de l'autoencoder amb l'original. També s'ha anat canviat el criterion i el número d'époques per continuar perfilant l'obtenció d'una gràfica de loss adequada. Aquesta prova de paràmetres de la xarxa s'ha anat combinant amb diferents èpoques.


La creació autoencoder segueix la següent de metodologia, quedant la següent arquitectura de l'encoder i el decoder:

#### Encoder
L'encoder és responsable de transformar la imatge original a una imatge de baixa dimensionalitat. El nostre encoder consisteix en capes de convolució seguides de funcions d'activació ReLU i capes de max pooling. Cada capa de convolució aprèn a extreure característiques específiques de l'entrada. Les capes de max pooling redueixen progressivament les dimensions espacials, ajudant a crear una representació comprimida de l'entrada.
Hem decidit utilitzar les capes convolucionals 2d, ja que en el nostre cas ens va bé per reduir la dimensionalitat i per captar tots els possibles patrons que segueixen les imatges.
Per altra banda, hem decidit fer servir un núm. de filtres bastant reduït per tal d'assolir l'autoencoder esbiaixat.
Tant al padding com a l'stride els hi hem posat un valor d'1, ja que volem que es mantengui la dimensionalitat (padding) i no saltar-nos cap píxel (stride).

- Convolucional - 2D(3, 32, 3, stride=1, padding=1)
- ReLu 
- MaxPool:  (2, stride=2, padding=1)
- Convolucional - 2D (32, 64, 3, stride=1, padding=1)
- ReLu
- MaxPool (2, stride=2, padding=1)
- Convolucional - 2D (64, 128, 3, stride=1, padding=1)
- ReLu
- MaxPool (2, stride=2, padding=1)
- Convolucional - 2D (128, 256, 3, stride=1, padding=1)
- ReLu
- MaxPool (2, stride=2, padding=1)



#### Decoder

El decoder s'encarrega de generar amb la sortida de l'encoder la imatge original d'entrada d'aquest, és a dir, pren la representació de baixa dimensionalitat generada per l'encoder i la reconstrueix de nou a la forma original i que sigui el més semblant possible a l'entrada original.
Cada capa deconvolucional o convolució transposada aprèn a generar característiques que són inverses a les apreses per les capes corresponents de l'encoder. L'ús de funcions d'activació ReLU també ajuda en aquest procés

- Convulocional Transposada (256, 128, 3, stride=2, padding=1) 
- ReLu
- Convulocional Transposada (128, 64, 3, stride=2, padding=1)
- ReLu
- Convulocional Transposada (64, 32, 3, stride=2, padding=1
- ReLu
- Convulocional Transposada (32, 3, 3, stride=2, padding=1)

#### Paràmetres

Els paràmetres que hem utilitzat al final han estat els següents:
- Optimizer: Adamax
- Criterion: MSELoss
- lr = 0.001
- epoques = 200

#### Loss

Entre les imatges generades per l'autoencoder i les originals calculem la seva diferència i amb l'optimizer MSE, obtenim la loss, específicament una loss per les dades train i una loss per les dades test. Això ho guardem en un objecte pickle, per tal de poder manipular més endavant com vulguem.
El codi on generem els gràfics pertinents està al fitxer "grafiques.py".
Aquestes gràfiques ens serveixen per monitorar i veure que l'autoencoder és robust i correcte i a la vegada esbiaixat, per tal de no poder recrear la imatge amb el bacteri.

 
### SEGONA PART: CLASSIFICACIÓ

Després d'entrenar l'autoencoder, hem de fer el classificador; per aquest procés hem decidit fer boxplot de les loss, per les dues classes per veure la diferencia que hi ha havia entre ells, ja que la loss de les que tenen el Helicobacter pylori ha de ser més gran que que les que no el tenen. Al visualitzar els boxplots veiem que no hi ha molta diferencia entre ells ja que la part del bacteri és molt reduida i no marca la diferencia amb les altres parts de color blau. Per tant per veure la diferencia només tractarem amb el canal vermell ja que el teixit sa no té presencia d'aquest color i el autoencoder no ha rebut cap imatge amb vermell per tan no el pot generar. Al calcular la loss només amb el canal vermell podem observar que el resultat dels boxplots és significatiu, ja que les parts sanen no tenen error i les parts infectades si. 
*** seguir cuando se acabe esta parte para eplicar que e hara i como e hara


## RESULTATS

## CONCLUSIONS



