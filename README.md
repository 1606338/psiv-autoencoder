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
  - segona_part.py: arxiu on es classifica si les imatges tenen o no l'Helicobacter pylori.
  - imatges_originals_train: carpeta que conté algunes de les imatges originals que utilitzem com a train que no tenen presència de l'Helicobacter
  - imatges_reconstruides_train: carpeta que conté algunes de les imatges reconstruides del train pel nostre autoencoder
  - imatges_originals_test: carpeta que conté algunes de les imatges originals que utilitzem com a test que no tenen presència de l'Helicobacter
  - imatges_reconstruides_test: carpeta que conté algunes de les imatges reconstruïdes del train pel nostre autoencoder
  - Gràfiques: carpeta que conté les gràfiques de les losses tant de train com de test de tots 4 models
  - Pickle.py: fitxter python que conté el codi on es generen les gràfiques de loss del train i test.
  - Pickle: carpeta que conté els arxius pickle per fer les losses




## BASE DE DADES
La base de dades que utilitzem han estat una que se'ns ha proporcionat en el Campus Virtual. La base de dades està distribuïda de la següent manera:
Una carpeta que està formada per moltes altres carpetes, on cadascuna fa referència a un pacient i conté les imatges de les mostres del teixit. Aquestes imatges venen etiquetades per la densitat d'Helicobacter pylori: baixa, alta i negativa.
Per la primera part hem separat en train test tots els pacients que no tenen presència de l'Helicobacter pylori. En el nostre cas hem decidit separar en train i test, per tal de poder comprovar que l'autoencoder que fem funciona correctament i és robust.
Pel train hem agafat 10,20,30,50 carpetes, és a dir, fem  4 models amb diferents valors de carpetes, però amb un test constant de 5 carpetes sempre. Hem anat probant al principi amb pocs pacients i hem anat augmentant al anar fent proves per tant tenim 4 models de base de dades.


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

Després d'haver entrenat un autoencoder amb només pacients sans, utilitzarem aquest model per poder classificar pacients no sans, és a dir, positius; i sans, és a dir, negatius. Per fer-ho carregarem els models entrenats i els passarem les imatges contendides a la carpeta Annotated patches.

Annotated patches conte 1330 imatges amb tres classes, negatiu, dubtós i positiu. La classe del mig no ens interessa per tant la ignorarem i ens quedarem amb 1255 imatges. Algo important és que ara ja no parlarem de pacients i carpetes sinó d'imatges, no separarem per pacients en cap moment.
Aquest dataset es troba des-balançejat, tenint només 164 positius d'aquest 1255. Per tant haurem de utlitzar tècniques que siguin resistents a això. En aquest dataset trobem un arxiu csv que ens indica el target (si és positiu o negatiu) de cada una de les imatges.

Llegirem totes les imatges i les passarem pel model, i això ens donarà una imatge reconstruïda. En les imatges sanes la reconstrucció hauria de ser semblant ja que el model ha vist imatges similars anteriorment. Però en veure les imatges positives el model no hauria de ser capaç de reconstruir les bacteries.
Aquí radica la part important, hi haura més diferència en les imatges positives que en les negatives. Aquesta diferència ens hauria de permetre classificar-les.

Per fer la diferència simplement restarem la entrada i la sortida. Clar que només ho farem per el canal blau. Això es deu a que es el que tindrà major diferència entre entrada i sortida, ja que en les positives el model s'haurà inventat punts blaus. Pensem que és millor fer-ho així que pel canal vermell. El motiu és que quan passa de tenir vermell a no tenir, en ser aquest 'vermell' molt fosc no hi haurà gaire diferència en números. En canvi, en el canal blau passem d'un punt que te un valor petit (ja que és fosc) a un valor molt gran.

Una vegada tenim el llistat de diferències toca trobar quin és el treshold que separa les dues classes. Per fer-ho hem utilitzat la corba roc.
Aquest mètode és resident al des-balançeig ,pràcticament està fet per evitar això, i per tant el nostre salvador en aquest cas. Dividirem la sortida en train i test. Tot i que no entrenarem res trobarem els valors de treshold en base a un i els provarem en base a l'altre.
Podem veure les corbes roc en *****
D'aqui podem trobar varios punts optims de tall, basats en la proximitat a la cantonada esquerra, el f-score o altres metriques que ens interesi.
Hem escollit *****
Ara per trobar els valors de com de bo es tot el nostre sistema simplement hem de pasar la part de test per aquest treshold i calcular les metriques.
Ens dona ****

Si està malament:

Historogrames,
explicació de què generalitza


## RESULTATS

Tenim dos resultats un per saber que tan bé funciona el nostre autoencoder amb unes gráfiques de la loss i per saber que també classifiquem pacients dient quins són negatius i quins són positius.  




### Resultats Autoencoder
 Com ha resultats de la loss pels diferents trains hem obtingut han resultats molt semblants. Com es pot veure a les imatges de la carpeta gràfiques. La principal diferencia entre les gràfiques que tenen diferent train és que la de 10 imatges és la que te pitjor resultats amb una loss de 0.00105 en el train i 0.00140 en el test, amb 200 époques. Per altre banda la de 20 imatges ha obtingut una loss de 0.00088 i 0.00105 en train i test respectivament molt semblant a la de 30 imatges que consta dels següents resultats 0.00076 i 0.00112 en train i test  respectivament. Veient la caiguda de la loss i els resultats final hem vist que la de 30 obté millor resultats ja que té la loss de test i train  més baixex, degut a que és la que més imatges d'entrenament per tant pot recrear millor el teixit, ja que tots utitlizen el mateix atuoencoder i paràmetres. Les losses de train i test obntingudes en conjunt més petites són  0.00076 i 0.00112 en train i test respectivament.
 **acabar cuando acabe de ejecutar **

Al veure les imatges reconstruides podem veure que fa una reoconstrucció de les imatges bastant properes a les orignals amb la diferencia de que les reconstruides són borrosses i les que estan infectades no tenen la capacitat de generar el color vermell. Per tant reconstrucció de les infectades no tenen el color vermell. Aquesta compració es poden veure en les carpetes d'imatges originals i en les carpetes d'imatges reconstruides. 


### Resultat Classificació

## CONCLUSIONS

L'autoencoder es veu bastant esbiaxat cap a les imatges sanes per tant podem les imatges reconstruides eren bastant semblants a les esperades, al no poder-se observar el bacteri en les imatges on el pacient està infectat. Degut a que amb les capes convulocionas ens permet etruere les característiques necesaries per reconstruir les imatges que rep, com si fossin sanes.

Per veure que també classifica el nostre model ens bassat en la ROC-curve i segons el treshold que marca la gràfica observem el recall que obtindrem. Tenim un recall de ****** rellenar caundo ete el final *********.

Tot i que l'apart de reconstruir imatges del autoencoder va força bé hi en alguns casos on el vermell no lo suficient significatiu i per tant no genera suficient loss perquè es pogui classificar com a infectat i no supera el threshold mercat la gràfica ROC-curve. 


