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
  - imatges_originals_train_X: carpeta que conté algunes de les imatges originals que utilitzem com a train que no tenen presència de l'Helicobacter (X fa referència al model)
  - imatges_reconstruides_train_X: carpeta que conté algunes de les imatges reconstruides del train pel nostre autoencoder (X fa referència al model)
  - imatges_originals_test_X: carpeta que conté algunes de les imatges originals que utilitzem com a test que no tenen presència de l'Helicobacter (X fa referència al model)
  - imatges_reconstruides_test_X: carpeta que conté algunes de les imatges reconstruïdes del train pel nostre autoencoder (X fa referència al model)
  - Gràfiques: carpeta que conté les gràfiques de les losses tant de train com de test de tots 4 models
  - Pickle.py: fitxter python que conté el codi on es generen les gràfiques de loss del train i test.
  - Pickle: carpeta que conté els arxius pickle per fer les losses
  - imagenes_originales_segona_part_X: carpeta que conté imatges amb pacients positius, és a dir, amb el bacteri (X fa referència al model)
  - imagenes_reconstruidas_segona_part_X: carpeta que conté imatges reconstruides de pacients positius (X fa referència al model)




## BASE DE DADES
La base de dades que utilitzem han estat una que se'ns ha proporcionat en el Campus Virtual. La base de dades està distribuïda de la següent manera:
Una carpeta que està formada per moltes altres carpetes, on cadascuna fa referència a un pacient i conté les imatges de les mostres del teixit. Aquestes imatges venen etiquetades per la densitat d'Helicobacter pylori: baixa, alta i negativa.
Per la primera part hem separat en train test tots els pacients que no tenen presència de l'Helicobacter pylori. En el nostre cas hem decidit separar en train i test, per tal de poder comprovar que l'autoencoder que fem funciona correctament i és robust.
Pel train hem agafat 10,20,30,50 pacients (carpetes), és a dir, fem  4 models amb diferents número de pacients, però amb un test constant de 5 pacients sempre. Hem anat probant al principi amb pocs pacients i hem anat augmentant al anar fent proves per tant tenim 4 models de base de dades.


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
- epoques = 200 (menys a la de 50 pacients al train que tenim 350 èpoques)

#### Loss

Entre les imatges generades per l'autoencoder i les originals calculem la seva diferència i amb l'optimizer MSE, obtenim la loss, específicament una loss per les dades train i una loss per les dades test. Això ho guardem en un objecte pickle, per tal de poder manipular més endavant com vulguem.
El codi on generem els gràfics pertinents està al fitxer "pickle.py".
Aquestes gràfiques ens serveixen per monitorar i veure que l'autoencoder és robust i correcte i a la vegada esbiaixat, per tal de no poder recrear la imatge amb el bacteri.

 
### SEGONA PART: CLASSIFICACIÓ

Després d'haver entrenat un autoencoder amb només pacients sans, utilitzarem aquest model per poder classificar pacients no sans, és a dir, positius; i sans, és a dir, negatius. Per fer-ho carregarem els models entrenats i els passarem les imatges contendides a la carpeta Annotated patches.

Annotated patches conte 1330 imatges amb tres classes, negatiu, dubtós i positiu. La classe del mig no ens interessa per tant la ignorarem i ens quedarem amb 1255 imatges. Una dada important és que ara ja no parlarem de pacients i carpetes sinó d'imatges, no separarem per pacients en cap moment.
Aquest dataset es troba des-balançejat, tenint només 164 positius d'aquest 1255. Per tant haurem d'utlitzar tècniques que siguin resistents a això. En aquest dataset trobem un arxiu csv que ens indica el target (si és positiu o negatiu) de cada una de les imatges.

Llegirem totes les imatges i les passarem pel model, i això ens donarà una imatge reconstruïda. En les imatges sanes la reconstrucció hauria de ser semblant, ja que el model ha vist imatges similars anteriorment. Però en veure les imatges positives el model no hauria de ser capaç de reconstruir les bacteries.
Aquí radica la part important, hi haurà més diferència en les imatges positives que en les negatives. Aquesta diferència ens hauria de permetre classificar-les.


Per fer la diferència simplement restarem la entrada i la sortida. Nomes restarem el canal blau ja que aquets es el que major diferencia presta entre la sortida i la entrada en les imatges positives. Això es perque tractem amb rgb. Els punts 'vermells' de les bacteries tenen un valor de vermell baix i per tant tot i que aquest valor desaparegui en cuan a nombres no es nota. En canvi en el canal blau si que hi ha diferrencia. En aquests punts el autoencoder s'inventa punts blaus que al ser mes clars si que tene un valor elevat. Basicament en el canal vermell pasem de valor baix a valor molt baix mentre que en el blau pasem de valor molt baix a valor alt.

En la carpeta grafiques podem trobar un seguit de histogrames on veiem les losses per cada un dels canals i separarat per clase. Aqui veiem com els histogrames de el canal general, es a dir tot, i del canal vermell son molt semblants. Hi ha bastant solapament (es poden comparar facilment ja que tenen el mateix eix x). En cambi en el canal blau no trobem tant solapament. En el cas del model 50 trobem que mes de la meitat de les dades es troben perfectament separables. Parlarem mes de això en el partat de resultats.


Una vegada tenim el llistat de diferències toca trobar quin és el millor treshold que separa les dues classes. Per fer-ho hem utilitzat la corba roc.
Aquest mètode és resistent al des-balançeig, pràcticament està fet per evitar això, i per tant el nostre salvador en aquest cas. Dividirem la sortida en train i test. Tot i que no entrenarem res, trobarem els valors de treshold en base a un i els provarem en base a l'altre.
Podem veure les corbes roc a la carpeta de "Grafiques".
D'aqui podem trobar alguns punts optims de tall, basats en la proximitat a la cantonada esquerra, el f-score o altres mètriques que ens interessi.
En el nostre cas i hem escollit el mes proxim a la cantonada esquerra superior. No hem agafat altres funcions ja que hem trobat que aquestes agafaven punts mes a la esquerra (en la grafica) del que voliem. Al ser un cas medic volem que no sens escapin posibles poacients amb enfermatats per tant buscarem un valor de true positive rate elevat tot i que impliqui augmentar false positive rate.

Per tant: Pasarem tot el conjunt de dades per els autonecoders, dividirem les loses resultants en train i test, farem una roc curve amb el train, calcularem el treshold mes proper a la cantonada esquerra superior i aplicarem aquest treshold al conjunt test. Comparant aquest ultim resultat amb el target real obtindrem les metriques.





## RESULTATS
La part de resultats hem decidit dividir-la en dos blocs. Primerament explicarem els resultats del autoencoder i despés de la calssificació.


### Resultats Autoencoder

Per comprovar que l'autoencoder és robust i correcte, hem mirat tant les imatges reconstruides dels diferents models, com les losses d'aquest. 

Al comparar les imatges originals i les reconstruides de tots els models, podem veure que es reconstrueixen bastant bé les imatges del train, tot i que estan una mica borroses, l'autoencoder és el suficientment específic i detallat per obtenir tots els patrons que hi ha a les imatges i reconstruir-les satisfactoriament. A més a més, una de les raons que surten borroses les imatges tan en les originals com les reconstruides és perquè al principi les fem més petites, de 256x256x3 a 64x64x3.
(per veure les imatges mireu les carpetes que posen imatges_originals_train_X i imatges_reconstruides_train_X)

Per altra banda, també hem comprovat que és robust i que no hi ha overfitting, així doncs provant també amb els pacients test, les imatges és reconstrueixen igual de bé que en el train. (per veure les imatges mireu les carpetes que posen imatges_originals_test_X i imatges_reconstruides_test_X)

Una altra manera de veure que el nostre autoencoder està esbiaixat, és comprovar amb imatges de pacients infectats. Les imatges es reconstrueixen sense reconstruir la part vermella que és la part infectada del teixit, i la reconstrueix de color blau. Per tant comprovem que efectivament el nostre autoencoder està esbiaixat. 
(per veure les imatges mireu les carpetes que posen imatges_amb_bacteri_original i imatges_amb_bacteri_reconstruida)

Un cop hem comprovat les imatges, ens fixem a les gràfiques de loss de train i test de tots 4 models. 
Podem veure que les gràfiques de loss on tenim com a train 10, 20 i 30 pacients són molt semblants. Al llarg de les 200 èpoques per la nostra xarxa neuronal que utilitza capes de convolució (Conv2), capes de maxpooling i funcions d'activació ReLU, es observa un patró inicial bastant eficient. La loss comença en un valor relativament baix, concretament en tots 3 casos, indicant un bon ajust inicial del model als patrons presents en les dades d'entrenament.

La ràpida disminució inicial de la pèrdua podria podria ser pel conegut fenòmen "ajust fi", on el model s'adapta de manera eficient a les característiques del conjunt d'entrenament. Aquest procés és comú i sol ser un senyal positiu, indicant que la xarxa neuronal està aprenent de manera efectiva al principi de l'entrenament.

No obstant això, a mesura que avancen les èpoques, s'observen pics i/o soroll a la corba de la loss. Això és degut a que hi ha variabilitat estocàstica. Però, encara així podem veure que amb més pacients i més èpoques com en el cas del model 4, per entrenar, hi ha menys pics, això vol dir que la presència de la variabilitat estocàstica no és tan pronunciada com abans. 

Per altra banda, podem veure que la loss del test segueix la mateixa tendència que la del train, així doncs podem afirmar que aprén igual de bé, però també té presència de pics per tant hi ha aquesta variablitat estocàstica. 
(podeu mirar les gràfiques a la carpeta Grafiques, les gràfiques són les següents: loss_test_X o loss_train_X on X és el model (10,20,30 o 50))



### Resultat Classificació

Pasant les imatges del segon dataset i calculant la diferencia com hem descrit anteriorment, ens trobem amb un problema. Els models no son capaç de distingir entre un 40% de les mostres positives, i les mostres negatives(amb la resta si que ho fa). Això es pot veure tant amb es histogrames com amb la roc curve. En el histograma podem veure que hi ha valors presents en ambdues imatges, En la curva roc podem veure com el true positive rate augmenta rapidament fins a cert punt , on tomba cap a la dreta. Es aquest punt on es començen a solapar fortament positius i negatius.
Això vol dir que per aquests positius el nostre model els esta reconstuint be i per això no es capaç de deiferenciar-los. El model generalitza massa.




|               | F-score | True positive rate  | False positive rate| Accuracy |
| ------------- | ------------- | ------------- | ------------- | ------------- |
| Model 10     | 0.5414  | 0.7246  | 0.1461  | 0.834  |
| Model 20     | 0.5632  | 0.7462  | 0.1339  | 0.848  |
| Model 30     | 0.4811  | 0.7647  | 0.2152  | 0.780  |
| Model 50     | 0.3963  | 0.7627  | 0.2698  | 0.732  |
| Model Mitja  | 0.4955±0.0700  | 0.7496±0.0183  | 0.1913±0.0635  | 0.7985±0.0532  |





## CONCLUSIONS

Podem concloure que l'autoencoder està bastant esbiaixat a les dades sanes i per tant podem afirmar que reconstrueix correctament les imatges sense bacteri i reconstrueix com volem le simatges amb presència del bacteri.
Tot i que les reconstruccions estan bastant bé, Per altra banda, a causa de la variablitat estocàstica fa que les losses de train i test tinguin una mica de soroll. Però hem vist que com més pacients i més èpoques per entrenar, menys soroll hi ha, així doncs vol dir que encara que hi ha aquesta variabilitat estocàstica no afecta massa, però per exemple a la del test si, perquè sempre tenim 5 pacients. 


Per altra banda, podem vuere que la classificació ....
Per veure que també classifica el nostre model ens bassat en la ROC-curve i segons el treshold que marca la gràfica observem el recall que obtindrem. Tenim un recall de ****** rellenar caundo ete el final *********.

Tot i que l'apart de reconstruir imatges del autoencoder va força bé hi en alguns casos on el vermell no lo suficient significatiu i per tant no genera suficient loss perquè es pogui classificar com a infectat i no supera el threshold mercat la gràfica ROC-curve. 


