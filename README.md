# Prerequisites
```
brew install ffmpeg
```


```sh
python main.py process-smartdoc ../datasets/SmartDoc/ ../data-doc/SmartDocProcessed/
```



Generate dataset from processed smartdoc
```sh
python main.py document-data-generator ../data-doc/SmartDocProcessedTrain/ ../data-doc/SmartDocMaskedTrain/
python main.py document-data-generator ../data-doc/SmartDocProcessedValid/ ../data-doc/SmartDocMaskedValid/

```

```sh
python main.py corner-data-generator ../data-doc/SmartDocProcessed/ ../data-doc/SmartDocCorners/
```

```sh
python main.py train --name t1 --train-dir ../data-doc/SmartDocMaskedTrain --valid-dir ../data-doc/SmartDocMaskedValid
```
