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
# Train the corner model
python main.py train --model-type corner --name test --train-dir ../data-doc/SmartDocCornersTrain/ --valid-dir ../data-doc/SmartDocCornersValid
# Train the document model
python main.py train --model-type document --name test --train-dir ../data-doc/SmartDocDocumentsTrain/ --valid-dir ../data-doc/SmartDocDocumentsValid

```
