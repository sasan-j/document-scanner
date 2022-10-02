# Prerequisites
```
brew install ffmpeg
```

A lot of the code especially processing SmartDoc dataset was initially taken from [this](https://github.com/khurramjaved96/Recursive-CNNs) repo, all the credits goes there. 


# Dataset
You need to download and adjust the command below accordingly. The first parameter is the path to raw dataset and second path is where the processed results will be placed into.
You can find the dataset [here](https://zenodo.org/record/1230218)

```sh
python main.py process-smartdoc ../datasets/SmartDoc/ ../data-doc/SmartDocProcessed/
```



Generate dataset from processed smartdoc
```sh
python main.py document-data-generator ../data-doc/SmartDocProcessedTrain/ ../data-doc/SmartDocMaskedTrain/
python main.py document-data-generator ../data-doc/SmartDocProcessedValid/ ../data-doc/SmartDocMaskedValid/

```

```sh
python main.py train --name t1 --train-dir ../data-doc/SmartDocMaskedTrain --valid-dir ../data-doc/SmartDocMaskedValid
```
