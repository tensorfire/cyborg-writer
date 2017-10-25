# Training New Models

Cyborg writer uses character-by-character LSTMs to generate text. 

You can see our implementation of the network in `train.py`. 

Dependencies are `keras`, `tensorflow`, and `unidecode`. 

Once a model has been trained, place the `.hdf5` data into this 
directory and use the `convert-hdf5.py` tool. 

    ~/P/c/training ❯ python train.py  --help
    usage: train.py [-h] [--dropout DROPOUT] [--size SIZE] [--layers LAYERS]
                    [--seq_length SEQ_LENGTH] --corpus CORPUS --model_name
                    MODEL_NAME

    optional arguments:
      -h, --help            show this help message and exit
      --dropout DROPOUT
      --size SIZE
      --layers LAYERS
      --seq_length SEQ_LENGTH
      --corpus CORPUS
      --model_name MODEL_NAME
