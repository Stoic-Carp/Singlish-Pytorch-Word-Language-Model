# Singlish Word-level Language Model

(The code is largely based on Pytorch official example of word language model on Github, link here: https://github.com/pytorch/examples/tree/master/word_language_model)

This example trains a light version of the Transformer model on a language modeling task. The light Transformer model retains the encoding layers but replaces the decoding layers with a simple linear and log-softmax layer. 
By default, the training script uses the Singlish dataset, provided.
Wikitext-2 and jokes data is also included. 
The trained model can then be used by the generate script to generate new text.

```bash 
python main.py --cuda --epochs 6           # Train a Transformer on Singlish with CUDA with 6 epochs
python generate.py                         # Generate samples from the trained LSTM model.

```

During training, if a keyboard interrupt (Ctrl-C) is received,
training is stopped and the current model is evaluated against the test dataset.

The `main.py` script accepts the following arguments:

```bash
optional arguments:
  -h, --help                       show this help message and exit
  --data DATA                      location of the data corpus
  --model MODEL                    type of recurrent net (RNN_TANH, RNN_RELU, LSTM, GRU, Transformer)
  --emsize EMSIZE                  size of word embeddings
  --nhid NHID                      number of hidden units per layer
  --nlayers NLAYERS                number of encoder layers of the light Transformer model
  --nheads NHEADS                  number of attention heads in the Transformer encoding layer 
  --lr LR                          initial learning rate
  --clip CLIP                      gradient clipping
  --epochs EPOCHS                  upper epoch limit
  --batch_size N                   batch size 
  --bptt BPTT                      sequence length to be used during training 
  --dropout DROPOUT                dropout applied to layers (0 = no dropout)
  --decay DECAY                    learning rate decay per epoch
  --tied                           tie the word embedding and softmax weights (only for GRU or LSTM)
  --seed SEED                      random seed
  --cuda                           use CUDA
  --log-interval N                 report interval
  --save SAVE                      path to save the final model
  --onnx-export                    path to export the final model in onnx format
```

With these arguments, a variety of models can be tested.
As an example, the following arguments produce slower but better models:

```bash
python main.py --cuda --emsize 650 --nhid 650 --dropout 0.5 --epochs 40           
python main.py --cuda --emsize 650 --nhid 650 --dropout 0.5 --epochs 40 --tied    
python main.py --cuda --emsize 1500 --nhid 1500 --dropout 0.65 --epochs 40        
python main.py --cuda --emsize 1500 --nhid 1500 --dropout 0.65 --epochs 40 --tied 
```
