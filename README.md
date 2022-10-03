# TCRpeg
TCRpeg is a deep probabilistic neural network framework used for inferring probability distribution for given CDR3 repertoires. Beyond that, TCRpeg can provide numerical embeddings for TCR sequences, generate new TCR sequences with highly similar statistical properties with the training repertoires. TCRpeg can be easily extended to act as a classifier for predictive purposes (TCRpeg-c). <br />

<img src="https://github.com/jiangdada1221/tensorflow_in_practice/blob/master/workflow.png" width="800"> <br />

## Installation
TCRpeg is a python software implemented based on the deeplearning library - Pytorch. It is available on PyPI and can be downloaded and installed via pip: <br />
 ```pip install tcrpeg``` <br />
TCRpeg can be also installed by cloning the Github repository and using the pip: <br />
 ```pip install .``` <br />
The required software dependencies are listed below:
 ```
Numpy
matplotlib
tqdm
pandas
scikit-learn
scipy
torch >= 1.1.0
 ```

## Data

 All the data used in the paper is publicly available, so we suggest readers refer to the original papers for more details. We also upload the processed data which can be downloaded via [this link](https://drive.google.com/file/d/1rqgn6G2js85QS6K7mvMwOEepm4ARi54H/view?usp=sharing)

## Usage instructions

Define and train TCRpeg model:
```
from tcrpeg.TCRpeg import TCRpeg
model = TCRpeg(embedding_path='tcrpeg/data/embedding_32.txt',load_data=True, path_train=tcrs) 
#'embedding_32.txt' records the numerical embeddings for each AA; We provide it under the 'tcrpeg/data/' folder.
#'tcrs' is the TCR repertoire ([tcr1,tcr2,....])
model.create_model() #initialize the TCRpeg model
model.train_tcrpeg(epochs=20, batch_size= 32, lr=1e-3) 
```
Use the pretrained TCRpeg model for downstream applications:
```
log_probs = model.sampling_tcrpeg_batch(tcrs)   #probability inference
new_tcrs = model.generate_tcrpeg(num_to_gen=1000, batch_size= 100)    #generation
embs = model.get_embedding(tcrs)    #embeddings for tcrs
```

 We provide a tutorial jupyter notebook named [tutorial.ipynb](https://github.com/jiangdada1221/TCRpeg/blob/main/tutorial.ipynb). It contains most of the functional usages of TCRpeg which mainly consist of three parts: probability inference, numerical encodings & downstream classification, and generation. The python scripts and their usages are shown below: <br />

| Module name                                    | Usage                                              |    
|------------------------------------------------|----------------------------------------------------|
| TCRpeg.py                                      | Contain most functions of TCRpeg                   |
| evaluate.py                                    | Evaluate the performance of probability inference  |
| word2vec.py                                    | word2vec model for obtaining embeddings of AAs     |
| model.py                                       | Deep learning models of TCRpeg,TCRpeg-c,TCRpeg_vj  |
| classification.py                              | Apply TCRpeg-c for classification tasks            |
| utils.py                                       | N/A (contains util functions)                      |
| process_data.py                                | Construct the universal TCR pool                   |

## Contact

We check email often, so for instant enquiries, please contact us via [email](mailto:jiangdada12344321@gmail.com). Or you may open an issue section.

## License

Free use of soNNia is granted under the terms of the GNU General Public License version 3 (GPLv3).

