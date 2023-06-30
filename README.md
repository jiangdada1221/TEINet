# TEINet 
TEINet is designed for the prediction of the specificity of TCR binding, using only the CDR3β chain of TCR and the epitope sequence within the pMHC complex. Following the concept of transfer learning, TEINet employs two separate pretrained encoders to convert TCRs and epitopes into numerical vectors, utilizing the architecture of recurrent neural networks to handle a variety of sequence lengths. __We summarize the four current negative sampling strategies applied in the previous work and contrast them.__ <br /> 

<img src="https://github.com/jiangdada1221/tensorflow_in_practice/blob/master/TEINet.jpg" width="800"> <br />

## Dependencies
TEINet is writen in Python based on the deeplearning library - Pytorch. Compared to Tensorflow, Pytorch is more user friendly and you can control the details of the training process. I would strongly suggest using Pytorch as the deeplearning library so that followers can easily modify and retrain the model. TEINet utilizes the TCRpeg as sequence encoders, thus we suggest users checking the [TCRpeg package](https://github.com/jiangdada1221/TCRpeg) first. (It's not neccessary to install tcrpeg as it is already included in this repository) <br />

The required software dependencies are listed below:
 ```
Numpy
matplotlib
tqdm
pandas
scikit-learn
scipy
torch >= 1.1.0 (Tested on 1.8.0)
Levenshtein
 ```

## Data

 All the data used in the paper is publicly available, so we suggest readers refer to the original papers for more details. We also upload the processed data which can be downloaded via [this link](https://drive.google.com/file/d/1ioEkYeIdLMafYgoNER33QrThKHlgZCzZ/view?usp=sharing). Description.txt in the data zip file gives a brief descripion for each file.

## Usage of TEINet

#### Training TEINet:
```
python train.py --train_file data/train_pos.csv --test_file data/test.csv --epochs 30 --model_path results/model.pth
```
Please check the train.py for details (Or type python train.py --h). Note that the default negative sampling strategy is Unified (Uniform) Epiope, in order to choose other strategies, you need to specify: <br /> <br />
```--fre 0 ``` for Random Epitope <br />
```----sample_strategy sample_tcr --reference_tcr path_to_reference_tcr``` for Reference TCR <br />
```--sample_strategy sample_tcr``` for Random TCR <br /> <br />
The default training process use the dynamic sampling strategy (sampling negatives on the fly). If you don't want that, please enter *--static 1* and at the same time the training file should also contain negative pairs in addition to the positive pairs. For constructing a full dataset (negative+positive pairs), please refer to the __epitope_sample_1fold__ or __tcr_sample_1fold__ functions in utils.py for sampling negatives based on the positive pairs using different strategies.

#### Predict TCR-epitope pairs
```python
from predict import predict_only
from utils import load_teinet
teinet = load_teinet('results/model.pth',device='cuda:0') #load the trained model
predictions = predict_only(ts,es,model=teinet) #make predictions; ts (es): a list containing the CDR3s (epitopes)
```
Or you can use the script to make predictions for user's input file using trained model:
```
python predict.py --dset_path path_to_data --model_path path_to_teinet --use_column CDR3.beta --save_prediction_path results/test.txt
```
We also provide our trained TEINet models: [teinet_data.pth](https://drive.google.com/file/d/12pVozHhRcGyMBgMlhcjgcclE3wlrVO32/view?usp=sharing) and [large_dset.pth](https://drive.google.com/file/d/1dguZKJL_NH6heBcIE1hpM7WBdYnPJCIT/view?usp=sharing). <br />
The teinet_data.pth model was trained on the dataset in the paper (~40,000 pairs); The large_dset.pth was trained on a larger dataset (~100,000 pairs) collected in [TCR2vec](https://www.biorxiv.org/content/10.1101/2023.03.31.535142v1), which combines data from VDJdb, McPAS-TCR and IEDB.

#### Compute the score difference in different region of Complexes in PDB database
```
python pdb_distance.py --threshold 5.0 --model_path results/model.pth
```
<br />

| Module name                                    | Usage                                              |    
|------------------------------------------------|----------------------------------------------------|
| model.py                                      | TEINet model                   |
| sampler.py                                    | Sampling strategies  |
| train.py                                    | Training the TEINet model     |
| pdb_distance.py                                       | Compute the score difference in different regions  |
| utils.py                              | Split and sampling functions as well as functions used in pdb_distance.py             |
| predict.py                                       | Make predictions for user input                      |
| tcrpeg                                | TCRpeg package; used for encoding sequences                   |

## Contact
```
Name: Yuepeng Jiang
Email: yuepjiang3-c@my.cityu.edu.hk/yuj009@eng.ucsd.edu/jiangdada12344321@gmail.com
Note: For instant query, feel free to send me an email since I check email often. Otherwise, you may open an issue section in this repository.
```
Welcome for reporting any bugs!

## License

Free use of TEINet is granted under the terms of the GNU General Public License version 3 (GPLv3).

## Citation
```
@article{jiang2023teinet,
  title={TEINet: a deep learning framework for prediction of TCR--epitope binding specificity},
  author={Jiang, Yuepeng and Huo, Miaozhe and Cheng Li, Shuai},
  journal={Briefings in Bioinformatics},
  volume={24},
  number={2},
  pages={bbad086},
  year={2023},
  publisher={Oxford University Press}
}
```

