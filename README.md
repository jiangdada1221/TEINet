# TEINet 
TEINet is designed for the prediction of the specificity of TCR binding, using only the CDR3β chain of TCR and the epitope sequence within the pMHC complex. Following the concept of transfer learning, TEINet employs two separate pretrained encoders to convert TCRs and epitopes into numerical vectors, utilizing the architecture of recurrent neural networks to handle a variety of sequence lengths. We summarize the four current negative sampling strategies applied in the previous work and contrast them. <br /> 

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

## Train TEINet

#### Training script:
```
python train.py --train_file data/train_pos.csv --test_file data/test.csv --epochs 1 --model_path results/model.pth
```
Please check the train.py for details (Or type python train.py --h). Note that the default negative sampling strategy is Unified (Uniform) Epiope, in order to choose other strategies, you need to specify: <br /> <br />
```--fre 0 ``` for Random Epitope <br />
```----sample_strategy sample_tcr --reference_tcr path_to_reference_tcr``` for Reference TCR <br />
```--sample_strategy sample_tcr``` for Random TCR <br /> <br />
Note that if you want to use a static training dataset, please refer to the __epitope_sample_1fold__ or __tcr_sample_1fold__ functions in utils.py for sampling negatives based on the positive pairs using different strategies.
#### Predict for TCR-epitope pairs [(t1,e1),(t2,e2),...]
```python
from predict import predict_only
from utils import load_teinet
teinet = load_teinet('results/model.pth',device='cuda:0')
predictions = predict_only(ts,es,model=teinet)
```
Or you can use the script to make predictions for user's input file using trained model:
```
python predict.py --dset_path path_to_data --model_path path_to_teinet --use_column CDR3.beta --save_prediction_path results/test.txt
```

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

## Citation (will be published in Briefings in Bioinformatics)
```
@article{jiang2022teinet,
  title={TEINet: a deep learning framework for prediction of TCR-epitope binding specificity},
  author={Jiang, Yuepeng and Huo, Miaozhe and Li, Shuaicheng},
  journal={bioRxiv},
  pages={2022--10},
  year={2022},
  publisher={Cold Spring Harbor Laboratory}
}
```

