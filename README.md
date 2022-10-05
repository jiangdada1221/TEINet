# TEINet
TEINet is designed for the prediction of the specificity of TCR binding, using only the CDR3β chain of TCR and the epitope sequence within the pMHC complex. Following the concept of transfer learning, TEINet employs two separate pretrained encoders to convert TCRs and epitopes into numerical vectors, utilizing the architecture of recurrent neural networks to handle a variety of sequence lengths. We summarize the four current negative sampling strategies applied in the previous work and contrast them. <br />

<img src="https://github.com/jiangdada1221/tensorflow_in_practice/blob/master/TEINet.jpg" width="800"> <br />

## Dependencies
TEINet is writen in Python based on the deeplearning library - Pytorch. Compared to Tensorflow, Pytorch is more user friendly and you can control the details of the training process. I would strongly suggest using Pytorch as the deeplearning library so that followers can easily modify and retrain the model. TEINet utilizes the TCRpeg as sequence encoders, thus we suggest users checking the [TCRpeg package](https://github.com/jiangdada1221/TCRpeg) first. <br />

The required software dependencies are listed below:
 ```
Numpy
matplotlib
tqdm
pandas
scikit-learn
scipy
torch >= 1.1.0
Levenshtein
 ```

## Data

 All the data used in the paper is publicly available, so we suggest readers refer to the original papers for more details. We also upload the processed data which can be downloaded via [this link](https://drive.google.com/file/d/1ioEkYeIdLMafYgoNER33QrThKHlgZCzZ/view?usp=sharing). Description.txt in the data zip file gives a brief descripion for each file.

## Train TEINet

####Training script:
```
python train.py --train_file data/train_pos.csv --test_file data/test.csv --epochs 1 --model_path results/model.pth
```
Please check the train.py for details (Or type python train.py --h). Note that the default negative sampling strategy is Uniform Epiope, in order to choose other strategies, you need to specify: <br />
```--fre 0 ``` for Random Epitope <br />
```----sample_strategy sample_tcr --reference_tcr path_to_reference_tcr``` for Reference TCR <br />
```--sample_strategy sample_tcr``` for Random TCR <br /> <br />
####Predict for TCR-epitope pairs [(t1,e1),(t2,e2),...]
```
from predict import predict_only
predictions = predict_only(ts,es,model=results/model.pth)
```
####Compute the score difference in different region of Complexes in PDB database
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

We check email often, so for instant enquiries, please contact us via [email](mailto:jiangdada12344321@gmail.com). Or you may open an issue section.

## License

Free use of TEINet is granted under the terms of the GNU General Public License version 3 (GPLv3).

