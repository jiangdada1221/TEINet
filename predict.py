from train import *
from scipy.special import expit

def predict(cdrs,epitopes,labels,model,batch_size = 128):
    '''
    Evaluate the test pairs.
    @cdrs: list of tcrs
    @epitopes: list of epitopes
    @labels: the corresponding labels for each tcr-epitope pair
    @model: The trained TEINet model
    @batch_size: batch_size
    Return:
        A dictionary with keys=[y_pres, y_trues, AUC]
    '''
    batch_num_total = len(cdrs) // batch_size if len(cdrs) % batch_size == 0 else len(cdrs) // batch_size + 1
    y_pres,y_trues = [],[]
    for batch_num in tqdm(range(batch_num_total)):
        end = (batch_num+1)*batch_size if (batch_num+1)*batch_size <= len(cdrs) else len(cdrs)                
        ts,es = cdrs[batch_num*batch_size:end],epitopes[batch_num * batch_size:end] 
        score = model(ts,es) 
        y_pres.extend(expit(score.view(-1).detach().cpu().numpy()))
        y_trues.extend(labels[batch_num*batch_size : end])
    return {'y_pres':y_pres,'y_trues':y_trues,'AUC':AUC(y_trues,y_pres)}

def predict_only(cdrs,epitopes,model,batch_size = 128):
    '''
    Predict the probs for each pairs.
    @cdrs: list of tcrs
    @epitopes: list of epitopes
    @model: The trained TEINet model
    @batch_size: batch_size
    Return:
        A list contains the scores
    '''
    batch_num_total = len(cdrs) // batch_size if len(cdrs) % batch_size == 0 else len(cdrs) // batch_size + 1
    y_pres = []
    for batch_num in tqdm(range(batch_num_total)):
        end = (batch_num+1)*batch_size if (batch_num+1)*batch_size <= len(cdrs) else len(cdrs)                
        ts,es = cdrs[batch_num*batch_size:end],epitopes[batch_num * batch_size:end] 
        score = model(ts,es) 
        y_pres.extend(expit(score.view(-1).detach().cpu().numpy()))
    return y_pres