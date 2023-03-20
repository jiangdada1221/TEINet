from train import *
from scipy.special import expit
import argparse
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

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--dset_path',type=str) 
    parser.add_argument('--save_prediction_path',type=str,default='None')
    parser.add_argument('--use_column',type=str,default='CDR3.beta')
    parser.add_argument('--batch_size',type=int,default=128)
    parser.add_argument('--model_path',type=str)
    parser.add_argument('--device',type=str,default='cuda:0')
    args = parser.parse_args()

    f = pd.read_csv(args.dset_path)
    #load model
    model_tcr = TCRpeg(hidden_size=768,num_layers = 3,load_data=False,embedding_path='encoders/aa_emb_tcr.txt',device=args.device)
    model_tcr.create_model()
    model_epi = TCRpeg(hidden_size=768,num_layers = 3,load_data=False,embedding_path='encoders/aa_emb_tcr.txt',device=args.device)
    model_epi.create_model()
    model = TEINet(en_tcr=model_tcr,en_epi = model_epi,cat_size=768*2,dropout=0.1,normalize=True,weight_decay = 0,device=args.device).to(args.device)
    model.load_state_dict(torch.load(args.model_path))
    # model = torch.load(args.model_path,map_location='cuda:0')
    cdrs,epitopes,labels = f['CDR3.beta'].values, f['Epitope'].values, f['Label'].values
    pres = predict_only(cdrs,epitopes,model)
    print(AUC(labels,pres))
    with open(args.save_prediction_path,'w') as f:
        for i in range(len(pres)):
            f.write(str(pres[i]) + ',' + str(labels[i])+'\n')
    
