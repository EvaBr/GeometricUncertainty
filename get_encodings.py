import torch
from pathlib import Path
import torch.nn as nn
import numpy as np
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
from dataloaderSCAPIS import SCAPISDataset
from torch.utils.data import DataLoader
from customPredictTiled import custom_predict_encoder_sliding_window


#here we load the trained nnunet weights, load the model and the weight, 
# and extract the bottleneck encodings for a given dataset.
model_path = Path("/home/evabreznik/Skrivbord/MAIASTUFF/nnunet_data"+
                  "/nnUNet_results/Dataset666_ASOCA/"+
                  "nnUNetTrainer__nnUNetResEncUNetMPlans__3d_fullres/")

fold = 0  # specify which fold to use

# Load a trained model
predictor = nnUNetPredictor(
    tile_step_size=0.5,
    use_gaussian=True,
    use_mirroring=True,
    perform_everything_on_device=True,
    device=torch.device('cuda')
)

predictor.initialize_from_trained_model_folder(
    model_path,  # path to model folder
    use_folds=(fold,),
    checkpoint_name="checkpoint_final.pth"
)

#model = predictor.network  # <-- this is the actual nnU-Net torch model
#with open("architecture.txt", "w") as f:
#    f.write(str(model))

datasetTrain = SCAPISDataset("../dataSC/train") #loads img, preprocesses, adds a dimension for channels
datasetVal = SCAPISDataset("../dataSC/val")
datasetTest = SCAPISDataset("../dataSC/test")
dataloaderTrain = DataLoader(datasetTrain, batch_size=1,
                        shuffle=True, num_workers=0) 
dataloaderVal = DataLoader(datasetVal, batch_size=1,
                        shuffle=False, num_workers=0)
dataloaderTest = DataLoader(datasetTest, batch_size=1,
                        shuffle=False, num_workers=0)
#if you wanna use batch>1, fix padding!!

def get_encs(dataloader, predictor, device):
    all_embeddings = []
    all_labels = []
    L = len(dataloader)
    for i, (scan, label) in enumerate(dataloader):
        print(f"\nscan {i+1}/{L}\n")
        with torch.no_grad():
            #feats = model(scan.to(device))           # bottleneck
            #print("scan shape:", scan.shape)
            feats = custom_predict_encoder_sliding_window(predictor, scan)
            #print("feats shape:", feats.shape)
            #pooled = torch.mean(feats, dim=(2,3,4))  # global avg pool
            all_embeddings.append(feats.cpu().numpy())
            all_labels.append(label)
    return np.vstack(all_embeddings), np.array(all_labels)


trainvaltest = "test"  # "train"  # "test"
dataload = dataloaderTest  # dataloaderTrain  # dataloaderTest
all_embeddings, all_labels = get_encs(dataload, predictor, device="cuda")
print("Embeddings shape:", all_embeddings.shape)
#print("embeddings before stack shape:", len(all_embeddings), all_embeddings[0].shape)
np.save(f"embeddings_{fold}_{trainvaltest}.npy", all_embeddings)
np.save(f"labels_{fold}_{trainvaltest}.npy", all_labels)



#TODO: 
#v dataloaderju se dataloaer za loadat tele embeddinge napisi. Vsaj mislim, da to rabmo?
# Pa definiraj classifier za on top, 
# train classifier. 
#eval to see if we're good enough. 
