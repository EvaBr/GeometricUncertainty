import torch
from pathlib import Path
import torch.nn as nn
import numpy as np
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
from dataloaderSCAPIS import SCAPISDataset
from torch.utils.data import DataLoader
from nnunetv2.inference.predict_from_raw_data import predict_sliding_window_return_logits


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
    device=torch.device('cuda')
)

predictor.initialize_from_trained_model_folder(
    model_path,  # path to model folder
    use_folds=(fold,),
    checkpoint_name="checkpoint_final.pth"
)

model = predictor.network  # <-- this is the actual nnU-Net torch model
#with open("architecture.txt", "w") as f:
#    f.write(str(model))

dataset = SCAPISDataset() #loads img, preprocesses, adds a dimension for batch
dataloader = DataLoader(dataset, batch_size=1,
                        shuffle=True, num_workers=0) 
#if you wanna use batch>1, fix padding!!
configuration = "3d_fullres"
plans_manager = predictor.plans_manager
config_manager = plans_manager.get_configuration(configuration)
patch_size = config_manager.patch_size
print("Patch size:", patch_size)

all_embeddings = []
all_labels = []
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.encoder 
model.to(device)
model.eval()
for scan, label in dataloader:
    with torch.no_grad():
        #feats = model(scan.to(device))           # bottleneck
        feats = predict_sliding_window_return_logits(
            network=model,
            data=scan.to(device),
            patch_size=patch_size,
            mirror_axes=(0, 1, 2),   # nnU-Net
            regions_class_order=None,
            use_gaussian=True,
            pad_border_mode="constant",
            pad_kwargs={"constant_values": 0},
            all_in_gpu=True, #False,
            step_size=0.5,
            disable_tqdm=True,
            predict_fn=forward_encoder   # key trick
        )
        pooled = torch.mean(feats, dim=(2,3,4))  # global avg pool
        #todo: add sth else to get more features? recimo 95%% max and min?
    all_embeddings.append(pooled.cpu().numpy())
    all_labels.append(label)

print("Embeddings shape:", np.vstack(all_embeddings).shape)
print("embeddings before stack shape:", len(all_embeddings), all_embeddings[0].shape)
np.save(f"embeddings_{fold}.npy", np.vstack(all_embeddings))
np.save("labels.npy", np.array(all_labels))



#TODO: 
#v dataloaderju se dataloaer za loadat tele embeddinge napisi. Vsaj mislim, da to rabmo?
# Pa definiraj classifier za on top, 
# train classifier. 
#pripravi se cca 250L+250R podatkov za eval! Make sure da se valla_id ne prekriva z treningom!!
#clean the code for detate prepare also; bos se rabla verjetno...
#eval to see if we're good enough. 
