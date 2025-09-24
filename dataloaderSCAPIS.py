import torch
import nibabel as nib
import numpy as np
from nnunetv2.preprocessing.preprocessors.default_preprocessor import DefaultPreprocessor
from batchgenerators.utilities.file_and_folder_operations import load_json
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager # ConfigurationManager


# Load configuration and plans
rootmapp = Path("/home/evabreznik/Skrivbord/MAIASTUFF/nnunet_data")
#fold_folder = "/home/evabreznik/Skrivbord/MAIASTUFF/nnunet_data/nnUNet_results/Dataset666_ASOCA/nnUNetTrainer__nnUNetResEncUNetMPlans__3d_fullres"
#plans = load_json(f"{fold_folder}/plans.json")
plans_folder = Path(rootmapp,"nnUNet_preprocessed/Dataset666_ASOCA/")
plans = load_json(Path(plans_folder,"nnUNetResEncUNetMPlans.json"))
dataset_json_file = str(Path(plans_folder, "dataset.json"))

# Initialize preprocessor
configuration = '3d_fullres'
plans_manager = PlansManager(plans)
preprocessor = DefaultPreprocessor()

class SCAPISDataset(Dataset):
    def __init__(self, img_path="dataSC"):
        self.imgs_path = img_path
        self.data = []
        
        datalist = list(Path(self.imgs_path).glob("*.nii.gz"))
        for sample in datalist:
            class_name = sample.name[0]
            self.data.append([str(sample), class_name])
        self.class_map = {"L" : 0, "R": 1}

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img_path, class_name = self.data[idx]
        print(f"Loading image: {img_path}, class: {class_name}")
        data, _, properties = preprocessor.run_case([img_path,], seg_file=None, plans_manager=plans_manager,
                                      configuration_manager=plans_manager.get_configuration(configuration),
                                      dataset_json=dataset_json_file)

        class_id = self.class_map[class_name]
        img_tensor = torch.from_numpy(data)
     #   print(img_tensor.shape)
        class_id = torch.tensor([class_id])
        return img_tensor, class_id


class EmbeddingDataset(Dataset):
    def __init__(self, embeddings_path, labels_path):
        self.embeddings = np.load(embeddings_path)
        self.labels = np.load(labels_path)
        assert len(self.embeddings) == len(self.labels), "Embeddings and labels must have the same length."

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, idx):
        embedding = torch.tensor(self.embeddings[idx], dtype=torch.float32)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return embedding, label