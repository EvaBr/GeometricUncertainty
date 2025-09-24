import numpy as np
import pandas as pd
import nibabel as nib
import random
import pydicom
import dicom2nifti
from pathlib import Path
import subprocess


random.seed(13)

# Example: preprocess one CT scan
#TODO first: make a csv list of the scans we want to use; let's say 2000 for each dominance
# then move the biggest 3D ccta scan for those subjects into Results or some SSD, together with a csv of name, coronary dominance. 
csv = pd.read_csv("../../SCAPISdata/SCAPIS-DATA-PETITION-659-20241024.csv").set_index("Subject")
testvalas = [i.name.split(".")[0][2:] for i in Path("../dataSC/train").glob("*.nii.gz")]
testvalas += [i.name.split(".")[0][2:] for i in Path("../dataSC/test").glob("*.nii.gz")]
#let's not choose among already chosen
csv = csv[~csv.index.isin(testvalas)]
# Separate by dominance
left = csv[csv["Dominance"] == "LEFT"]
right = csv[csv["Dominance"] == "RIGHT"]

folder = "val"
nrSubj = 40 #100 #2000
# Sample subjects
left_samples = random.sample(list(left.index), nrSubj)
right_samples = random.sample(list(right.index), nrSubj)
#move them to a separate folder, for easier access. 
sites = {"Göteborg":1,"Malmö":2,"Stockholm":3,"Linköping":4,"Uppsala":5,"Umeå":6}
empty = [] #all that dont have a viable candidate for conversion
duplicates = []
datap = Path(f"/home/evabreznik/Skrivbord/GeomUncertainty/dataSC/{folder}")
datap.mkdir(parents=True, exist_ok=True)
Path("tempMapp").mkdir(exist_ok=True)

for subject, label in zip(left_samples + right_samples, ["LEFT"] * nrSubj + ["RIGHT"] * nrSubj):
    subprocess.run("rm -r tempMapp/*", shell=True)
    print(f"Processing subject {subject} with label {label}")
    #get all paths to subject:
    studies = list(Path(f"/media/Data2/CT-site-{sites[csv.loc[subject].Site]}-ccta/").glob(f"*/{subject}/*"))
   
    #check if studies are doubled; in that case move to folder
    if len(studies)!=len(set([i.name for i in studies])):
        duplicates.append(subject)
        for pot in studies:
            subprocess.run(f"cp -r {str(pot)} tempMapp/.", shell=True)
        studies = list(Path("tempMapp").glob("*"))

    #check if any study has good properties (dystolic and not truestack and slicethicknes==0.5)
    toconvert = []
    for study in studies:
        for series in study.glob("*"):
            slices = list(series.glob("*.dcm"))
            if len(slices)>250: #avoid small series
                head = pydicom.dcmread(slices[0])
                if ((head.SliceThickness=='0.5') and 
                    not ('truestack' in head.SeriesDescription.lower()) and
                    not ('syst' in head.SeriesDescription.lower())):
                    #we have a good candidate!
                    toconvert.append((series, len(slices)))

    if len(toconvert) == 0:
        empty.append(subject)
        continue
    #convert the one good candidate with most slices; if multiple, we just take the first.
    toconvert.sort(key=lambda x: x[1], reverse=True)
    Path(datap, f"{label[0]}_{subject}").mkdir(parents=True, exist_ok=True)
    dicom2nifti.convert_directory(str(toconvert[0][0]), str(Path(datap, f"{label[0]}_{subject}")))
    subprocess.run(f"mv {str(Path(datap, f'{label[0]}_{subject}/*.nii.gz'))} {str(Path(datap, f'{label[0]}_{subject}.nii.gz'))}", shell=True)
    subprocess.run(f"rm -d {str(Path(datap, f'{label[0]}_{subject}'))}", shell=True)
    #print(studies)
    
with open("empty.txt", "w") as f:
    for e in empty:
        f.write(e+"\n")
with open("duplicates.txt", "w") as f:
    for d in duplicates:
        f.write(d+"\n")

print("subjects with no viable candidate:", empty)
print("(missing L = {}, missing R = {})".format(len([i for i in empty if i in left_samples]), len([i for i in empty if i in right_samples])))



