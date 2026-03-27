# animal2vec_sr_invariant

This repository contains the scripts and config files used for the paper "Bioacoustic raw-audio transformer is invariant to input sample rate", submitted for publication at Scientific reports. 

The code in this repository is based on the [animal2vec](https://github.com/livingingroups/animal2vec?tab=readme-ov-file) model. 
```
Schäfer-Zimmermann, J. C., Demartsev, V., Averly, B., Dhanjal-Adams, K., Duteil, M., Gall, G., Faiß, M., Johnson-Ulrich, L., Stowell, D., Manser, M., Roch, M. A. & Strandburg-Peshkin, A. (2025)
animal2vec and MeerKAT: A self-supervised transformer for rare-event raw audio input and a large-scale reference dataset for bioacoustics
Methods in Ecology and Evolution
```
The code was modified to enable training animal2vec on variable sample rate datasets and test its performance against training on upsampled datasets. This repository also contains the scripts that were used to curate and process the [dataset](https://doi.org/10.17617/3.DOHL0O) used in the paper, as well as scripts for evaluating and visualizing animal2vecs performance.


## Installation
### Install Dependencies

- Clone the repo self-supervised-animal-vocalizations and install dependencies:
    - Clone the repo:
    - ```git clone https://github.com/mariusfaiss/animal2vec_sr_invariant```
    - Create a conda environment:
        - ```conda create --name ML_inference```
        - ```conda activate ML_inference```
        - ```conda install pip```
- Now switch to the repo directory and install dependencies
    - ```cd self-supervised-animal-vocalizations ```
    - ```pip install -r requirements.txt```

Wait until that completes.


Then some manual installation is needed:
- Clone the fairseq repo into some directory other than our repo directory:
    - ```cd ~```
    - ```git clone https://github.com/pytorch/fairseq```
    - ```cd fairseq```
    - ```pip install --editable ./```


- **Only if you get an libcublasLt.so.11 error** You might need to uninstall a cuda library that was installed during the pytorch install, but conflicts with your active cuda driver set. See [here](https://stackoverflow.com/questions/74394695/how-does-one-fix-when-torch-cant-find-cuda-error-version-libcublaslt-so-11-no).
    - ```pip uninstall nvidia_cublas_cu11```

**Now you are good to go and you can use the repo.**

## [Dataset upsampling](https://github.com/mariusfaiss/animal2vec_sr_invariant/tree/main/scripts/upsample_dataset)
This folder contains the scripts that were used to upsample the [source dataset](https://doi.org/10.17617/3.DOHL0O) to 48 kHz. 

### [upsample_dataset.py](https://github.com/mariusfaiss/animal2vec_sr_invariant/blob/main/scripts/upsample_dataset/upsample_dataset.py)
This script upsamples the audio files to a shared sample rate. It's input arguments are:
- --input_dir: The folder containing the audio files in the source dataset: SR_MeerKAT_XCbirds_10s/wav
- --output_dir: The directory holding the upsampled files

## [upsample_lbl_files.py](https://github.com/mariusfaiss/animal2vec_sr_invariant/blob/main/scripts/upsample_dataset/upsample_lbl_files.py)
This script updates the start and end audio frame values in the label files to match the upsampled resolution. It's input arguments are:
- --base_folder: The base folder of the source dataset SR_MeerKAT_XCbirds_10s
- --out_folder: The base folder of the upsampled dataset 48_MeerKAT_XCbirds_10s

## [update_tsv_frames.py](https://github.com/mariusfaiss/animal2vec_sr_invariant/blob/main/scripts/upsample_dataset/update_tsv_frames.py)
This script updates the manifest files. All file extensions are changed to .wav and the numbers of audio frames are updated to match the new sample rate. It's input arguments are:
- --base_dir: The base folder of the source dataset SR_MeerKAT_XCbirds_10s
- --out_dir: The base folder of the upsampled dataset 48_MeerKAT_XCbirds_10s


