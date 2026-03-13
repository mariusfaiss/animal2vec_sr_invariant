# animal2vec_sr_invariant

This repository contains the scripts and config files used for the paper "Bioacoustic raw-audio transformer is invariant to input sample rate", submitted for publication at Scientific reports. 

The code in this repository is based on the [animal2vec](https://github.com/livingingroups/animal2vec?tab=readme-ov-file) model. 
```
Schäfer-Zimmermann, J. C., Demartsev, V., Averly, B., Dhanjal-Adams, K., Duteil, M., Gall, G., Faiß, M., Johnson-Ulrich, L., Stowell, D., Manser, M., Roch, M. A. & Strandburg-Peshkin, A. (2025)
animal2vec and MeerKAT: A self-supervised transformer for rare-event raw audio input and a large-scale reference dataset for bioacoustics
Methods in Ecology and Evolution
```
The code was modified to enable training animal2vec on variable sample rate datasets and test its performance against training on upsampled datasets. This repository also contains the scripts that were used to curate and process the datasets used in the paper, as well as scripts for evaluating and visualizing animal2vecs performance.


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
