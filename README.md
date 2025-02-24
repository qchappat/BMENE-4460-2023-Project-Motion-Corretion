# BMENE-4460-2023-Project-Motion-Corretion
BMENE 4460 (2023) Project by Zachary ABESSERA, Quentin CHAPPAT and Nikhil KUMAR KUPPA.

# Project description

This is the secret folder for the Project 'MRI Motion Detection and Correction with Complex-value AFT-Net'.

As the title already clearly illustrates, our project consists in detecting and correcting brain MRI scans of humans from the [fastMRI dataset](https://fastmri.med.nyu.edu/).
The added value of our project is that we use raw k-space data as input to correct MRI scans.

To do so, we first were able to simulate motion artefacts on our uncorrupted raw complex data by modifying the class `RandomMotion` of the package [TorchIO](https://torchio.readthedocs.io/transforms/augmentation.html#randommotion) that itself does not take complex images as input.

Then, we made a data augmentation of our data by 2 (creating for each raw-uncorrupted-k-space a simulate-corrupted-k-space) and used this complete dataset as training data for our model.

We made sure to keep our two classes (corrupted and uncorrupted) balanced in each of our fold (60% training, 20% validation, 20% testing).

We trained our models using Google Colaboratory Virtual Machines with NVIDIA A100 GPUs (12 GB VRAM).

We let most of the models trained until the end. Instead for the ResCUNet-AFT-ResCUNet models for each dataset that we stoped after 20 hours. Our total cumulated training time amount is about 180/200 hours.

You will find in this repository the weights from our different training and the results of our models in the folder `saved_model`.

We trained models on a mix of different datasets (T1 3T human MRI brain scans, T2 1.5 T human MRI brain scans and both combined).

After all the training, we analyzed the statistical significance of our results to prove that our models were able to learn features to correct motion artefacts. You can check the paper inside the folder `paper` for more information.

The improvements and interest of our project are that we take raw-k-space as input which means there is no preprocessing needed, no detection of movement need and no lost of data (like the phase).

The limitations of our models are that there in 2D and therefore our models has hard time correcting 3D types of motion. We think that if we were to use a 3D network, we could correct much more artefacts. If you are interested in performing this work, do not hesitate to contact qjc2002@columbia.edu (I am interested in working with this).

In this repository you will be able to find our report paper, the code (inside the folder `scripts`), the notebooks we used on Google Colaboratory and for the analysis, the results and weights of our models.

# Commands to train, evaluate and analyze the models

- [Prerequisites](#prerequisites)
- [Data selection](#data-selection)
- [Generate motion artefacts & data preprocessing](#generate-motion-artefacts--data-preprocessing)
- [Train AFT](#train-aft)
- [Train ResCUNet](#train-rescunet)
- [Train AFT\_ResCUNet](#train-aft_rescunet)
- [Train ResCUNet\_AFT](#train-rescunet_aft)
- [Train ResCUNet\_AFT\_ResCUNet](#train-rescunet_aft_rescunet)
- [Analysis](#analysis)

## Prerequisites

- Review the paper `paper/Motion-Correction.pdf`.
- Download FastMRI Brain MRI Multicoil dataset from https://fastmri.med.nyu.edu/
- Inside the `data` folder, put the folders `multicoil_train` and `multicoil_val` from the Google Drive.
- Access the packages `pytorch-complex` and `vision-complex` via the [AFT-Net GitHub]{https://github.com/yanting-yang/AFT-Net}.
- Create an Anaconda environment using `environment_requirements.yml`.

## Data selection

The data structure of each subject in the fastMRI dataset is (Slice, Channel, Height, Weight). 
As each subject varies in shape, acquisition type and system field strength, we should select and filter out the data we want.

The data we selected was as follows:

|                       | T2_1.5T | T1_3T    |
|-----------------------|---------|----------|
| acquisition           | AXT2    | AXT1PRE  |  
| slices                | 16      | 16       |  
| receiverChannels      | 4       | 4        |
| height                | 640     | 640      |
| weight                | 320     | 320      |
| systemFieldStrength_T | 1.494   | 2.8936   |
| number of subjects    | 692     | 109      |

## Generate motion artefacts & data preprocessing

You can use the notebooks inside the folder `Preprocessing` using Google Colaboratory or use the following commands in Anaconda on your computer (note the code is adapted for Linux, Windows and MacOS)

First generate the motion artefacts images:

```bash
python scripts/preprocessing/convert_data.py --data_dir data
```

Generate the meta infomation for all data:

```bash
python scripts/preprocessing/generate_meta_info.py --data_dir data --output_csv meta.csv
```

Filter out the selected data:

```bash
python scripts/preprocessing/filter_meta_info.py --input_csv meta.csv --output_csv T2_1.5T.csv --acquisition AXT2 --slices 16 --receiverChannels 4 --height 640 --width 320 --systemFieldStrength_T 1.494
python scripts/preprocessing/filter_meta_info.py --input_csv meta.csv --output_csv T1_3T.csv --acquisition AXT1PRE --slices 16 --receiverChannels 4 --height 640 --width 320 --systemFieldStrength_T 2.8936
```

Combine the selected data uniformly:

```bash
python scripts/preprocessing/uniform_concat_csv.py --csv_in T1_3T.csv T2_15T.csv --csv_out data_mix_T1_3T_T2_15T.csv
python scripts/preprocessing/uniform_concat_csv.py --csv_in T1_3T.csv --csv_out data_T1_3T.csv
python scripts/preprocessing/uniform_concat_csv.py --csv_in T2_15T.csv --csv_out data_T2_15T.csv
```

Append the max value of magnitude image to the `data.csv`:

```bash
python scripts/preprocessing/add_max_value.py --input_csv data_mix_T1_3T_T2_15T.csv --output_csv data_mix_T1_3T_T2_15T.csv
python scripts/preprocessing/add_max_value.py --input_csv data_T1_3T.csv --output_csv data_T1_3T.csv
python scripts/preprocessing/add_max_value.py --input_csv data_T2_15T.csv --output_csv data_T2_15T.csv
```

## Train AFT

Task: MRI image reconstruction

> The time for training epoch is fixed to 14 mins.
> The validation time can be (xtremely) long depending on your split. We would advise you to reduce the size of the validation set to 5/10%. You should not modify the max_len argument of the DataLoader because you want to keep the classes balanced.
> Use `--debug` for code testing.
> Adapt the different arguments of the command (--data-csv and --save-dir) depending on the dataset you want to train and test on.

```bash
python scripts/230401_Recon_AFT/train.py --data_csv data.csv --save_dir saved_model/230401_Recon_AFT --loss_func mse_loss --device cuda:0
python scripts/230401_Recon_AFT/eval.py --data_csv data.csv --save_dir saved_model/230401_Recon_AFT --device cuda:0
```

## Train ResCUNet

Task: MRI motion correction (image domain to image domain)

> The time for training epoch is fixed to 14 mins.
> The validation time can be (xtremely) long depending on your split. We would advise you to reduce the size of the validation set to 5/10%. You should not modify the max_len argument of the DataLoader because you want to keep the classes balanced.
> Use `--debug` for code testing.
> Adapt the different arguments of the command (--data-csv and --save-dir) depending on the dataset you want to train and test on.

```bash
python scripts/230401_Acc_ResCUNet_i2i/train.py --data_csv data_T1_3T.csv --save_dir saved_model/T1_3T/230401_Acc_ResCUNet_i2i --loss_func mse_loss --json_path data/data_to_original_dict.json --device cuda:0
python scripts/230401_Acc_ResCUNet_i2i/eval.py --data_csv data.csv --save_dir saved_model/230401_Acc_ResCUNet_i2i --json_path data/data_to_original_dict.json --device cuda:0
```

## Train AFT_ResCUNet

Task: MRI motion correction (sensor domain to image domain)

> The time for training epoch is fixed to 14 mins.
> The validation time can be (xtremely) long depending on your split. We would advise you to reduce the size of the validation set to 5/10%. You should not modify the max_len argument of the DataLoader because you want to keep the classes balanced.
> Use `--debug` for code testing.
> Adapt the different arguments of the command (--data-csv and --save-dir) depending on the dataset you want to train and test on.

```bash
python scripts/230401_Acc_AFT_ResCUNet/train.py --data_csv data.csv --save_dir saved_model/230401_Acc_AFT_ResCUNet --AFT saved_model/230401_Recon_AFT/weights.pt --ResCUNet saved_model/230401_Acc_ResCUNet_i2i/weights.pt --loss_func mse_loss --json_path data/data_to_original_dict.json --device cuda:0
python scripts/230401_Acc_AFT_ResCUNet/eval.py --data_csv data.csv --save_dir saved_model/230401_Acc_AFT_ResCUNet --json_path data/data_to_original_dict.json --device cuda:0
```

## Train ResCUNet_AFT

Task: MRI motion correction (sensor domain to image domain)

> The time for training epoch is fixed to 14 mins.
> The validation time can be (xtremely) long depending on your split. We would advise you to reduce the size of the validation set to 5/10%. You should not modify the max_len argument of the DataLoader because you want to keep the classes balanced.
> Use `--debug` for code testing.
> Adapt the different arguments of the command (--data-csv and --save-dir) depending on the dataset you want to train and test on.

```bash
python scripts/230401_Acc_ResCUNet_AFT/train.py --data_csv data.csv --save_dir saved_model/230401_Acc_ResCUNet_AFT --AFT saved_model/230401_Recon_AFT/weights.pt --loss_func mse_loss --json_path data/data_to_original_dict.json --device cuda:0
python scripts/230401_Acc_ResCUNet_AFT/eval.py --data_csv data.csv --save_dir saved_model/230401_Acc_ResCUNet_AFT --json_path data/data_to_original_dict.json --device cuda:0
```

## Train ResCUNet_AFT_ResCUNet

Task: MRI motion correction (sensor domain to image domain)

> The time for training epoch is fixed to 14 mins.
> The validation time can be (xtremely) long depending on your split. We would advise you to reduce the size of the validation set to 5/10%. You should not modify the max_len argument of the DataLoader because you want to keep the classes balanced.
> Use `--debug` for code testing.
> Adapt the different arguments of the command (--data-csv and --save-dir) depending on the dataset you want to train and test on.

```bash
python scripts/230401_Acc_ResCUNet_AFT_ResCUNet/train.py --data_csv data.csv --save_dir saved_model/230401_Acc_ResCUNet_AFT_ResCUNet --AFT saved_model/230401_Recon_AFT/weights.pt --ResCUNet saved_model/230401_Acc_ResCUNet_i2i/weights.pt --loss_func mse_loss --json_path data/data_to_original_dict.json --device cuda:0
python scripts/230401_Acc_ResCUNet_AFT_ResCUNet/eval.py --data_csv data.csv --save_dir saved_model/230401_Acc_ResCUNet_AFT_ResCUNet --json_path data/data_to_original_dict.json --device cuda:0 --debug
```

## Analysis

To perform the analysis of your results, you can use the notebook `plotting_and_stats.ipynb` in the folder `\Notebooks\Analysis`.
The notebook uses [spiketools](https://spiketools.github.io/spiketools/) to perform most of the analysis.
