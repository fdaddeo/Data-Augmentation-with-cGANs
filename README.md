# Data Augmentation with cGAN

## Requirements

- `Pytorch`
- `Numpy`
- `Pandas`
- `Seaborn`
- `Sklearn`

## CGAN 

### Training

To train the CGAN is necessary to create a configuration file and then execute:

`python3 -m train_cgan --run_name <model name> --config <path/to/configuration/file.yaml>`

NOTE: examples of configuration file are contained in `config/cgan/` folder.

### Testing

To test the CGAN execute:

`python3 -m test_cgan --config <path/to/configuration/file.yaml>`

NOTE: for each model specify the same configuration file used during its training.

## Classificator 

### Training

To train and test the classificator it's required to create a configuration file and then execute:

`python3 -m train_classifier --run_name <model name> --config <path/to/configuration/file.yaml>`

NOTE: examples of configuration file are contained in `config/classifier/` folder.

### Testing

To test a classificator model execute:

`python3 -m test_classifier --model_name <model name> --config <path/to/configuration/file.yaml>`

NOTES: 
- For each model specify the same configuration file used during its training;
- As model name specify only the file name and do not write the complete path.

## Results 

### Graphs visualization

To visualize the obtained results, both for CGAN and classifier, execute:

`tensorboard --logdir=<path/to/tensorboard/folder/>`

### FID scores registered

Each model has been used to generate a new CIFAR10 dataset, that has been compared with the original one class by class through the FID score.
Here are the obtained values.

| RunName                              | AllDataset | Airplane | Automobile |  Bird   |   Cat   |   Deer  |   Dog   |   Frog  |  Horse  |   Ship  |  Truck  |
|:-------------------------------------|    :---:   |  :---:   |   :---:    |  :---:  |  :---:  |  :---:  |  :---:  |  :---:  |  :---:  |  :---:  |  :---:  |
| `basic_cGAN` - epoch30               | 25.0814    | 43.8033  | 51.2340    | 50.1187 | 35.8533 | 38.5345 | 50.2043 | 51.5834 | 50.0480 | 48.3488 | 51.9783 |
| `basic_cGAN` - epoch100              | 19.7900    | 39.9388  | 43.6565    | 45.3907 | 32.5473 | 33.6034 | 51.6767 | 35.6730 | 39.1974 | 36.5174 | 40.2165 |
| `basic_cGAN_BCEWithLogits` - epoch30 | 26.1962    | 46.6924  | 59.9631    | 53.5438 | 37.7538 | 47.6463 | 56.2596 | 41.2485 | 49.8263 | 43.2729 | 59.8866 |
| `basic_cGAN_BCEWithLogits` - epoch100| 18.3587    | 39.3256  | 47.1962    | 41.2120 | 28.7295 | 28.8009 | 42.4902 | 35.0227 | 35.2195 | 39.2759 | 43.8909 |
| `basic_cGAN_NOINIT` - epoch30        | 28.2961    | 49.6503  | 55.5690    | 52.5145 | 41.0350 | 48.4470 | 57.8165 | 49.0617 | 66.2571 | 42.2449 | 53.7113 |
| `basic_cGAN_NOINIT` - epoch100       | 20.6019    | 42.3565  | 48.9903    | 43.9808 | 35.1413 | 32.7063 | 50.8310 | 37.9150 | 46.3769 | 38.1915 | 45.6485 |
| `basic_cGAN_NONORM` - epoch30        | 54.6617    | 85.8362  | 83.3916    | 93.3328 | 65.6713 | 74.7721 | 80.4029 | 82.7151 | 92.3878 | 75.4047 | 77.9362 |
| `basic_cGAN_NONORM` - epoch100       | 33.9906    | 60.6132  | 62.9891    | 64.2217 | 47.2884 | 46.7887 | 61.3682 | 49.3307 | 62.6472 | 49.1161 | 57.0115 |
| `basic_cGAN_SGD` - epoch30           | 230.954    | 354.929  | 272.712    | 335.822 | 365.622 | 351.331 | 403.674 | 324.052 | 325.617 | 336.270 | 342.140 |
| `basic_cGAN_SGD` - epoch100          | 214.799    | 365.356  | 308.137    | 356.645 | 316.402 | 296.593 | 317.441 | 316.219 | 306.465 | 302.288 | 367.789 |
| `basic_cGAN_SmoothL1loss` - epoch30  | 25.2347    | 47.6879  | 47.8939    | 51.5360 | 38.2146 | 45.1380 | 56.9448 | 40.9715 | 56.9873 | 44.1011 | 51.9278 |
| `basic_cGAN_SmoothL1loss` - epoch100 | 17.5960    | 36.6824  | 44.6527    | 39.5682 | 27.6457 | 33.0219 | 43.7501 | 31.2807 | 35.8994 | 35.7174 | 44.7159 |
| `cGAN_SmoothL1loss_bs128` - epoch30  | 28.8892    | 50.3035  | 65.5648    | 53.7833 | 37.6564 | 45.9779 | 58.8280 | 45.3261 | 62.7498 | 51.2253 | 93.3664 |
| `cGAN_SmoothL1loss_bs128` - epoch100 | 18.4161    | 37.6183  | 52.4211    | 39.3867 | 30.0114 | 30.3955 | 42.6866 | 30.4660 | 39.6675 | 36.7077 | 84.9438 |
| `cGAN_SmoothL1loss_Adamax` - epoch30 | 56.9044    | 75.2484  | 105.268    | 85.8292 | 85.4688 | 86.5913 | 103.241 | 86.7148 | 128.169 | 64.7571 | 86.9369 |
| `cGAN_SmoothL1loss_Adamax` - epoch100| 29.5539    | 47.2033  | 83.6270    | 53.9002 | 45.3708 | 44.6316 | 57.7528 | 60.4991 | 59.0338 | 62.2606 | 66.2551 |
| `cGAN_SmoothL1loss_NAdam` - epoch30  | 26.9565    | 43.4741  | 53.5850    | 56.4188 | 41.4548 | 46.8531 | 64.0583 | 68.0111 | 59.6425 | 34.2241 | 53.8282 |
| `cGAN_SmoothL1loss_NAdam` - epoch100 | 20.2487    | 39.7137  | 44.5671    | 44.1415 | 32.1569 | 34.8685 | 47.4983 | 72.6950 | 40.1842 | 39.2248 | 43.1007 |
| `basic_wGAN` - epoch30               | 48.2562    | 68.4371  | 75.0996    | 71.9104 | 54.4870 | 83.9637 | 81.9094 | 84.2123 | 83.1763 | 54.3745 | 72.1841 |
| `basic_wGAN` - epoch100              | 28.1914    | 54.6635  | 58.3065    | 49.8910 | 38.9040 | 37.1997 | 52.3032 | 38.4660 | 42.6299 | 41.8378 | 53.9019 |
| `cGAN_64_bs128` - epoch30            | 204.184    | 222.695  | 228.378    | 349.715 | 283.556 | 274.376 | 255.650 | 265.553 | 235.954 | 249.086 | 310.055 |
| `cGAN_64_bs128` - epoch100           | 152.912    | 217.718  | 224.458    | 231.542 | 219.200 | 198.162 | 230.474 | 222.847 | 218.450 | 222.499 | 238.959 |
| `custom_cGAN` - epoch30              | 35.8927    | 58.0080  | 68.7365    | 69.9182 | 50.5328 | 72.1439 | 70.0219 | 68.9697 | 88.6795 | 62.6273 | 80.5999 |
| `custom_cGAN` - epoch100             | 26.6001    | 46.4214  | 52.4751    | 59.7820 | 50.5737 | 49.3311 | 83.9058 | 71.9584 | 51.5222 | 56.8851 | 52.5101 |
| `custom_wGAN` - epoch30              | 69.4120    | 93.0382  | 102.035    | 104.180 | 98.7037 | 92.2057 | 129.500 | 137.826 | 119.934 | 66.3337 | 94.8316 |
| `custom_wGAN` - epoch100             | 42.3130    | 62.3584  | 80.7449    | 79.4685 | 59.9742 | 69.1960 | 85.8635 | 61.7054 | 68.5446 | 54.6184 | 72.8860 |
| `custom_cGAN_SmoothL1loss` - epoch30 | 45.7718    | 61.5156  | 98.6816    | 68.1409 | 69.3301 | 61.7120 | 81.4490 | 69.4798 | 92.9643 | 56.3719 | 94.6059 |
| `custom_cGAN_SmoothL1loss` - epoch100| 18.4845    | 37.1320  | 40.4285    | 38.8271 | 31.3358 | 32.3157 | 47.3368 | 43.6842 | 35.2573 | 34.4679 | 40.3287 |
| `custom_cGAN_2` - epoch30            | 104.208    | 159.158  | 150.377    | 174.049 | 154.885 | 177.240 | 186.384 | 210.636 | 175.558 | 136.752 | 133.861 |
| `custom_cGAN_2` - epoch100           | 73.0790    | 111.615  | 184.046    | 175.774 | 139.551 | 151.019 | 128.772 | 141.181 | 174.255 | 126.151 | 210.390 |

### Accuracy registered

Each generated dataset has been used to perform a classification task. So, it has been performed a fine tuning last 5 epochs over a pretrained AlexNet. The training has been performed using the generated datasets both as training set and as augmentation of the original CIFAR10 training set, for which has been used the 70% of images. The reference value for the accuracy over the original model was 9.1 %.

Here are the obtained values.

#### No Augmentation

| RunName                                           | Accuracy |
|:--------------------------------------------------|  :---:   |
| `basic_cGAN` - epoch30                            | 57.75 %  |
| `basic_cGAN` - epoch100                           | 63.66 %  |
| `basic_cGAN_BCEWithLogits` - epoch30              | 57.36 %  |
| `basic_cGAN_BCEWithLogits` - epoch100             | 62.76 %  |
| `basic_cGAN_NOINIT` - epoch30                     | 55.90 %  |
| `basic_cGAN_NOINIT` - epoch100                    | 61.26 %  |
| `basic_cGAN_NONORM` - epoch30                     | 52.05 %  |
| `basic_cGAN_NONORM` - epoch100                    | 63.38 %  |
| `basic_cGAN_SGD` - epoch30                        |  9.86 %  |
| `basic_cGAN_SGD` - epoch100                       | 15.75 %  |
| `basic_cGAN_SmoothL1loss` - epoch30               | 55.27 %  |
| `basic_cGAN_SmoothL1loss` - epoch100              | 63.26 %  |
| `adamax_bs128_basic_cGAN_SmoothL1loss` - epoch30  | 54.93 %  |
| `adamax_bs128_basic_cGAN_SmoothL1loss` - epoch100 | 54.91 %  |
| `adamax_basic_cGAN_SmoothL1loss` - epoch30        | 54.92 %  |
| `adamax_basic_cGAN_SmoothL1loss` - epoch100       | 64.27 %  |
| `nadam_basic_cGAN_SmoothL1loss` - epoch30         | 53.20 %  |
| `nadam_basic_cGAN_SmoothL1loss` - epoch100        | 60.84 %  |
| `cGAN_SmoothL1loss_bs128` - epoch30               | 29.02 %  |
| `cGAN_SmoothL1loss_bs128` - epoch100              | 64.05 %  |
| `cGAN_SmoothL1loss_Adamax` - epoch30              | 24.39 %  |
| `cGAN_SmoothL1loss_Adamax` - epoch100             | 38.45 %  |
| `adamax_basic_cGAN_SmoothL1loss_NAdam` - epoch30  | 54.75 %  |
| `adamax_basic_cGAN_SmoothL1loss_NAdam` - epoch100 | 56.96 %  | 
| `basic_wGAN` - epoch30                            | 52.54 %  |
| `basic_wGAN` - epoch100                           | 66.28 %  |
| `cGAN_64_bs128` - epoch30                         | 16.97 %  |
| `cGAN_64_bs128` - epoch100                        | 26.91 %  |
| `custom_cGAN` - epoch30                           | 42.74 %  |
| `custom_cGAN` - epoch100                          | 57.42 %  |
| `custom_wGAN` - epoch30                           | 35.85 %  |
| `custom_wGAN` - epoch100                          | 54.04 %  |
| `custom_cGAN_SmoothL1loss` - epoch30              | 44.47 %  |
| `custom_cGAN_SmoothL1loss` - epoch100             | 62.13 %  |
| `custom_cGAN_2` - epoch30                         | 22.38 %  |
| `custom_cGAN_2` - epoch100                        | 24.77 %  |

#### Augmentation

| RunName                                           | Accuracy |
|:--------------------------------------------------|  :---:   |
| `basic_cGAN` - epoch30                            | 80.13 %  |
| `basic_cGAN` - epoch100                           | 78.99 %  |
| `basic_cGAN_BCEWithLogits` - epoch30              | 79.29 %  |
| `basic_cGAN_BCEWithLogits` - epoch100             | 78.66 %  |
| `basic_cGAN_NOINIT` - epoch30                     | 78.81 %  |
| `basic_cGAN_NOINIT` - epoch100                    | 79.85 %  |
| `basic_cGAN_NONORM` - epoch30                     | 79.77 %  |
| `basic_cGAN_NONORM` - epoch100                    | 80.48 %  |
| `basic_cGAN_SGD` - epoch30                        | 81.80 %  |
| `basic_cGAN_SGD` - epoch100                       | 81.39 %  |
| `basic_cGAN_SmoothL1loss` - epoch30               | 78.98 %  |
| `basic_cGAN_SmoothL1loss` - epoch100              | 80.08 %  |
| `adamax_basic_cGAN_SmoothL1loss` - epoch30        | 81.38 %  |
| `adamax_basic_cGAN_SmoothL1loss` - epoch100       | 81.50 %  |
| `nadam_basic_cGAN_SmoothL1loss` - epoch30         | 79.81 %  |
| `nadam_basic_cGAN_SmoothL1loss` - epoch100        | 78.83 %  |
| `adamax_bs128_basic_cGAN_SmoothL1loss` - epoch30  | 81.47 %  |
| `adamax_bs128_basic_cGAN_SmoothL1loss` - epoch100 | 81.59 %  |
| `cGAN_SmoothL1loss_bs128` - epoch30               | 80.35 %  |
| `cGAN_SmoothL1loss_bs128` - epoch100              | 79.46 %  |
| `cGAN_SmoothL1loss_Adamax` - epoch30              | 80.12 %  |
| `cGAN_SmoothL1loss_Adamax` - epoch100             | 79.78 %  |
| `adamax_basic_cGAN_SmoothL1loss_NAdam` - epoch30  | 81.68 %  |
| `adamax_basic_cGAN_SmoothL1loss_NAdam` - epoch100 | 81.91 %  |
| `basic_wGAN` - epoch30                            | 79.23 %  |
| `basic_wGAN` - epoch100                           | 80.30 %  |
| `cGAN_64_bs128` - epoch30                         | 81.11 %  |
| `cGAN_64_bs128` - epoch100                        | 80.87 %  |
| `custom_cGAN` - epoch30                           | 79.39 %  |
| `custom_cGAN` - epoch100                          | 79.91 %  |
| `custom_wGAN` - epoch30                           | 79.15 %  |
| `custom_wGAN` - epoch100                          | 79.86 %  |
| `custom_cGAN_SmoothL1loss` - epoch30              | 78.92 %  |
| `custom_cGAN_SmoothL1loss` - epoch100             | 79.61 %  |
| `custom_cGAN_2` - epoch30                         | 80.04 %  |
| `custom_cGAN_2` - epoch100                        | 80.91 %  |