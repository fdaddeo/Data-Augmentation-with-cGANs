# Data Augmentation with cGAN

Project on going. TODO

## FID scores registered

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
| `basic_wGAN` - epoch30               | 48.2562    | 68.4371  | 75.0996    | 71.9104 | 54.4870 | 83.9637 | 81.9094 | 84.2123 | 83.1763 | 54.3745 | 72.1841 |
| `basic_wGAN` - epoch100              | 28.1914    | 54.6635  | 58.3065    | 49.8910 | 38.9040 | 37.1997 | 52.3032 | 38.4660 | 42.6299 | 41.8378 | 53.9019 |

## Accuracy registered

Each generated dataset has been used to perform a classification task. So, it has been performed a fine tuning last 5 epochs over a pretrained AlexNet. The training has been performed usign the generated datasets both as training set and as an augmentation of the original CIFAR10 training set, for which has been used the 70% of images.

Here are the obtained values.

### No Augmentation

| RunName                              | Accuracy |
|:-------------------------------------|  :---:   |
| `basic_cGAN` - epoch30               | 57.75 %  |
| `basic_cGAN` - epoch100              | 63.66 %  |
| `basic_cGAN_BCEWithLogits` - epoch30 | 57.36 %  |
| `basic_cGAN_BCEWithLogits` - epoch100| 62.76 %  |
| `basic_cGAN_NOINIT` - epoch30        | 55.90 %  |
| `basic_cGAN_NOINIT` - epoch100       | 61.26 %  |
| `basic_cGAN_NONORM` - epoch30        | 52.05 %  |
| `basic_cGAN_NONORM` - epoch100       | 63.38 %  |
| `basic_cGAN_SGD` - epoch30           |  9.86 %  |
| `basic_cGAN_SGD` - epoch100          | 15.75 %  |
| `basic_cGAN_SmoothL1loss` - epoch30  | 55.27 %  |
| `basic_cGAN_SmoothL1loss` - epoch100 | 63.26 %  |
| `basic_wGAN` - epoch30               | 52.54 %  |
| `basic_wGAN` - epoch100              | 66.28 %  |

### Augmentation

| RunName                              | Accuracy |
|:-------------------------------------|  :---:   |
| `basic_cGAN` - epoch30               | 80.13 %  |
| `basic_cGAN` - epoch100              | 78.99 %  |
| `basic_cGAN_BCEWithLogits` - epoch30 | 79.29 %  |
| `basic_cGAN_BCEWithLogits` - epoch100| 78.66 %  |
| `basic_cGAN_NOINIT` - epoch30        | 78.81 %  |
| `basic_cGAN_NOINIT` - epoch100       | 79.85 %  |
| `basic_cGAN_NONORM` - epoch30        | 79.77 %  |
| `basic_cGAN_NONORM` - epoch100       | 80.48 %  |
| `basic_cGAN_SGD` - epoch30           | 81.80 %  |
| `basic_cGAN_SGD` - epoch100          | 81.39 %  |
| `basic_cGAN_SmoothL1loss` - epoch30  | 78.98 %  |
| `basic_cGAN_SmoothL1loss` - epoch100 | 80.08 %  |
| `basic_wGAN` - epoch30               | 79.23 %  |
| `basic_wGAN` - epoch100              | 80.30 %  |