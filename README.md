<h1 align=center> Exploring the underlying symmetries in particle physics with equivariant neural networks</h1>

## Common Task 1 (Electron/photon classification):
&emsp; **Binary classification** problem for **2 channel** images of Electron and Photons.

### Architecture:
&emsp; ResNet Architecture was implemented. The backbone of the network was made with ResNet50 architecture. Since the image was only 32X32, only 2 out of 5 blocks of the ResNet50 architecture were used.

### Optimiser:
&emsp; ADAM with Reduce Learning on Plateau. The model was made to train for 100 epochs. The checkpoints were collected from epoch no 22 since it had the highest Validation AUC.
### Result:
Validation **AUC: 0.8224**
We have trained the following models as a part of the evaluation test:-
- Pytorch
  * [![Notebook](https://img.shields.io/badge/Made%20with-Jupyter-orange?style=for-the-badge&logo=Jupyter)](https://github.com/dc250601/GSoC-2023-Evaluation-Test/blob/c9ec6f303f8e9b144ff9c8002a8325dca7b1a138/Common-I/Pytorch%20Training.ipynb)
- Tensorflow
  * [![Notebook](https://img.shields.io/badge/Made%20with-Jupyter-orange?style=for-the-badge&logo=Jupyter)](https://github.com/dc250601/GSoC-2023-Evaluation-Test/blob/c9ec6f303f8e9b144ff9c8002a8325dca7b1a138/Common-I/Tensorflow%20Training.ipynb)


| Loss | AUC |
| --- | --- |
| ![Loss graph (common I)](readme_images/Common_I_Loss.png) | ![AUC graph (common I)](readme_images/Common_I_Auc.png) |



## Common Task 2 (Deep Learning-based Quark-Gluon Classification):
&emsp; **Binary classification** problem for **3 channel** images of Quark and Gluons.

### Architecture:
&emsp; An efficient Architecture was used as the main backbone of the network followed by the Global Average Pooling layer and Fully Connected Layers along with Dropout layers.

### Optimiser:
&emsp; ADAM with a cyclic learning rate policy was used. The model was made to train for 200 epochs. The learning rate policy was triangular with decreasing max learning rate. The checkpoints were collected from epoch no 186 as it had the maximum validation AUC score of 0.7980.

### Result:
Validation **AUC: 0.7980**
We have trained the following models as a part of the evaluation test:-
- Efficient Net architecture
  * [![Notebook](https://img.shields.io/badge/Made%20with-Jupyter-orange?style=for-the-badge&logo=Jupyter)](https://github.com/dc250601/GSoC-2023-Evaluation-Test/blob/c9ec6f303f8e9b144ff9c8002a8325dca7b1a138/Common%20-%20II/Training.ipynb)

| Loss | AUC | Learning Rate |
| --- | --- | --- |
| ![Loss graph (common II)](readme_images/common_II_Loss.png) | ![AUC graph (common II)](readme_images/common_II_AUC.png) | ![Learning Rate graph (common II)](readme_images/common_2_lr.png) |



## Equivariant Neural Networks:
&emsp; **Binary classification** problem for **3 channel** images of Quark and Gluons.


### Basic Intuition:
&emsp; The basic idea of using Equivariant Neural Network in the place of Non-Equivariant ones are to preserve the Equivariance of the data and build our models with the inductive bias that certain symmetries exists in our data. Conventional Convoltutional Neural Networks are Translation equivariant due to the sliding nature of the convolutional kernels. Often in many real life examples a number of symmetries exists in the data such as rotational, scaling, etc. To encode these type of equivariances in our model we use Equivariant Neural Networks. The principle of weight sharing across different axis of equivariance helps the model to generalize better than Vanilla Convolutional Neural Networks.

### Architecture:
&emsp; We train a number of Regular Group Convolutional Neural Networks (G-CNN) along with a traditional CNN to understand the nature and performance of the G-CNNs with respect to the CNNs. We build our Regular G-CNNs on the basis of these two groups:-
* Cyclic Groups (C<sub>n</sub>): The group of all rotations generated by the generator of C<sub>n</sub>
* Dihedral group (D<sub>n</sub>): The group of symmetries of a regular polygon which includes rotation and reflections.</br>

We have trained the following models as a part of the evaluation test:-
- C<sub>4</sub>: A 19 layer deep network with C<sub>4</sub> equivariance
  * [![Notebook](https://img.shields.io/badge/Made%20with-Jupyter-orange?style=for-the-badge&logo=Jupyter)](./Equivariant/C4.ipynb)
  * [![WandB](https://img.shields.io/badge/Weights_&_Biases-FFBE00?style=for-the-badge&logo=WeightsAndBiases&logoColor=white)](https://wandb.ai/dc250601/Equivariant/runs/h34bera8?workspace=user-dc250601)
- C<sub>8</sub>: A 19 layer deep network with C<sub>8</sub> equivariance
  * [![Notebook](https://img.shields.io/badge/Made%20with-Jupyter-orange?style=for-the-badge&logo=Jupyter)](./Equivariant/C8.ipynb)
  * [![WandB](https://img.shields.io/badge/Weights_&_Biases-FFBE00?style=for-the-badge&logo=WeightsAndBiases&logoColor=white)](https://wandb.ai/dc250601/Equivariant/runs/h7k0lrqi?workspace=user-dc250601)
- D<sub>4</sub>: A 19 layer deep network with D<sub>4</sub> equivariance
  * [![Notebook](https://img.shields.io/badge/Made%20with-Jupyter-orange?style=for-the-badge&logo=Jupyter)](./Equivariant/D4.ipynb)
  * [![WandB](https://img.shields.io/badge/Weights_&_Biases-FFBE00?style=for-the-badge&logo=WeightsAndBiases&logoColor=white)](https://wandb.ai/dc250601/Equivariant/runs/8hur7zps?workspace=user-dc250601)
- D<sub>8</sub>: A 19 layer deep network with D<sub>8</sub> equivariance
  * [![Notebook](https://img.shields.io/badge/Made%20with-Jupyter-orange?style=for-the-badge&logo=Jupyter)](./Equivariant/D8.ipynb)
  * [![WandB](https://img.shields.io/badge/Weights_&_Biases-FFBE00?style=for-the-badge&logo=WeightsAndBiases&logoColor=white)](https://wandb.ai/dc250601/Equivariant/runs/azixgk4c?workspace=user-dc250601)
- C<sub>4</sub>_lite: A lighter version(Parameters reduced) of the C<sub>4</sub> network
  * [![Notebook](https://img.shields.io/badge/Made%20with-Jupyter-orange?style=for-the-badge&logo=Jupyter)](./Equivariant/C4_lite.ipynb)
  * [![WandB](https://img.shields.io/badge/Weights_&_Biases-FFBE00?style=for-the-badge&logo=WeightsAndBiases&logoColor=white)](https://wandb.ai/dc250601/Equivariant/runs/hqauau5y?workspace=user-dc250601)
- C<sub>8</sub>_lite: A lighter version(Parameters reduced) of the C<sub>8</sub> network
  * [![Notebook](https://img.shields.io/badge/Made%20with-Jupyter-orange?style=for-the-badge&logo=Jupyter)](./Equivariant/C8_lite.ipynb)
  * [![WandB](https://img.shields.io/badge/Weights_&_Biases-FFBE00?style=for-the-badge&logo=WeightsAndBiases&logoColor=white)](https://wandb.ai/dc250601/Equivariant/runs/w3gdhrx1?workspace=user-dc250601)
- D<sub>4</sub>_lite: A lighter version(Parameters reduced) of the D<sub>4</sub> network
  * [![Notebook](https://img.shields.io/badge/Made%20with-Jupyter-orange?style=for-the-badge&logo=Jupyter)](./Equivariant/D4_lite.ipynb)
  * [![WandB](https://img.shields.io/badge/Weights_&_Biases-FFBE00?style=for-the-badge&logo=WeightsAndBiases&logoColor=white)](https://wandb.ai/dc250601/Equivariant/runs/w3gdhrx1?workspace=user-dc250601)
- D<sub>8</sub>_lite: A lighter version(Parameters reduced) of the D<sub>8</sub> network
  * [![Notebook](https://img.shields.io/badge/Made%20with-Jupyter-orange?style=for-the-badge&logo=Jupyter)](./Equivariant/D8_lite.ipynb)
  * [![WandB](https://img.shields.io/badge/Weights_&_Biases-FFBE00?style=for-the-badge&logo=WeightsAndBiases&logoColor=white)](https://wandb.ai/dc250601/Equivariant/runs/5x8db3cw?workspace=user-dc250601)
- Non_Equivariant: A 19 layer deep network with similar architecture.
  * [![Notebook](https://img.shields.io/badge/Made%20with-Jupyter-orange?style=for-the-badge&logo=Jupyter)](./Equivariant/Non_equivariant.ipynb)
  * [![WandB](https://img.shields.io/badge/Weights_&_Biases-FFBE00?style=for-the-badge&logo=WeightsAndBiases&logoColor=white)](https://wandb.ai/dc250601/Equivariant/runs/bc417kut?workspace=user-dc250601)



### Optimiser:
&emsp; ADAM with LR_Reduce_On_Plateau policy was used to train the networks with an initial learning rate of 1e-3.


### Result:
| Model 	| Peak-AUC 	| Model 	| Peak-AUC 	|
|:---:	|:---:	|:---:	|:---:	|
| C4 	| 79.29 	| C4_lite 	| 79.34 	|
| C8 	| 79.52 	| C8_lite 	| 79.34 	|
| D4 	| 79.54 	| D4_lite 	| 79.41 	|
| D8 	| 79.52 	| D8_lite 	| 79.46 	|
| CNN 	| 79.51 	|  	|  	|


## Discussions:
&emsp; The Particle Images in reality is SE(2) Equivariant by to keep things simple we used the C<sub>n</sub> and D<sub>n</sub> groups. Since convolution operation is translational equivariant we can use C<sub>n</sub> and D<sub>n</sub> with large values of n to mimic SE(2) equivariance. </br> Keeping the number of layers and channels fixed we see that we get better performance on increasing the Equivariance level in our network. D<sub>n</sub> has a much greater degree of symmetry in comparison to C<sub>n</sub> and hence we see better performance with D<sub>n</sub>. Moreover as we increase n the AUC scores go up suggesting that Equivariance is indeed helping. The Equivariant networks have slightly greater performance than the Non-Equivariant ones. The margin of imporvement should increase with a larger dataset sicne most of the Equivariant networks tend to overfit due to higher learning capacity than CNNs.
