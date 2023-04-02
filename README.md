# CMS GSoC Submission

## Common Task 1 (Electron/photon classification):
&emsp; **Binary classification** problem for **2 channel** images of Electron and Photons.

### Architecture:
&emsp; ResNet Architecture was implemented. The backbone of the network was made with ResNet50 architecture. Since the image was only 32X32, only 2 out of 5 blocks of the ResNet50 architecture were used.

### Optimiser:
&emsp; ADAM with Reduce Learning on Plateau. The model was made to train for 100 epochs. The checkpoints were collected from epoch no 22 since it had the highest Validation AUC.
### Result:
Validation **AUC: 0.8224**
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
| Loss | AUC | Learning Rate |
| --- | --- | --- |
| ![Loss graph (common II)](readme_images/common_II_Loss.png) | ![AUC graph (common II)](readme_images/common_II_AUC.png) | ![Learning Rate graph (common II)](readme_images/common_2_lr.png) |



## Specific Task (Vision Transformers):
&emsp; **Binary classification** problem for **2 channel** images of Electron and Photons


### Architecture:
&emsp; A vanilla Visual Image Transformer(ViT)  Architecture was used as the main backbone of the network. The two channels were separately fed into two different Vit networks simultaneously. Finally, the obtained features were concatenated and fed into a single classification layer with sigmoid activation. Contrary to the first(CNN) architecture here we find that feeding both the channels is useful. The ViT was more efficient when it was fed the two channels separately. The ViT performed better without a hidden layer. 


### Optimiser:
&emsp; ADAM with a cyclic learning rate policy was used. The model was made to train for 150 epochs. The learning rate policy was triangular with decreasing max learning rate. The checkpoints were collected from the last epoch. The validation AUC achieved from this model is 79.17.


### Result:
Validation **AUC: 0.7980**
| Loss | AUC | Learning Rate |
| --- | --- | --- |
| ![Loss graph (Specific Task)](readme_images/Vit_Loss.png) | ![AUC graph (Specific Task)](readme_images/Vit_AUC.png) | ![Loss graph (Specific Task)](readme_images/VIT_LR.png) |


## Discussions:
&emsp; Although the ViT architecture is the new State of the Art for image classification still we obtained a lesser AUC compared to the CNN. This is due to the fact that Visual Image Transformer was not pre-trained and the dataset was not sufficient to get State of the Art results form the ViT.
