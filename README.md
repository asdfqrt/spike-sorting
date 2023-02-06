# Deep Learning-Based Spike Sorting Method for Extracellular Recordings
Matlab implementation of [Deep Learning-Based Template Matching Spike Classification for Extracellular Recordings](https://doi.org/10.3390/app10010301).
This project was done in fall 2020 at Yonsei University MAI intership.

## Outline
![슬라이드11](https://user-images.githubusercontent.com/79451613/216958190-57974de0-84b5-4472-b558-68d395833917.JPG)

Proposed a deep learning-based spike sorting method on extracellular recordings from a single electrode that is
efficient, robust to noise, and accurate. In circumstances where labelled data does not exist, we created
pseudo-labels through principal component analysis and K-means clustering to be used for multi-layer
perceptron training and built high performing spike classification model

## Dataset
![슬라이드5](https://user-images.githubusercontent.com/79451613/216957836-3be7821d-2374-48f5-9da9-d8bce4883b31.JPG)

## PCA
![슬라이드6](https://user-images.githubusercontent.com/79451613/216957434-a5bedff6-1943-445d-8f48-bd96e58d0302.JPG)

reduce 32d features of data into 2d

## K-means clustring
### Easy
![슬라이드9](https://user-images.githubusercontent.com/79451613/216957571-1a5a0e40-51aa-48d1-a344-8420ef952a02.JPG)

### Hard
![슬라이드10](https://user-images.githubusercontent.com/79451613/216957591-fc7ee6af-12e7-4b1b-ba54-770d0a603ba5.JPG)

## Select 10% of data nearest to cluster centers and train MLPs
![슬라이드12](https://user-images.githubusercontent.com/79451613/216963170-caf1e820-f0e0-4515-a10a-9938b0c88676.JPG)

For this model, the input layer is a 32d layer where each dimension corresponds to the 32 sampling points of data. 256 nodes for hidden layer.
The number of layers and nodes were chosen heuristically, used hyperbolic tangent function for activation fuction


## Results
![슬라이드12](https://user-images.githubusercontent.com/79451613/216963456-e1243661-9d74-4a41-9103-b4e1d04f1eed.JPG)

![슬라이드13](https://user-images.githubusercontent.com/79451613/216963463-439f1744-05af-4fc2-97b4-cce502408f94.JPG)

![슬라이드14](https://user-images.githubusercontent.com/79451613/216963471-79b0ad05-da6d-4196-a36e-241fd637582a.JPG)
