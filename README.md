# Deep Learning-Based Spike Sorting Method for Extracellular Recordings
Matlab implementation of [Deep Learning-Based Template Matching Spike Classification for Extracellular Recordings](https://doi.org/10.3390/app10010301).

## Outline
![슬라이드11](https://user-images.githubusercontent.com/79451613/216958190-57974de0-84b5-4472-b558-68d395833917.JPG)

## Dataset
![슬라이드5](https://user-images.githubusercontent.com/79451613/216957836-3be7821d-2374-48f5-9da9-d8bce4883b31.JPG)

## PCA
![슬라이드6](https://user-images.githubusercontent.com/79451613/216957434-a5bedff6-1943-445d-8f48-bd96e58d0302.JPG)
reduce 32d features into 2d

## K-means clustring
### Easy
![슬라이드9](https://user-images.githubusercontent.com/79451613/216957571-1a5a0e40-51aa-48d1-a344-8420ef952a02.JPG)

### Hard
![슬라이드10](https://user-images.githubusercontent.com/79451613/216957591-fc7ee6af-12e7-4b1b-ba54-770d0a603ba5.JPG)

## Select 10% of data nearest to cluster centers and train MLPs
![슬라이드12](https://user-images.githubusercontent.com/79451613/216958241-2100bffa-0832-4d4e-a71a-835a0c42995f.JPG
For this model, the input layer is a 32d layer where each dimension corresponds to the 32 sampling points of data. 256 nodes for hidden layer.
The number of layers and nodes were chosen heuristically. used hyperbolic tangent function for activation fuction


## Results
![슬라이드13](https://user-images.githubusercontent.com/79451613/216958271-e51d56
![슬라이드14](https://user-images.githubusercontent.com/79451613/216958278-921314eb-e769-4511-842c-d6170ad33624.JPG)
01-8f33-49bb-a9c0-c56b2efff75d.JPG)