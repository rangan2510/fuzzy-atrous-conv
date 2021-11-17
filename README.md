# Fuzzy Atrous Convolutions for Covid-19 screening through Chest X-Rays

## Checking with your own images

### Using the notebook

To run your own image through our model, open the notebook named "train.ipynb". After that, please run all cells in order till cell 12, followed by cells 20 to 23. In cell 22, please assign the path of your image to the variable named "img_path", before running it. The output heatmap image is saved as "gradcam1.png" in the present working directory.

### Using the python script

To run your own image through our model, run the python script named "Funnelnet.py" and enter the path of your image as input. The predicted class is printed and the output heatmap image is saved as "gradcam1.png" in the present working directory.

## Verifying results

To verify our results,please download the dataset from kaggle, at the URL: https://www.kaggle.com/andyczhao/covidx-cxr2, making sure the path of the train folder relative to the notebook is as follows: "../input/covidx-cxr2/train". Then, open the notebook named "train.ipynb". After that, please run all cells in order till cell 19 (skipping cell 16, where we train the model). 


To train the model from scratch, please follow the above steps, except for skipping cell 14, where we load the pretrained weights, instead of cell 16. 