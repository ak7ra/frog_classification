# Frog Call Classification with Convolutional Neural Network
**Authors:** Ami Kano, Kate Meldrum, Tyler Valentine <br />

## Abstract
Amphibian species are facing increasing rates of extinction. In an effort to better monitor the changing population levels and understand how to help at-risk species, researchers have relied on automated audio recording devices placed throughout their habitats. These devices produce thousands of audio files and are time-consuming for researchers to parse through manually. Automated software for audio classification has been applied to problems such as speech or music recognition for over a decade; however, applications for conservation efforts are more recent. Previous studies have shown success with applying convolutional neural networks (CNNs) for classifying audio of bird species. This work focuses on classification of six frog species. Our results indicate high prediction accuracy with our test set and an ability of the model to generalize to new, long-form audio files. 

## Data
The data for this project were collected in the Yasuni Rainforest. The audio files ranged from 2-20 minutes and the species in each audio file was identified and annotated by researchers from the Museo de Zoología QCAZ. The original dataset contained ten species, each with at least 39 call examples. We limited this project to the six species that had at least 100 frog calls represented in the audio files in order to ensure that we would have enough training data. 

### Location of Data
https://data.mendeley.com/datasets/5j852hzfjs/1

### Citation
QCAZ, Museo de Zoología; Estrella Terneux, Andrés; Nicolalde, Damián; Nicolalde, Daniel; Padilla, Samael (2019), “Labeled frog-call dataset of Yasuní National Park for training Machine Learning algorithms”, Mendeley Data, V1, doi: 10.17632/5j852hzfjs.1

## Metadata
**Authors:** Ami Kano, Kate Meldrum, Tyler Valentine <br />
**GitHub Username:** ak7ra, meldrumkathryn, Tyv132 <br />
**Project Name:** Frog Call Classification with Convolutional Neural Network

## Synopsis

### Required Python Packages

To run the files within this repository, you must have these Python Packages installed:

* `pandas`
* `numpy`
* `librosa`
* `simpleaudio`
* `IPython`
* `ipywidgets`
* `scipy.io`
* `matplotlib`
* `matplotlib.pyplot`
* `torch`
* `sklearn`
* `random`
* `timeit`
* `shutil`
* `os`
