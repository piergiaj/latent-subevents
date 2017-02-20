================================================================================
# Learning Latent Sub-events in Activity Videos Using Temporal Attention Filters

This repository contains the code for our [AAAI 2017 paper](https://arxiv.org/abs/1605.08140):

    AJ Piergiovanni, Chenyou Fan and Michael S Ryoo
    "Learning Latent Sub-events in Activity Videos Using Temporal Attention Filters"
    in Proc. AAAI 2017

If you find the code useful for your research, please cite our paper:

        @inproceedings{piergiovanni2016learning,
            title={Learning Latent Sub-events in Activity Videos Using Temporal Attention Filters},
            author={Piergiovanni, AJ and Fan, Chenyou and Ryoo, Michael S},
            booktitle={Proceedings of the Thirty-First {AAAI} Conference on Artificial Intelligence},
            year={2016}
        }


# Temporal Attention Filters
The core of our approach, the temporal attention filters can be found in [temporal_attention.py](layers/temporal_attention.py). This file contains all the code to create and apply the attention filters. We provide several different models we tested, such as the [baseline_model.py](baseline_model.py) which applies either max, mean or sum pooling over the input features. We have the [temporal_pyramid_model.py](temporal_pyramid_model.py) which applies the unlearned pyramid of filters, [temporal_lstm_model.py](temporal_lstm_model.py) which create a model to dynamically adjust the filters with an LSTM. Our best performing model, [binary_learned_model.py](binary_learned_model.py) learns a set of attention filters for each activity class.

# Activity Classification Experiments
This code is for the activity classification task. We are able to learn latent sub-events with only activity labels, no labels for the sub-events are given. Our model extract per-frame CNN features and learns a set of temporal attention filters on those features. Each filter corresponds to a unique sub-event and our code takes advantage of these sub-events for improved performance on the recognition of activities.
![model](/examples/model.png)

We tested our models on both the [DogCentric](http://robotics.ait.kyushu-u.ac.jp/~yumi/db/first_dog.html) dataset as well as the [HMDB](http://serre-lab.clps.brown.edu/resource/hmdb-a-large-human-motion-database/) dataset.


# Example Learned Sub-events
We trained our model on the [HMDB](http://serre-lab.clps.brown.edu/resource/hmdb-a-large-human-motion-database/) dataset which contains ~7000 videos of 51 different human activities. Our model learned 3 sub-events for each activity. Here are some of the sub-events our model learned.

For the pushup activity, our model learned two key sub-events: the "moving down" sub-event (frames 18 to 26) and the "pushing up" sub-event (frames 48 to 55).

![going down](/examples/down.gif?raw=true "Going down Sub-event")
![pushing up](/examples/up.gif?raw=true "Pushing up Sub-event")

The somersault activity, shown here, is a more complex action where a person rotates over their feet:
![somersault](/examples/somersault.gif?raw=true "Somersault Activity")

Our model learned 3 sub-events. Two focused on the intervals where the person is upside-down (frames 31 to 51 and frames 40 to 48).

![sub-event 1](/examples/subevent1.gif?raw=true "Sub-event1")
![sub-event 2](/examples/subevent2.gif?raw=true "Sub-event 2")

The third sub-event focuses on the person standing up after completing the flip (frames 54 to 60).
![sub-event 3](/examples/subevent3.gif?raw=true "Sub-event 3")


================================================================================


# Requirements

Our code has been tested on Ubuntu 14.04 and 16.04 using the [Theano](https://github.com/Theano/Theano) version 0.9.0dev4 (but will likely work with other versions) with a Titan X GPU. We also rely on [Fuel](https://github.com/mila-udem/fuel) to help with the HDF5 datasets.


# Setup

1. Download the code ```git clone https://github.com/piergiaj/latent-subevents.git```

2. Extract features from your dataset. These can be any per-frame feature, such as [VGG](https://gist.github.com/baraldilorenzo/07d7802847aaad0a35d3), or per-segment features, such as [C3D](https://gist.github.com/albertomontesg/d8b21a179c1e6cca0480ebdf292c34d2) applied over many segments, or [ITF](https://lear.inrialpes.fr/people/wang/improved_trajectories) features. We relied on pretrained models for this step. 

3. Once the features have been extracted, we provide an example script to create an HDF5 file to train the models. create_hmdb_dataset.py shows how we did this. The way we store the features and load them with the uniform_dataset.py file allows us to properly apply masking to videos of different lengths in the same batch.

4. We provide several example scripts to train the models. train_tdd_hmdb.py trains the models for attention filters **shared for all classes**. This script will also output the performance of the model and save the final model. train_tdd_binary.py trains binary classifiers for **each activity** learns a set of attention filters.

5. Once all the binary classifiers have been trained, the test_all.py script provides an example to load the trained binary classifiers and run on the test set. The results are saved and the get_accuracy.py script combines those results to create the final predictions.

