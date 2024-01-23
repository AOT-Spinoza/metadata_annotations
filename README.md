

### Metadata Annotations
Metadata annotations for video data by various means. This framework is set up for the "Arrow of Time" project and is used for creating a diverse set of metadata annotations. This README provides setup and usage instructions for the framework.

## Available Metadata Annotations
Video Captioning
Object Detection
Action Detection
Action Classification
Instance Segmentation
Keypoint Detection
Depth Map Generation
Semantic Segmentation

## Setup Steps
To use the pipeline, you need to set up a conda environment and follow these steps:

# Create an environment with mamba

```bash
conda create -n metadata mamba

conda activate metadata

```

# Install the CUDA toolkit corresponding to your CUDA version (e.g., 11.8).

```bash
mamba install cudatoolkit==11.8
```

# Install PyTorch and Dependencies:
Use Mamba to install PyTorch with the correct CUDA toolkit version.

```bash
mamba install cudatoolkit=11.8 pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
```

# Check if your cuda is correctly installed


```bash
python -c "import torch; print(torch.cuda.is_available())"
```

# Install Requirements:

In the root of the metadata_annotations repository, install the remaining requirements. Cuda should still be correctly installed after. Make sure you activated the environment and use update otherwise you can end up creating a nested environment.
```bash
mamba env update --file environment_metadata.yml
```

## Usage
Configure the framework in the config.yaml file.
You can run a model on your video by changing the input and output folders at the bottom of config.yaml.
You can comment out models that you don't want to use.
You can add models by modifying the config.yaml file.

Run the pipeline by:

```bash 
python scriptpipeline.py
```

## Adding Models
You can add models by following the structure in the config.yaml file:

tasks:
  $task_name:
    load_model:
      framework: torch
      model_loading_function: 'torch.hub.load'
      parameters_load_function:
        repo_or_dir: 'github_repo_or_directory'
        model: 'model_name'
        pretrained: True


The config.yaml file includes configurations for different machine learning tasks, model loading, preprocessing, postprocessing, and exporting options.

## Pipeline
The scriptpipeline.py shows the pipeline's execution. Follow the functions to understand how it handles different functions for different frameworks.

The load_models_and_config.py file demonstrates the goal of the final format for all other modalities in the pipeline (transformation/preprocessing, inference/predictions, postprocessing, exporting). It works dynamically, and you specify the model loading function and its parameters in the config.

Under the export section in the config, you can specify whether to export the results as video, CSV, or HDF5.

The files will be exported under the dir that you determined under "outputs" in the config.yaml near the bottom, make sure the dir exists.
The paths will be printed in the terminal so you can easily find them
## Notes
Ensure that the config.yaml file is properly configured for your specific use case.
Different tasks and models can be configured in the config.yaml file.
Follow the comments and documentation within the config.yaml file for detailed configuration options.

## To Do:

# The pipeline always finishes
The pipeline now stops when a model is not able to create predictions (action detection when no persons are present), this will be resolved asap.
Because it will do predictions for all videos and models in inputs and the config first and then starts exporting all. 
# Figuring out batch sizes to speed up computation:

# Resolve warning about depreciated functions, 
Some functions are being removed in upcoming updates and can easily be changed, these are the waarningsd at the beginning when you run the script
Some functions are just internally used and will be updated by packages themselves others are being called themselves in transformations.4

warnings to resolve per model:
GIT: Processing videos for GIT:/tank/tgn252/anaconda3/envs/metadata/lib/python3.9/site-packages/torchvision/io/video_reader.py:245: 
UserWarning: Accurate seek is not implemented for pyav backend
  warnings.warn("Accurate seek is not implemented for pyav backend")
/tank/tgn252/anaconda3/envs/metadata/lib/python3.9/site-packages/transformers/feature_extraction_utils.py:149: UserWarning: Creating a tensor from a list of numpy.ndarrays is extremely slow. Please consider converting the list to a single numpy.ndarray with numpy.array() before converting to a tensor. (Triggered internally at /opt/conda/conda-bld/pytorch_1695392022560/work/torch/csrc/utils/tensor_new.cpp:261.)


# Tracker Issues
The tracker needs to be upgraded maybe to DEEPSORT, or the parameters need to be studied so it performs better. OR something else entirely. 
Now it seems to track persons for example quite well but not overlay or hiding. which occurs quite often. Also ID numbers suddenly go from 1,2 to 13,2. But therefore the Sort needs to be studies better why it is doing that