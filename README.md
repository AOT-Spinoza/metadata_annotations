# metadata_annotations
video annotations by a variety of means
This framework is set up for the arrow of time project and is used for creating a diverse set of metadata annotations.
This readme will provide a necessary set up and usage tutorial of the framework.

The meta data annotations present in this moment are:
-Video Captioning
-Object Detection
-Action Detection   
-Action Classification
-Instance Segmentation
-Keypoint detection
-Depth map generation
-Semantic Segmentation

To use the model one has to set up an conda environment, I would do these things in order:
In an empty conda environment first download mamba.
Then you need to set up torch so that it works and utilizes the GPUS and that cuda is active. 
Use mamba also for this.
Then you can use the 