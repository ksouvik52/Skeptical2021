A> This is the codebase for the skeptical students that can distill from a nasty teacher. This code base supports 
CIFAR-10 and CIFAR-100 dataset for ResNet18, ResNet50 and MobileNetV2 models.
B> The json files related to the hyperparameters of each model is present in corresponding sub-folders 
inside the experiments folder.

C> The experiments should run in the following order:
===================================================
1. run : run_scratch.py to first create a baseline model on a specific dataset. Change the save_path location accordingly.
2. run: run_nasty.py to run and generate a nasty teacher model for a specific model type on a specific dataset. change save_path location based on which ever model you wish to run on whichever dataset.
3. run: run_kd_from_nasty.py to run a skeptical student that learns from a nasty teacher that was generated in step 2. change save_path accordingly.  