<p align="center"><img width="30%" src="/Fig/neurips_logo.png"></p><br/> 

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)


Welcome to the official repo of the `NeurIPS 2021` paper **`Analyzing the Confidentiality of Undistillable Teachers in Knowledge Distillation`**.

**This repo currently contains the test codes. Training code will be updated soon!**

### Authors:
1. **Souvik Kundu** (souvikku@usc.edu)
2. Qirui Sun (qiruisun@usc.edu)
3. Yao Fu (yaof@usc.edu)
4. Massoud Pedram (pedram@usc.edu)
5. Peter A. Beerel (pabeerel@usc.edu)

### Abstract:
Knowledge distillation (KD) has recently been identified as a method that can unintentionally leak private information regarding the details of a teacher model toan unauthorized student. Recent research in developing undistillable nasty teachersthat can protect model confidentiality has gained significant attention. However,the level of protection these nasty models offer has been largely untested. In thispaper, we show that transferring knowledge to a shallow sub-section of a studentcan largely reduce a teacher’s influence.  By exploring the depth of the shallow subsection, we then present a distillation technique that enables a skeptical student model to learn even from a nasty teacher. To evaluate the efficacy of our skeptical students, we conducted experiments with several models with standard KD on both training data-available and data-free scenarios for various datasets. Compared to the normal student models, our skeptical students consistently provide superior classification performance of up to ∼59.5% and ∼5.8% in the presence (data-available) and absence (data-free) of the training data, respectively.  Moreover,similar to normal students, skeptical students maintain high classification accuracy when distilled from a normal teacher, showing their efficacy irrespective of theteacher being nasty or not. We believe the ability of skeptical students to largely diminish the KD-immunity of a potentially nasty teacher will motivate the research community to create more robust mechanisms for model confidentiality.

### To use the repo:
A> This is the codebase for the skeptical students that can distill from a nasty teacher. This code base supports 
CIFAR-10 and CIFAR-100 dataset for ResNet18, ResNet50 and MobileNetV2 models.

B> The json files related to the hyperparameters of each model is present in corresponding sub-folders 
inside the experiments folder.

C> The experiments should run in the following order:
1. run : run_scratch.py to first create a baseline model on a specific dataset. Change the save_path location accordingly.
2. run: run_nasty.py to run and generate a nasty teacher model for a specific model type on a specific dataset. change save_path location based on which ever model you wish to run on whichever dataset.
3. run: run_kd_from_nasty.py to run a skeptical student that learns from a nasty teacher that was generated in step 2. change save_path accordingly.  

### Cite this work
If you find this project useful to you, please cite our work:

      @inproceedings
      {kundu2021analyze, 
      author  ={S. {Kundu} and Q. {Sun} and Y. {FU} and M. {Pedram} and P. A. {Beerel}}, 
      booktitle ={35th Neural Information Processing Systems}, 
      title   ={Analyzing the Confidentiality of Undistillable Teachers in Knowledge Distillation}, 
      year    ={2021}}

### Acknowledgments
[Undistillable repo](https://github.com/VITA-Group/Nasty-Teacher)

[Be your own teacher repo](https://github.com/luanyunteng/pytorch-be-your-own-teacher)

[ZSKD](https://github.com/polo5/ZeroShotKnowledgeTransfer)
