## CONFIG ENV (BOVIFOCR)

#### 1. Clone this repo:
```
git clone https://github.com/BOVIFOCR/CDCN_FAS.git
cd CDCN_FAS
```

#### 2. Create conda env and install python libs:
```
export CONDA_ENV=bjgbiesseck_cdcn_py39
conda create -y -n $CONDA_ENV python=3.9
conda activate $CONDA_ENV
conda env config vars set CUDA_HOME="/usr/local/cuda-11.6"; conda deactivate; conda activate $CONDA_ENV
conda env config vars set LD_LIBRARY_PATH="$CUDA_HOME/lib64"; conda deactivate; conda activate $CONDA_ENV
conda env config vars set PATH="$CUDA_HOME:$CUDA_HOME/bin:$LD_LIBRARY_PATH:$PATH"; conda deactivate; conda activate $CONDA_ENV

conda install pytorch=1.13.0 torchvision pytorch-cuda=11.6 -c pytorch -c nvidia
conda install -c fvcore -c iopath -c conda-forge fvcore iopath
conda install -c bottler nvidiacub
conda install pytorch3d -c pytorch3d
pip3 install -r requirements.txt
```

#### 3. Train model:
```
cd CVPR2020_paper_codes
export CUDA_VISIBLE_DEVICES=0; python train_CDCN_1RandomFrame.py
```

<br> <br> <be> 



# CDCN (Original)

Main code of [**CVPR2020 paper "Searching Central Difference Convolutional Networks for Face Anti-Spoofing"**  ](https://arxiv.org/pdf/2003.04092v1.pdf) 

------

Based on the **Central Difference Convolution (CDC)** and **Contrastive Depth Loss (CDL)**, we achieved

**1st Place** in ChaLearn Multi-Modal Face Anti-spoofing Attack Detection Challenge @CVPR2020


**2nd Place** in ChaLearn Single-Modal(RGB) Face Anti-spoofing Attack Detection Challenge @CVPR2020

-------
 It is just for **research purpose**, and commercial use is not allowed.

Citation
------- 
If you use the CDC, D-CDC or CDL, please cite these six papers:  

>@inproceedings{yu2020nasfas,  
 >&nbsp;&nbsp;&nbsp;&nbsp;title={NAS-FAS: Static-Dynamic Central Difference Network Search for Face Anti-Spoofing},      
 >&nbsp;&nbsp;&nbsp;&nbsp;author={Yu, Zitong and Wan, Jun and Qin, Yunxiao and Li, Xiaobai and Li, Stan Z. and Zhao, Guoying},  
 >&nbsp;&nbsp;&nbsp;&nbsp;booktitle= {TPAMI},  
 >&nbsp;&nbsp;&nbsp;&nbsp;year = {2020}  
 >}  

 >@inproceedings{yu2021dual,  
 >&nbsp;&nbsp;&nbsp;&nbsp;title={Dual-Cross Central Difference Network for Face Anti-Spoofing},      
 >&nbsp;&nbsp;&nbsp;&nbsp;author={Yu, Zitong and Qin, Yunxiao and Zhao, Hengshuang and Li, Xiaobai and Zhao, Guoying},  
 >&nbsp;&nbsp;&nbsp;&nbsp;booktitle= {IJCAI},  
 >&nbsp;&nbsp;&nbsp;&nbsp;year = {2021}  
 >}  

 >@inproceedings{yu2020searching,  
 >&nbsp;&nbsp;&nbsp;&nbsp;title={Searching Central Difference Convolutional Networks for Face Anti-Spoofing},      
 >&nbsp;&nbsp;&nbsp;&nbsp;author={Yu, Zitong and Zhao, Chenxu and Wang, Zezheng and Qin, Yunxiao and Su, Zhuo and Li, Xiaobai and Zhou, Feng and Zhao, Guoying},  
 >&nbsp;&nbsp;&nbsp;&nbsp;booktitle= {CVPR},  
 >&nbsp;&nbsp;&nbsp;&nbsp;year = {2020}  
 >}  

 >@inproceedings{yu2020face,  
 >&nbsp;&nbsp;&nbsp;&nbsp;title={Face Anti-spoofing with Human Material Perception},      
 >&nbsp;&nbsp;&nbsp;&nbsp;author={Yu, Zitong and Li, Xiaobai and Niu, Xuesong and Shi, Jingang and Zhao, Guoying},  
 >&nbsp;&nbsp;&nbsp;&nbsp;booktitle= {ECCV},  
 >&nbsp;&nbsp;&nbsp;&nbsp;year = {2020}  
 >}  

 >@inproceedings{wang2020deep,  
 >&nbsp;&nbsp;&nbsp;&nbsp;title={Deep Spatial Gradient and Temporal Depth Learning for Face Anti-spoofing},      
 >&nbsp;&nbsp;&nbsp;&nbsp;author={Wang, Zezheng and Yu, Zitong and Zhao, Chenxu and Zhu, Xiangyu and Qin, Yunxiao and Zhou, Qiusheng and Zhou, Feng and Lei, Zhen},  
 >&nbsp;&nbsp;&nbsp;&nbsp;booktitle= {CVPR},  
 >&nbsp;&nbsp;&nbsp;&nbsp;year = {2020}  
 >}  

 >@inproceedings{qin2019learning,  
 >&nbsp;&nbsp;&nbsp;&nbsp;title={Learning Meta Model for Zero-and Few-shot Face Anti-spoofing},      
 >&nbsp;&nbsp;&nbsp;&nbsp;author={Qin, Yunxiao and Zhao, Chenxu and Zhu, Xiangyu and Wang, Zezheng and Yu, Zitong and Fu, Tianyu and Zhou, Feng and Shi, Jingping and Lei, Zhen},  
 >&nbsp;&nbsp;&nbsp;&nbsp;booktitle= {AAAI},  
 >&nbsp;&nbsp;&nbsp;&nbsp;year = {2020}  
 >}  


