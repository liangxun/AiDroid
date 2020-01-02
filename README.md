# AiDroid
Reimplementation of AiDroid ([AiDroid: When Heterogeneous Information Network Marries Deep Neural Network for Real-time Android Malware Detection](https://arxiv.org/abs/1811.01027)).  

*Disclaimer: This **is not** official code repository, and I am not responsible for the accuracy of the code.*  


**dataset description**: 
There are 6 types of nodes (i.e., apk, api(dense), permission, tpl, url, component) in HIN and 4 metapaths (i.e., apk-tpl-apk, apk-permission-apk, apk-api-apk, apk-component-apk, apk-url-apk) are consider in this repository.  


**run scripts**:
(your data_path needs to specificed in each script before runing)
```
# step1: biased random walk. get corpus from HIN
python randomwalk.py

# step2: learn representation for each node
python node2vec.py

# step3: hin2Img. 
python hin2img.py

# step4: train classifer.
python train.py
```
