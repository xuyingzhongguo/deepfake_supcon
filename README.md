# Supervised Contrastive Learning for Generalizable and Explainable DeepFakes Detection
[**[Paper]**](https://openaccess.thecvf.com/content/WACV2022W/XAI4B/html/Xu_Supervised_Contrastive_Learning_for_Generalizable_and_Explainable_DeepFakes_Detection_WACVW_2022_paper.html)\
Ying Xu, Kiran Raja, Marius Pedersen
# Introduction:
We propose a generalizable detection model that can detect novel and unknown/unseen DeepFakes using a supervised contrastive (SupCon) loss. We obtain the highest accuracy of 78.74% using proposed SupCon model and an accuracy of 83.99% with proposed fusion in a true open-set evaluation scenario where the test class is unknown at the training phase.
# Framework:
<img src="/plots/proposed_approach1_big.png" alt="Framework" width="700"/>

# How to use:
```
main_supcon.py
main_linear.py
```
Just remember two trains needed to be conducted.

# Datalist
It is a .txt file that includes 'image_path label' every line.
Here is an example:
```
FaceForensics++/original_sequences/youtube/c23/face_images/870/frame121.png 0
FaceForensics++/manipulated_sequences/Deepfakes/c23/face_images/979_875/frame1.png 1
...
```

# Download model
Let me check if I have it.

# Citing:
Please kindly cite the following paper, if you find this code helpful in your work.
```
@inproceedings{xu2022supervised,
  title={Supervised contrastive learning for generalizable and explainable deepfakes detection},
  author={Xu, Ying and Raja, Kiran and Pedersen, Marius},
  booktitle={Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision},
  pages={379--389},
  year={2022}
}
```
# Contact:
Please feel free to contact ying.xu@ntnu.no, if you have any questions.

