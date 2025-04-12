# InternIntelligence_plant_disease_classification
# Plant_Disease_classification with Xception model

## Description
This notebook focuses on plant disease classification using the Xception model. The model analyzes plant images and identifies potential diseases, helping farmers and researchers detect plant health issues.
I chose Xception because it proved to be one of the most effective model for classifying plant disease, achieving an accuracy of 96% along with balanced
precision and recall metrics for all classes.
One challenge I faced was the long training times and high computational power needed, as well as the large size of the notebook (around 50 MB) due to the use of transfer learning, which led to loading the Xception model’s parameter values directly into the notebook. This made the notebook slower to load and run. In the future, I hope to address these issues by reducing the model size, exploring optimization techniques, and implementing more efficient methods for training and loading the model.


## Dataset
For this project, I used a dataset of plant images, which includes images of plants affected by various diseases.The dataset has two versions: one with data augmentation and another without. I used the non-augmented version. It consists of 39 images class, each labeled with the corresponding disease category, with total NO. 61,486 images. The dataset is publicly available and was chosen for its diversity in plant species and disease types, allowing us to evaluate the models' ability to generalize across different plant species.

you can use these photos to make the model predict on: 

![image (123)](https://github.com/user-attachments/assets/e3525a06-bcf9-4202-a5db-dcebd6075968)
![image (1005)](https://github.com/user-attachments/assets/7cd6df40-baf0-4a6d-9fff-9f6f698c458a)
![image (1009)](https://github.com/user-attachments/assets/f08626d8-801c-469e-9538-2706a0386215)
![image (1004)](https://github.com/user-attachments/assets/13290043-0e09-42c2-bdf2-dab3712a9213)
![image (101)](https://github.com/user-attachments/assets/6fbbf2be-9053-4e99-a598-48309b2400e6)
![image (110)](https://github.com/user-attachments/assets/ea13e44d-af6a-496f-bd5a-9a3c10e9bafb)

or you may use anyother plant leaf image, just stick to the plant cateogeries you find in the dataset.

## References
Xception: Deep Learning with Depthwise Separable Convolutions-> [https://arxiv.org/pdf/1610.02357]

[https://paperswithcode.com/method/xception]

[https://medium.com/data-science/review-xception-with-depthwise-separable-convolution-better-than-inception-v3-image-dc967dd42568]

You can access the dataset here:
[https://data.mendeley.com/datasets/tywbtsjrjv/1]

## Note 
i uploaded the "model.keras" to drive due to size constraints in github free plan.

the note is big in size so to view it you can load it in your local machine or simply check it here:
[https://www.kaggle.com/code/mariamelsaket/plant-disease-classification-with-xception]

