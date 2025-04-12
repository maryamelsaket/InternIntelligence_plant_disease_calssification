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

![image (10)](https://github.com/user-attachments/assets/78dd6840-c9c8-4347-bb96-d604fc64637b)
![image (100)](https://github.com/user-attachments/assets/1ca15cde-abce-4cdb-839b-616fbaf04668)
![image (1021)](https://github.com/user-attachments/assets/dc13a4af-5c74-43a1-bb5e-fb33438eddb2)
![image (1)](https://github.com/user-attachments/assets/75acc03e-06a2-4c68-8631-d73b56f6d0e2)
![image (101)](https://github.com/user-attachments/assets/ccedc46a-0a0e-48ca-ae3e-3c6edafeca43)
![image (10)](https://github.com/user-attachments/assets/36b380b2-4813-4360-a29b-da061420e610)

or you may use anyother plant leaf image, just stick to the plant cateogeries you find in the dataset.

## References
Xception: Deep Learning with Depthwise Separable Convolutions-> [https://arxiv.org/pdf/1610.02357]

[https://paperswithcode.com/method/xception]

[https://medium.com/data-science/review-xception-with-depthwise-separable-convolution-better-than-inception-v3-image-dc967dd42568]

You can access the dataset here:
[https://data.mendeley.com/datasets/tywbtsjrjv/1]

## Note 
i uploaded the model.keras to drive due to size constraints in github free plan 
Due to time constraints, the results for this project are not included in this repository. However, the code is fully functional, and you can check it directly in the Kaggle Notebook for now.
here is the link:
[https://www.kaggle.com/code/mariamelsaket/plant-disease-classification-with-xception]

