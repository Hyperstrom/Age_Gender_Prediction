# **Age and Gender Prediction Program 🧑📏👦**

This project focuses on predicting the age and gender of individuals using a deep learning model. Leveraging transfer learning with the VGG16 architecture enhances the model's performance, allowing accurate predictions on unseen data.

## **Transfer Learning 🔄**

Transfer learning plays a crucial role in this project. By utilizing the pre-trained VGG16 model as a convolutional base, we leverage its ability to extract high-level features from images. Fine-tuning the model by unfreezing the last block (block5) of VGG16 and adding custom dense layers enables it to learn age and gender-specific features from the provided dataset effectively.

## **Model Architecture 🏗️**

The model architecture comprises a combination of pre-trained VGG16 layers and custom dense layers. After extracting features from the VGG16 base, the data is split into two branches for age and gender prediction. Each branch passes through three dense layers, incorporating dropout and batch normalization to prevent overfitting and improve generalization.

Over View of model Architecture
![model](https://github.com/Hyperstrom/Age_Gender_Prediction/assets/112319058/fa5e23e5-0f63-47df-a417-fecf784fd606)

### Model Parameters 📊

- **Total Parameters:** 34,148,546
- **Trainable Parameters:** 19,433,730
- **Non-trainable Parameters:** 14,714,816


## **Data Preprocessing 🛠️**

Before feeding the data into the model, extensive preprocessing steps are undertaken. This includes data augmentation techniques to increase the diversity of the training dataset, ensuring robustness and preventing overfitting. Additionally, data is normalized and resized to meet the input requirements of the VGG16 model.

## **Model Training and Evaluation 📊**

- The model is trained using a combination of training and validation datasets.
- A batch size of 32 is used during training.
- Training extends over 200 epochs to optimize model parameters.
- The Adam optimizer is employed to update model weights.
- During training, both age and gender losses are monitored.
- Mean Absolute Error (MAE) is utilized for age prediction loss.
- Binary Cross-Entropy is used for gender classification loss.
- Training plots are provided to visualize the model's learning progress and performance over epochs.

![model plot](https://github.com/Hyperstrom/Age_Gender_Prediction/assets/112319058/18f470e4-aaad-4b77-aeae-847aff20e486)

## **Model Deployment and Testing 🚀**

Upon successful training, the model is saved for deployment. Testing is conducted using unseen data to evaluate its performance and generalization capabilities. We also deploy the model using Streamlit, providing a user-friendly interface for real-time age and gender prediction using webcam data.

![IMG_20240525_005118](https://github.com/Hyperstrom/Age_Gender_Prediction/assets/112319058/81921889-61ce-4553-9969-587e5cb07606)

## **Future Improvements 🚀**

As with any machine learning project, there's always room for improvement. Future iterations may include fine-tuning hyperparameters, exploring different architectures, and incorporating more advanced techniques for even better performance.

## **How to Use 📝**

1. Clone this repository.
2. Install dependencies (`pip install -r requirements.txt`).
3. Run Streamlit app (`streamlit run app.py`).

Feel free to contribute, test with your own images, and provide feedback!

