#  🌱 Plant Disease Identification Model

The **Crop Disease Detection System** leverages Convolutional Neural Networks (CNN) to accurately identify plant diseases from leaf images. Trained on the **Plant Disease Image Dataset**, which includes 70,295 images in the training set and 17,572 images in the validation set, the model covers 38 different plant disease classes across 14 crops. It detects and classifies diseases such as **Apple Scab**, **Tomato Blight**, and **Powdery Mildew**, providing farmers with a reliable tool for early disease detection.

## Dataset

The **Plant Disease Image Dataset**, employed for identifying crop diseases, contains 70,295 images in the training set and 17,572 images in the validation set, representing 38 distinct plant disease categories. The images are resized to a uniform resolution of 128x128 pixels, and the total dataset requires around five gigabytes of storage space.

## Model Architecture

The **Plant Disease Detection Model** utilizes a Convolutional Neural Network (CNN) architecture, tailored for recognizing crop diseases. By applying deep learning methods, this CNN processes plant leaf images to precisely identify and categorize various diseases. This model supports farmers in the early detection and management of plant diseases, helping enhance crop health and overall yield.

### Key Features:
- **Targeted Crops**: The model is developed to detect diseases in a defined group of crops.
- **Disease Classification**: It is capable of identifying diseases from leaf images.
- **High Precision**: The CNN model shows exceptional precision in detecting plant diseases, assisting farmers and researchers with early problem identification.

### Supported Crops and Diseases:
- The model operates with a fixed set of 14 crops.
- For each crop, it is trained to identify and categorize up to 38 distinct diseases.

As the model is trained for a specific set of crops, it is capable of diagnosing only those particular crops. The following is the list of crops for which this model is useful:

```
[ 'Apple',
'Blueberry',
'Cherry_(including sour)',
'Corn_(maize)',
'Grape',
'Orange',
'Peach', 'Pepper, _bell',
'Potato',
'Raspberry',
'Soybean',
'Squash',
'Strawberry',
'Tomato' ]
```


The model can only diagnose specific diseases for each crop, as it is trained on particular conditions. Below is the list of crop diseases that the model has been trained to identify:

```
Found 17572 files belonging to 38 classes.
['Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy', 'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 'Cherry_(including_sour)___healthy',
'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 'Grape___Black_rot', 'Grape___Esca_(Black_Measles)',
'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy', 'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy', 'Pepper,_bell___Bacterial_spot',
'Pepper,_bell___healthy', 'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy', 'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew', 'Strawberry___Leaf_scorch',
'Strawberry___healthy', 'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite',
'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus', 'Tomato___healthy']

```
## 📄 [Crop Disease Guide](DISEASE-GUIDE.md)

### Functionality:
- The model analyzes images of plant leaves to recognize signs of different diseases.
- It utilizes CNN-based image recognition to accurately identify the disease associated with a specific crop.







| **Examples of Diseased and Healthy Images** |
| ------------------------------------------- |
| ![Apple Cedar Rust](https://github.com/Yashrajgithub/Crop-Disease-Identification-Model/blob/main/KrishiGyaan%20Plant%20Disease%20Identification/test/AppleCedarRust1.JPG) ![Apple Scab](https://github.com/Yashrajgithub/Crop-Disease-Identification-Model/blob/main/KrishiGyaan%20Plant%20Disease%20Identification/test/AppleScab1.JPG) ![Tomato Healthy](https://github.com/Yashrajgithub/Crop-Disease-Identification-Model/blob/main/KrishiGyaan%20Plant%20Disease%20Identification/test/TomatoHealthy4.JPG) ![Potato Healthy](https://github.com/Yashrajgithub/Crop-Disease-Identification-Model/blob/main/KrishiGyaan%20Plant%20Disease%20Identification/test/PotatoHealthy2.JPG) |


## Model Prediction Results

| **Home Page** |
| -------------- |
| ![Home Page](https://github.com/Yashrajgithub/Crop-Disease-Identification-Model/blob/main/KrishiGyaan%20Plant%20Disease%20Identification/Home.png) |

| **Result Page** |
| ---------------- |
| ![Result Page](https://github.com/Yashrajgithub/Crop-Disease-Identification-Model/blob/main/KrishiGyaan%20Plant%20Disease%20Identification/Result.png) |

