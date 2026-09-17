\# 🌱 Multimodal Plant Disease Detection



A multimodal AI system for detecting potato plant diseases using \*\*leaf images and textual symptom descriptions\*\*.



The project combines a \*\*MobileNetV2-based computer vision model\*\* with an \*\*NLP symptom classification model\*\* and uses weighted probability fusion to produce a final disease prediction through a Flask web application.



\## 🎯 Project Overview



Plant diseases can often be identified from visible symptoms such as spots, lesions, discoloration, and wilting. However, relying only on an image may not capture all the information available to a user.



This project combines two sources of information:



\* 🖼️ \*\*Leaf image\*\* → Computer Vision model

\* 📝 \*\*Symptom description\*\* → NLP model



The predictions from both models are then combined using weighted multimodal fusion.



```text

&#x20;                   User

&#x20;                    │

&#x20;           ┌────────┴────────┐

&#x20;           │                 │

&#x20;      Leaf Image       Symptom Description

&#x20;           │                 │

&#x20;           ▼                 ▼

&#x20;      MobileNetV2          TF-IDF

&#x20;           │                 │

&#x20;           ▼                 ▼

&#x20;   Image Classification   Logistic Regression

&#x20;           │                 │

&#x20;           └────────┬────────┘

&#x20;                    ▼

&#x20;            Weighted Fusion

&#x20;             70% Image

&#x20;             30% Text

&#x20;                    │

&#x20;                    ▼

&#x20;            Final Prediction

&#x20;                    │

&#x20;                    ▼

&#x20;             Flask Web App

```



\## ✨ Features



\* Potato leaf disease classification

\* MobileNetV2 transfer learning

\* Three disease/health classes:



&#x20; \* Early Blight

&#x20; \* Healthy

&#x20; \* Late Blight

\* NLP-based symptom classification

\* TF-IDF text feature extraction

\* Logistic Regression text classifier

\* Multimodal image + text prediction

\* Confidence-based weighted fusion

\* Flask web interface

\* Command-line image prediction

\* Model evaluation using accuracy, precision, recall, F1-score, and confusion matrix



\## 🧠 Machine Learning Components



\### 1. Computer Vision



The image classification component uses \*\*MobileNetV2\*\* with transfer learning.



The input image is:



1\. Converted to RGB

2\. Resized to `224 × 224`

3\. Preprocessed using MobileNetV2 preprocessing

4\. Passed through the trained model

5\. Converted into class probabilities



The model predicts one of:



```text

early\_blight

healthy

late\_blight

```



\### 2. NLP Symptom Classification



The NLP component classifies textual descriptions of potato leaf symptoms.



Example:



```text

"The potato leaves have large dark brown patches spreading quickly."

```



The text is converted into numerical features using \*\*TF-IDF\*\* and classified using \*\*Logistic Regression\*\*.



The NLP model produces probabilities for the same three classes:



```text

early\_blight

healthy

late\_blight

```



\### 3. Multimodal Fusion



The image and text predictions are combined using weighted probability fusion.



```text

Final Probability =

&#x20;   0.7 × Image Probability

&#x20; + 0.3 × Text Probability

```



The class with the highest final probability becomes the final prediction.



The current configuration gives:



```text

Image contribution: 70%

Text contribution: 30%

```



\## 📊 Results



\### Image Classification



The trained image model achieved:



```text

Test Accuracy: 95.21%

Test Loss:     0.1137

```



Classification performance on the test set:



| Class        | Precision | Recall | F1-score |

| ------------ | --------: | -----: | -------: |

| Early Blight |      0.99 |   0.97 |     0.98 |

| Healthy      |      0.73 |   0.96 |     0.83 |

| Late Blight  |      0.97 |   0.93 |     0.95 |



Overall:



```text

Accuracy: 95%

Macro F1: 0.92

Weighted F1: 0.95

```



\### NLP Classification



The symptom classifier achieved:



```text

Accuracy: 88.89%

Macro F1: 0.89

```



Test-set classification:



| Class        | Precision | Recall | F1-score |

| ------------ | --------: | -----: | -------: |

| Early Blight |      0.75 |   1.00 |     0.86 |

| Healthy      |      1.00 |   0.67 |     0.80 |

| Late Blight  |      1.00 |   1.00 |     1.00 |



\### Example Multimodal Prediction



Input:



```text

Image:

Potato leaf image



Symptoms:

"The potato leaves have large dark brown irregular patches

that are spreading quickly. The affected areas look wet

and dark, and the leaves are beginning to wilt."

```



Output:



```text

Image Prediction:

Late Blight

Confidence: 77.65%



Text Prediction:

Late Blight

Confidence: 53.03%



Final Multimodal Prediction:

Late Blight

Confidence: 70.26%

```



\## 🛠️ Technologies



\### Machine Learning



\* Python

\* TensorFlow

\* Keras

\* MobileNetV2

\* NumPy

\* Pandas

\* Scikit-learn



\### NLP



\* TF-IDF

\* Logistic Regression

\* Scikit-learn



\### Computer Vision



\* Pillow

\* MobileNetV2



\### Web Application



\* Flask

\* HTML



\### Development



\* Git

\* GitHub

\* Jupyter Notebook



\## 📁 Project Structure



```text

Plant-Disease-Detection/

│

├── data/

│   └── nlp/

│       └── symptom\_data.csv

│

├── models/

│   └── plant\_disease\_mobilenetv2.keras

│

├── notebooks/

│   └── plant\_disease\_detection.ipynb

│

├── src/

│   ├── app.py

│   ├── nlp\_classifier.py

│   ├── predict.py

│   └── prepare\_dataset.py

│

├── templates/

│   └── index.html

│

├── .gitignore

├── requirements.txt

└── README.md

```



> The full image dataset and trained model are excluded from GitHub using `.gitignore`.



\## 📦 Dataset



The computer vision component was trained using potato leaf images from the PlantVillage-based dataset.



The project focuses on three classes:



```text

Early Blight

Healthy

Late Blight

```



The dataset was divided into:



```text

Training

Validation

Testing

```



The training data contained fewer healthy examples than the two disease classes, so class weights were used during model training to reduce the effect of class imbalance.



\## 🚀 Installation



Clone the repository:



```bash

git clone https://github.com/ELANGESWARANKISHOR/plant-disease-detection-ml.git

cd plant-disease-detection-ml

```



Create a virtual environment:



```bash

python -m venv .venv

```



Activate it on Windows PowerShell:



```powershell

.venv\\Scripts\\Activate.ps1

```



Install the dependencies:



```bash

pip install -r requirements.txt

```



\## ▶️ Running the Application



The Flask application requires the trained model file.



Place the trained model at:



```text

models/plant\_disease\_mobilenetv2.keras

```



Then run:



```powershell

python src\\app.py

```



Open the application in a browser:



```text

http://127.0.0.1:5000

```



\## 🔬 Using the Application



\### Step 1 — Upload a leaf image



Upload a potato leaf image.



\### Step 2 — Describe the symptoms



For example:



```text

The potato leaves have large dark brown patches

that are spreading quickly. The affected areas

look wet and the leaves are beginning to wilt.

```



\### Step 3 — Analyze



The system performs:



```text

Image Analysis

&#x20;      +

Text Analysis

&#x20;      ↓

Multimodal Fusion

&#x20;      ↓

Final Disease Prediction

```



\## 💻 Command-Line Prediction



An individual image can also be tested without the web application:



```powershell

python src\\predict.py test\_leaf.jpg

```



Example:



```text

Plant Disease Detection

\-----------------------

Image: test\_leaf.jpg

Prediction: late\_blight

Confidence: 77.65%

```



\## ⚠️ Limitations



\* The system currently focuses on potato leaves.

\* Only three classes are supported.

\* The NLP dataset is relatively small.

\* The multimodal fusion weights are manually selected as 70% image and 30% text.

\* Confidence values represent model/fusion scores and should not be interpreted as guaranteed real-world diagnostic probabilities.

\* Performance on arbitrary internet images may differ from performance on the test dataset.

\* The system is intended as an experimental machine learning project rather than a professional agricultural diagnostic system.



\## 🔮 Future Improvements



Possible future improvements include:



\* Expand the NLP symptom dataset

\* Support additional crops and diseases

\* Improve symptom extraction

\* Experiment with transformer-based NLP models

\* Optimize multimodal fusion weights

\* Add explainable AI for image predictions

\* Add model versioning

\* Deploy the Flask application

\* Add automated testing and CI/CD

\* Evaluate using larger external datasets



\## 📌 Project Goal



The main goal of this project is to explore how \*\*computer vision and natural language processing can be combined in a multimodal machine learning system\*\*.



Rather than relying only on an image, the system allows users to provide both visual and textual information and combines these signals to produce a final prediction.



