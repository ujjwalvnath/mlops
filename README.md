# Iris Species Classification using Decision Tree

This project implements a simple machine learning model to classify iris flower species using a Decision Tree classifier. The model is trained on the classic [Iris dataset](https://en.wikipedia.org/wiki/Iris_flower_data_set) and outputs the model accuracy, as well as saves the trained model for future use.

---

## 📁 Project Structure
```
mlops/
├── data/
│ └── iris.csv # Dataset (required)
└── model.joblib # Output: Saved trained model
├── main.py # Main script to run the classifier
├── requirements.txt # Python dependencies
└── README.md # Project documentation
```

---

How to Run

 1. Clone the repository (if applicable)
```bash
git clone <your-repo-url>
cd mlops
```
2. Set up a virtual environment (recommended)
```
python3 -m venv .env
source .env/bin/activate
```
3. Install dependencies
```
pip install -r requirements.txt
```
5. Ensure the dataset exists

Make sure the Iris dataset CSV is placed in the data/ directory:
data/iris.csv
If the dataset doesn't have proper headers, update or rename them to:
```
sepal_length,sepal_width,petal_length,petal_width,species
```
5. Run the script
```
python main.py
```
