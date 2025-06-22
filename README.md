# Flask Liver Cirrhosis Prediction Application

A comprehensive Flask web application for predicting liver cirrhosis using machine learning models. This application provides both individual and batch prediction capabilities with a user-friendly interface.

## Features

- **Multiple ML Models**: Ensemble, Random Forest, XGBoost, SVM, KNN, and MLP Neural Network
- **Individual Prediction**: Real-time prediction with probability visualization
- **Batch Prediction**: Upload CSV files for bulk predictions with PDF report generation
- **User Authentication**: Secure login system with MongoDB
- **Interactive Dashboard**: Model performance metrics and ROC curves visualization
- **Responsive UI**: Modern, user-friendly interface

## Technologies Used

- **Backend**: Flask, Python
- **Database**: MongoDB
- **Machine Learning**: scikit-learn, XGBoost, CatBoost
- **Frontend**: HTML, CSS, JavaScript, Bootstrap
- **Visualization**: Matplotlib, Plotly

## Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd flask-mongo-auth
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   # On Windows
   venv\Scripts\activate
   # On macOS/Linux
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure MongoDB**
   - Ensure MongoDB is running on your system
   - Update database configuration in `config.py` if needed

5. **Run the application**
   ```bash
   python app.py
   ```

## Usage

### Individual Prediction
1. Navigate to the prediction page
2. Enter patient data (age, bilirubin levels, etc.)
3. Select a machine learning model
4. Get instant prediction with confidence scores

### Batch Prediction
1. Prepare a CSV file with patient data
2. Upload the file through the batch prediction interface
3. Download results as PDF report

### Dashboard
- View model performance metrics
- Compare ROC curves across different models
- Monitor prediction accuracy and other statistics

## Project Structure

```
flask-mongo-auth/
├── app.py                 # Main Flask application
├── auth.py               # Authentication blueprint
├── config.py             # Configuration settings
├── database.py           # Database initialization
├── models.py             # Model definitions
├── generate_metrics.py   # Model evaluation script
├── requirements.txt      # Python dependencies
├── templates/            # HTML templates
├── static/              # Static files (CSS, JS, images)
├── models/              # Trained ML models
├── venv/                # Virtual environment (excluded from git)
└── README.md            # Project documentation
```

## Model Information

The application includes several pre-trained machine learning models:
- **Ensemble Model**: Combined predictions from multiple algorithms
- **Random Forest**: Tree-based ensemble method
- **XGBoost**: Gradient boosting framework
- **SVM**: Support Vector Machine
- **KNN**: K-Nearest Neighbors
- **MLP**: Multi-layer Perceptron Neural Network

## API Endpoints

- `GET /`: Home page
- `GET /dashboard`: Model dashboard with metrics
- `POST /predict`: Individual prediction endpoint
- `POST /batch_predict`: Batch prediction endpoint
- `GET /download_batch_result`: Download batch prediction results

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## License

This project is licensed under the MIT License.

## Contact

For questions or support, please open an issue on GitHub.
