# Innovexa Traffic Analytics: Traffic Demand Prediction System

An end-to-end traffic analytics optimization system utilizing an advanced Stacking Ensemble model (LightGBM + XGBoost) integrated with a real-time web dashboard powered by Flask and Bootstrap 5.

## 🔗 Live Deployment Link
👉 **[Click Here to Access the Live Dashboard](https://traffic-prediction-dashboard-issabel.onrender.com)** *(Free instance containers take a moment to boot up if idle).*

## 🚀 Key Project Features
* **Custom Dataset**: 2+ years of historical log data at 15-minute intervals across 4 distinct road segments (280,000+ records).
* **Advanced Feature Engineering**: Implements temporal components, cyclical encoding, multi-step lag features, and rolling context windows.
* **Stacking Ensemble Engine**: Blends optimized LightGBM (55%) and XGBoost (45%) models to map deep dependencies.

## 📊 Performance Metrics
* **Final Stacking Ensemble $R^2$ Score**: **95.13%**

## 🏃‍♂️ How to Run Locally
```bash
git clone [https://github.com/your-username/traffic-demand-prediction-ensemble.git](http://github.com/isabel-2006/traffic-demand-prediction-ensemble)
cd traffic-demand-prediction-ensemble
pip install flask lightgbm xgboost scikit-learn pandas numpy joblib
python app.py