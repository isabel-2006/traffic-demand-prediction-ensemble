from flask import Flask, request, jsonify, render_template_string
import joblib
import pandas as pd
import numpy as np

app = Flask(__name__)

print("Loading production artifacts into memory...")
try:
    lgb_model = joblib.load("lgb_model.pkl")
    xgb_model = joblib.load("xgb_model.pkl")
    model_features = joblib.load("model_features.pkl")
    print("All models and features successfully mounted!")
except Exception as e:
    print(f"Error loading model artifacts: {str(e)}")

# HTML Template using Bootstrap 5 for a clean UI
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Innovexa Traffic Analytics Dashboard</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <style>
        body { background-color: #f4f6f9; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; }
        .card { border: none; border-radius: 12px; box-shadow: 0 4px 12px rgba(0,0,0,0.05); }
        .header-banner { background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%); color: white; padding: 30px; border-radius: 12px; margin-bottom: 30px; }
        .metric-badge { font-size: 1.5rem; font-weight: bold; padding: 10px 20px; border-radius: 8px; display: inline-block; }
    </style>
</head>
<body>
    <div class="container py-5">
        <div class="header-banner text-center">
            <h1 class="fw-bold">INNOVEXA TRAFFIC ANALYTICS</h1>
            <p class="lead mb-0">Real-time Traffic Demand Prediction System (LightGBM & XGBoost Stacking Ensemble)</p>
        </div>

        <div class="row g-4">
            <div class="col-lg-6">
                <div class="card p-4 h-100">
                    <h3 class="text-secondary fw-bold mb-4">Feature Vector Input</h3>
                    <form id="predictionForm">
                        <div class="row g-3">
                            <div class="col-md-6">
                                <label class="form-label fw-semibold">Speed Limit (km/h)</label>
                                <input type="number" class="form-control" name="Speed_Limit" value="60" required>
                            </div>
                            <div class="col-md-6">
                                <label class="form-label fw-semibold">Number of Lanes</label>
                                <input type="number" class="form-control" name="Number_of_Lanes" value="3" required>
                            </div>
                            <div class="col-md-6">
                                <label class="form-label fw-semibold">Historical Avg Speed</label>
                                <input type="number" step="0.1" class="form-control" name="Historical_Avg_Speed" value="45.5" required>
                            </div>
                            <div class="col-md-6">
                                <label class="form-label fw-semibold">Congestion Index</label>
                                <input type="number" step="0.1" class="form-control" name="Congestion_Index" value="24.2" required>
                            </div>
                            <div class="col-md-6">
                                <label class="form-label fw-semibold">Traffic Signals Count</label>
                                <input type="number" class="form-control" name="Traffic_Signals" value="2" required>
                            </div>
                            <div class="col-md-6">
                                <label class="form-label fw-semibold">Nearby Intersections</label>
                                <input type="number" class="form-control" name="Nearby_Intersections" value="3" required>
                            </div>
                            <div class="col-md-6">
                                <label class="form-label fw-semibold">Temperature (°C)</label>
                                <input type="number" step="0.1" class="form-control" name="Temperature" value="28.0" required>
                            </div>
                            <div class="col-md-6">
                                <label class="form-label fw-semibold">Rainfall (mm)</label>
                                <input type="number" step="0.1" class="form-control" name="Rainfall" value="0.0" required>
                            </div>
                            <div class="col-md-6">
                                <label class="form-label fw-semibold">Demand Lag 1 (15m ago)</label>
                                <input type="number" class="form-control" name="Demand_Lag_1" value="120" required>
                            </div>
                            <div class="col-md-6">
                                <label class="form-label fw-semibold">Rolling Mean 4 (1h avg)</label>
                                <input type="number" class="form-control" name="Rolling_Mean_4" value="115" required>
                            </div>
                        </div>
                        <button type="submit" class="btn btn-primary w-100 mt-4 fw-bold py-2 fs-5">Generate Real-Time Prediction</button>
                    </form>
                </div>
            </div>

            <div class="col-lg-6">
                <div class="card p-4 h-100 text-center d-flex flex-column justify-content-center align-items-center" id="resultCard">
                    <div id="placeholderText">
                        <h4 class="text-muted">Awaiting Input Parameters</h4>
                        <p class="text-muted small">Fill out the feature vector values on the left and submit to trigger ensemble model inference.</p>
                    </div>
                    <div id="resultsWrapper" class="d-none w-100">
                        <h3 class="text-secondary fw-bold mb-3">Prediction Output</h3>
                        <div class="my-4">
                            <span class="text-muted d-block uppercase fw-semibold">Predicted Vehicle Volume</span>
                            <div class="metric-badge bg-primary text-white my-2" id="outputDemand">0</div>
                        </div>
                        <div class="my-4">
                            <span class="text-muted d-block fw-semibold">Status / Congestion Classification</span>
                            <div class="alert fw-bold d-inline-block px-4 py-2 mt-2" id="outputStatus">Normal</div>
                        </div>
                        <hr class="my-4">
                        <h5 class="text-start text-muted fw-bold mb-2">Ensemble Model Breakdown:</h5>
                        <div class="d-flex justify-content-around text-start w-100">
                            <div><strong>LightGBM (55%):</strong> <span id="lgbValue">0</span></div>
                            <div><strong>XGBoost (45%):</strong> <span id="xgbValue">0</span></div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <script>
        document.getElementById('predictionForm').addEventListener('submit', async (e) => {
            e.preventDefault();
            const formData = new FormData(e.target);
            const payload = {};
            
            // Map form data fields into a JSON payload object
            formData.forEach((value, key) => {
                payload[key] = parseFloat(value);
            });

            // Trigger fetch request to backend REST endpoint
            try {
                const response = await fetch('/api/predict', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(payload)
                });
                const result = await response.json();

                if (result.status === 'success') {
                    document.getElementById('placeholderText').classList.add('d-none');
                    document.getElementById('resultsWrapper').classList.remove('d-none');
                    
                    document.getElementById('outputDemand').innerText = result.prediction_results.predicted_traffic_demand;
                    
                    const statusDiv = document.getElementById('outputStatus');
                    statusDiv.innerText = result.prediction_results.congestion_classification;
                    
                    // Dynamic coloring based on traffic intensity classes
                    statusDiv.className = "alert fw-bold d-inline-block px-4 py-2 mt-2 ";
                    if(result.prediction_results.predicted_traffic_demand > 140) {
                        statusDiv.classList.add('alert-danger');
                    } else if(result.prediction_results.predicted_traffic_demand > 70) {
                        statusDiv.classList.add('alert-warning');
                    } else {
                        statusDiv.classList.add('alert-success');
                    }

                    document.getElementById('lgbValue').innerText = result.prediction_results.individual_components.lightgbm_sub_prediction;
                    document.getElementById('xgbValue').innerText = result.prediction_results.individual_components.xgboost_sub_prediction;
                }
            } catch (error) {
                alert('Inference Request Failed: ' + error);
            }
        });
    </script>
</body>
</html>
"""

@app.route('/', methods=['GET'])
def index():
    """Renders the HTML Dashboard UI interface."""
    return render_template_string(HTML_TEMPLATE)

@app.route('/api/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"status": "error", "message": "No input data provided"}), 400
        
        input_df = pd.DataFrame([data])
        
        # Structure matching and default missing value imputation guardrails
        for col in model_features:
            if col not in input_df.columns:
                input_df[col] = 0
                
        input_df = input_df[model_features]
        
        # Stacking model execution calls
        lgb_res = lgb_model.predict(input_df)[0]
        xgb_res = xgb_model.predict(input_df)[0]
        
        # Apply Stacking Blend logic weights
        final_traffic_demand = (0.55 * lgb_res) + (0.45 * xgb_res)
        final_traffic_demand = int(max(0, final_traffic_demand))
        
        if final_traffic_demand > 220:
            congestion_level = "Critical Gridlock"
        elif final_traffic_demand > 140:
            congestion_level = "High Congestion"
        elif final_traffic_demand > 70:
            congestion_level = "Moderate Traffic"
        else:
            congestion_level = "Smooth Flow"
            
        return jsonify({
            "status": "success",
            "prediction_results": {
                "predicted_traffic_demand": final_traffic_demand,
                "congestion_classification": congestion_level,
                "individual_components": {
                    "lightgbm_sub_prediction": round(float(lgb_res), 2),
                    "xgboost_sub_prediction": round(float(xgb_res), 2)
                }
            }
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
