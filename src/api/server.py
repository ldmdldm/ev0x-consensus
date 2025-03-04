"""API server implementation for the ev0x project."""

import logging
from flask import Flask, jsonify, request
from src.evolution.selection import AdaptiveModelSelector
from src.bias.detector import BiasDetector
from src.bias.neutralizer import BiasNeutralizer

# Initialize logger
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Initialize components
model_selector = AdaptiveModelSelector()
bias_detector = BiasDetector()
bias_neutralizer = BiasNeutralizer()

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({"status": "healthy"})

@app.route('/predict', methods=['POST'])
def predict():
    """
    Endpoint for generating predictions using the ev0x system.
    
    This endpoint:
    1. Selects the appropriate models using the adaptive selection
    2. Detects any biases in the input
    3. Generates predictions from multiple models
    4. Neutralizes biases in the output
    5. Returns the final consensus result
    """
    try:
        data = request.json
        if not data or 'input' not in data:
            return jsonify({"error": "Invalid input"}), 400
        
        # Select appropriate models
        models = model_selector.select_models(data['input'])
        
        # Detect bias in input
        bias_report = bias_detector.detect(data['input'])
        
        # Get predictions from models (placeholder)
        predictions = [{"model": "model1", "output": "prediction1"},
                    {"model": "model2", "output": "prediction2"}]
        
        # Neutralize bias in output
        neutralized_predictions = bias_neutralizer.neutralize(predictions, bias_report)
        
        # Generate consensus (placeholder)
        consensus = "Final consensus output"
        
        return jsonify({
            "consensus": consensus,
            "models_used": [m.name for m in models],
            "bias_report": bias_report.to_dict() if bias_report else None,
            "predictions": neutralized_predictions
        })
        
    except Exception as e:
        logger.exception("Error processing prediction request")
        return jsonify({"error": str(e)}), 500

def start_server(host='0.0.0.0', port=5000, debug=False):
    """Start the API server."""
    app.run(host=host, port=port, debug=debug)

if __name__ == '__main__':
    start_server(debug=True)

