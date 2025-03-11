"""API server implementation for the ev0x project."""

import logging
import os
import argparse
import yaml
from flask import Flask, jsonify, request
from src.evolution.selection import AdaptiveModelSelector
from src.bias.detector import BiasDetector
from src.bias.neutralizer import BiasNeutralizer

# Initialize logger
logger = logging.getLogger(__name__)


class APIServer:
    """
    APIServer class that wraps Flask server functionality.
    
    This class encapsulates the Flask application, route handlers,
    and server management functionality.
    """
    
    def __init__(self, config_path=None):
        """
        Initialize the APIServer with a Flask app and required components.
        
        Args:
            config_path (str): Path to the configuration file.
        """
        self.app = Flask(__name__)
        self.config = self.load_config(config_path)
        
        # Initialize components
        self.model_selector = AdaptiveModelSelector()
        self.bias_detector = BiasDetector()
        self.bias_neutralizer = BiasNeutralizer()
        
        # Set up routes
        self._setup_routes()
    
    def _setup_routes(self):
        """Set up all the API routes for the Flask application."""
        self.app.add_url_rule('/health', 'health_check', self.health_check, methods=['GET'])
        self.app.add_url_rule('/predict', 'predict', self.predict, methods=['POST'])
    
    def health_check(self):
        """Health check endpoint."""
        return jsonify({"status": "healthy"})
    
    def predict(self):
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
            models = self.model_selector.select_models(data['input'])
            
            # Detect bias in input
            bias_report = self.bias_detector.detect(data['input'])
            
            # Get predictions from models (placeholder)
            predictions = [{"model": "model1", "output": "prediction1"},
                        {"model": "model2", "output": "prediction2"}]
            
            # Neutralize bias in output
            neutralized_predictions = self.bias_neutralizer.neutralize(predictions, bias_report)
            
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
    
    def load_config(self, config_path=None):
        """
        Load configuration from YAML files.
        
        Args:
            config_path (str): Path to the configuration file.
            
        Returns:
            dict: The loaded configuration.
        """
        if not config_path:
            config_path = os.environ.get('EV0X_CONFIG_PATH', 'config/production.yml')
        
        if not os.path.exists(config_path):
            logger.warning(f"Config file {config_path} not found. Using default configuration.")
            return {}
        
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            logger.info(f"Loaded configuration from {config_path}")
            return config
        except Exception as e:
            logger.error(f"Error loading configuration: {str(e)}")
            return {}
    
    def start_server(self, host='0.0.0.0', port=5000, debug=False):
        """
        Start the API server.
        
        Args:
            host (str): The hostname to bind to.
            port (int): The port to bind to.
            debug (bool): Whether to run in debug mode.
        """
        # Override parameters with config if available
        server_config = self.config.get('api', {}).get('server', {})
        host = server_config.get('host', host)
        port = server_config.get('port', port)
        debug = server_config.get('debug', debug)
        
        # Ensure host is set to 0.0.0.0 if not specified in config
        if not host or host == '*******':
            host = '0.0.0.0'
            
        logger.info(f"Starting server on {host}:{port} (debug={debug})")
        self.app.run(host=host, port=port, debug=debug)


# Create a global instance for backward compatibility
app = Flask(__name__)
_api_server = None

def get_api_server(config_path=None):
    """
    Get or create an APIServer instance.
    
    This function ensures only one global APIServer instance is created.
    
    Args:
        config_path (str): Path to the configuration file.
        
    Returns:
        APIServer: The API server instance.
    """
    global _api_server
    if _api_server is None:
        _api_server = APIServer(config_path)
    return _api_server

# Set up global route handlers for backward compatibility
@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return get_api_server().health_check()

@app.route('/predict', methods=['POST'])
def predict():
    """Endpoint for generating predictions using the ev0x system."""
    return get_api_server().predict()

def load_config(config_path=None):
    """Load configuration from YAML files."""
    return get_api_server(config_path).load_config(config_path)

def start_server(host='0.0.0.0', port=5000, debug=False, config_path=None):
    """Start the API server."""
    api_server = get_api_server(config_path)
    api_server.start_server(host=host, port=port, debug=debug)

if __name__ == '__main__':
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Start the ev0x API server')
    parser.add_argument('-c', '--config', dest='config_path', 
                        default=os.environ.get('EV0X_CONFIG_PATH', 'config/production.yml'),
                        help='Path to the configuration file (default: config/production.yml or EV0X_CONFIG_PATH env variable)')
    parser.add_argument('-d', '--debug', action='store_true',
                        help='Run the server in debug mode')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Start server with parsed arguments
    start_server(debug=args.debug, config_path=args.config_path)
