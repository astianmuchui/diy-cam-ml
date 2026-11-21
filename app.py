import os

from flask import Flask, jsonify
from routes.detect import detect_bp
from routes.diagnostics import diagnostics_bp
from routes.catchlogs import catchlogs_bp
from routes.photos import photos_bp
from routes.insights import insights_bp
from detection.vision import init_models
import config


def create_app() -> Flask:
    app = Flask(__name__)

    model_dir = os.path.dirname(config.SCENE_MODEL_PATH) or "models"
    init_models(model_dir)

    app.register_blueprint(detect_bp,      url_prefix="/api/v1")
    app.register_blueprint(diagnostics_bp, url_prefix="/api/v1/analytics")
    app.register_blueprint(catchlogs_bp,   url_prefix="/api/v1")
    app.register_blueprint(photos_bp,      url_prefix="/api/v1")
    app.register_blueprint(insights_bp,    url_prefix="/api/v1")

    @app.route("/health")
    def health():
        return jsonify({"status": "ok"})

    @app.errorhandler(400)
    def bad_request(e):
        return jsonify({"error": str(e)}), 400

    @app.errorhandler(500)
    def internal_error(e):
        return jsonify({"error": "internal server error"}), 500

    return app


if __name__ == "__main__":
    app = create_app()
    app.run(host="0.0.0.0", port=5000, debug=False)
