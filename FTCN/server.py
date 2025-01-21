# Two server routes that OctoAI containers should have:
# a route for inference requests (e.g. ”/predict”). This route for inference requests must receive JSON inputs and JSON outputs.
# a route for health checks (e.g. ”/healthcheck”).
# Number of workers (not required). Typical best practice is to make this number some function of the # of CPU cores that the server has access to and should use.

"""HTTP Inference serving interface using sanic."""
import os

from custommodel import CustomModel
from sanic import Request, Sanic, response
import time
import logging
from datetime import datetime
import multiprocessing
import torch

multiprocessing.set_start_method("spawn", force=True)
Sanic.START_METHOD_SET = True

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

_DEFAULT_PORT = 8000
"""Default port to serve inference on."""

checkpoint = "checkpoints/ftcn_Nov_19_2024_06_34_21_epoch_1.pth"

# Load and initialize the model on startup globally, so it can be reused.
model_instance = CustomModel(checkpoint=checkpoint, max_frames=768, batch_size=6)
"""Global instance of the model to serve."""

server = Sanic("server")
"""Global instance of the web server."""
server.config.REQUEST_TIMEOUT = 180

@server.route("/healthcheck", methods=["GET"])
def healthcheck(_: Request) -> response.JSONResponse:
    """Responds to healthcheck requests.

    :param request: the incoming healthcheck request.
    :return: json responding to the healthcheck.
    """
    return response.json({
        "healthy": "yes", 
        "model_version": "ftcn_Nov_19_2024_06_34_21_epoch_1.pth", "Training data": "Until End of Sep", 
        "Face recognition": "retina", 
        "Final prediction": "softmax result of the model"
        })


@server.route("/predict", methods=["POST"])
def predict(request: Request) -> response.JSONResponse:
    """Responds to inference/prediction requests.

    :param request: the incoming request containing inputs for the model.
    :return: json containing the inference results along with start and end times.
    """

    try:
        start_time = time.time()  # Capture the start time in seconds
        start_time_human = datetime.fromtimestamp(start_time).strftime('%Y-%m-%d %H:%M:%S')  # Human-readable format

        logging.info(f"Prediction started at {start_time_human}")

        inputs = request.json
        output = model_instance.predict(inputs)  
        
        end_time = time.time()  # Capture the end time in seconds
        end_time_human = datetime.fromtimestamp(end_time).strftime('%Y-%m-%d %H:%M:%S')  # Human-readable format

        logging.info(f"Prediction ended at {end_time_human}")

        # Return the output along with both start and end times in human-readable format
        return response.json({
            "df_probability": output["df_probability"], "prediction": output["prediction"],
            'start_time': start_time_human,
            'end_time': end_time_human
        })
    except Exception as e:
        end_time = time.time()  # Capture the end time in case of an error
        end_time_human = datetime.fromtimestamp(end_time).strftime('%Y-%m-%d %H:%M:%S')  # Human-readable format

        logging.error(f"Error occurred: {str(e)}, Start time: {start_time_human}, End time: {end_time_human}")

        return response.json({
            'error': str(e),
            'start_time': start_time_human,
            'end_time': end_time_human
        }, status=500)


def main():
    """Entry point for the server."""
    port = int(os.environ.get("SERVING_PORT", _DEFAULT_PORT))
    print(f'Workers: {torch.cuda.device_count()}')
    server.run(host="0.0.0.0", port=port, workers=torch.cuda.device_count())


if __name__ == "__main__":
    main()
