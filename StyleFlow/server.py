import os
from custommodel import CustomModel
from sanic import Request, Sanic, response
import time
import logging
from datetime import datetime
import asyncio

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

_DEFAULT_PORT = 8002

# Load and initialize the model on startup globally
model_instance = CustomModel()

# Boolean flag to track if the server is processing a request
is_processing = False

import functools
import contextvars

async def to_thread(func, *args, **kwargs):
    loop = asyncio.get_running_loop()
    ctx = contextvars.copy_context()
    func_call = functools.partial(ctx.run, func, *args, **kwargs)
    return await loop.run_in_executor(None, func_call)

server = Sanic("server")

@server.route("/healthcheck", methods=["GET"])
def healthcheck(_: Request) -> response.JSONResponse:
    """Responds to healthcheck requests."""
    return response.json({
        "healthy": "yes", 
        "model_version": "10-23", "Training data": "Until End of Sep", 
        "Face recognition": "dlib with retina as backup, for larger sample size, dlib only", 
        "Final prediction": "softmax result of the model"
    })


@server.route("/predict", methods=["POST"])
async def predict(request: Request) -> response.JSONResponse:
    global is_processing

    # Check if the server is already processing a request
    if is_processing:
        return response.json({"message": "Server is too busy, please try again later"}, status=503)

    # Set the flag to indicate processing has started
    is_processing = True
    try:
                
        start_time = time.time()
        start_time_human = datetime.fromtimestamp(start_time).strftime('%Y-%m-%d %H:%M:%S')
        logging.info(f"Prediction started at {start_time_human}")

        inputs = request.json
        # Run predict in a separate thread to avoid blocking the event loop
        # output = await asyncio.to_thread(model_instance.predict, inputs)
        output = await to_thread(model_instance.predict, inputs)
        
        end_time = time.time()
        end_time_human = datetime.fromtimestamp(end_time).strftime('%Y-%m-%d %H:%M:%S')
        logging.info(f"Prediction ended at {end_time_human}")

        return response.json({
            "df_probability": output["df_probability"], "prediction": output["prediction"],
            'start_time': start_time_human,
            'end_time': end_time_human
        })
    except Exception as e:
        end_time = time.time()
        end_time_human = datetime.fromtimestamp(end_time).strftime('%Y-%m-%d %H:%M:%S')
        logging.error(f"Error occurred: {str(e)}, Start time: {start_time_human}, End time: {end_time_human}")

        return response.json({
            'error': str(e),
            'start_time': start_time_human,
            'end_time': end_time_human
        }, status=500)
    finally:
        # Reset the flag when done processing
        is_processing = False


def main():
    """Entry point for the server."""
    port = int(os.environ.get("SERVING_PORT", _DEFAULT_PORT))
    server.run(host="0.0.0.0", port=port, workers=1)  # Single worker to process one request at a time

if __name__ == "__main__":
    main()