# ------------------------------------------------------------
# Base image for AWS Lambda with Python 3.10 runtime
# This image is required for compatibility with Lambda's container support
# ------------------------------------------------------------
FROM public.ecr.aws/lambda/python:3.10


# ------------------------------------------------------------
# Copy project files (source code and dependencies config) into container
# Note: Copy all at once for context simplicity
# ------------------------------------------------------------
COPY . ${LAMBDA_TASK_ROOT}

# ------------------------------------------------------------
# Install Python dependencies into the Lambda function root
# --target ensures they're installed where Lambda expects them
# ------------------------------------------------------------
RUN pip3 install --no-cache-dir -r requirements.txt --target "${LAMBDA_TASK_ROOT}"

# ------------------------------------------------------------
# Define the Lambda handler
# This refers to the `handler` function in the `app.py` file
# For FastAPI, we typically use Mangum for AWS Lambda support
# ------------------------------------------------------------
CMD ["app.handler"]
