FROM public.ecr.aws/lambda/python:3.12

# Install uv
RUN pip install uv

# Copy requirements and install dependencies
COPY pyproject.toml ${LAMBDA_TASK_ROOT}/
WORKDIR ${LAMBDA_TASK_ROOT}
RUN python -m uv pip install --system -e .

# Copy function code
COPY lambda_function.py ${LAMBDA_TASK_ROOT}/

# Set the handler
CMD [ "lambda_function.lambda_handler" ]
