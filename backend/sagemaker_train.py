import os
import argparse
import time
import boto3
from sagemaker.pytorch import PyTorch
from sagemaker.tuner import HyperparameterTuner, ContinuousParameter, CategoricalParameter

def get_sagemaker_role():
    role = os.environ.get("SAGEMAKER_ROLE_ARN")
    if not role:
        raise ValueError("Please provide SAGEMAKER_ROLE_ARN environment variable")
    return role

def upload_data_to_s3(bucket_name, local_data_dir="data", s3_prefix="sports-pitch-data"):
    s3_client = boto3.client('s3')
    
    s3_paths = {}
    for split in ["train", "val", "test"]:
        local_path = os.path.join(local_data_dir, split)
        if not os.path.exists(local_path):
            print(f"Warning: {local_path} not found, skipping")
            continue
            
        s3_prefix_path = f"{s3_prefix}/{split}"
        
        for root, _, files in os.walk(local_path):
            for file in files:
                local_file_path = os.path.join(root, file)
                relative_path = os.path.relpath(local_file_path, local_data_dir)
                s3_key = f"{s3_prefix}/{relative_path}"
                
                s3_client.upload_file(local_file_path, bucket_name, s3_key)
                print(f"Uploaded {local_file_path} to s3://{bucket_name}/{s3_key}")
        
        s3_paths[split] = f"s3://{bucket_name}/{s3_prefix_path}"
    
    return s3_paths

def setup_sagemaker_training(bucket_name, s3_data_paths, hyperparameter_tuning=False, **kwargs):
    """
    Set up and run SageMaker training job with optional hyperparameter tuning
    
    Args:
        bucket_name (str): S3 bucket name
        s3_data_paths (dict): Dictionary with s3 paths for 'train', 'val', 'test'
        hyperparameter_tuning (bool): Whether to use hyperparameter tuning
        **kwargs: Additional arguments for the estimator
    """
    # Get SageMaker execution role
    role = get_sagemaker_role()
    
    # Default hyperparameters
    hyperparameters = {
        "epochs": kwargs.get("epochs", 10),
        "learning_rate": kwargs.get("learning_rate", 0.001),
        "batch_size": kwargs.get("batch_size", 32),
        "patience": kwargs.get("patience", 2)
    }
    
    # Define SageMaker PyTorch estimator
    estimator = PyTorch(
        entry_point="train.py",
        source_dir=".",  # Directory with training script
        framework_version="2.0.0",  # PyTorch version
        py_version="py39",
        role=role,
        instance_count=1,
        instance_type=kwargs.get("instance_type", "ml.g4dn.xlarge"),  # GPU instance 
        hyperparameters=hyperparameters,
        use_spot_instances=kwargs.get("use_spot_instances", True),  # Use spot instances for cost savings
        max_run=kwargs.get("max_run", 3600),  # Maximum runtime in seconds (1 hour)
        max_wait=kwargs.get("max_wait", 7200),  # Maximum wait time including spot delays (2 hours)
        checkpoint_s3_uri=f"s3://{bucket_name}/checkpoints/"
    )
    
    # Define data channels for SageMaker
    data_channels = {
        "train": s3_data_paths["train"],
        "val": s3_data_paths["val"]
    }
    
    if hyperparameter_tuning:
        # Define hyperparameter ranges
        hyperparameter_ranges = {
            "learning_rate": ContinuousParameter(1e-5, 1e-2),  # Range between 0.00001 and 0.01
            "batch_size": CategoricalParameter([16, 32, 64]),  # Try different batch sizes
        }
        
        # Set objective metric name from your training script
        objective_metric_name = "validation:Accuracy"
        
        # Create hyperparameter tuner
        tuner = HyperparameterTuner(
            estimator=estimator,
            objective_metric_name=objective_metric_name,
            hyperparameter_ranges=hyperparameter_ranges,
            objective_type="Maximize",  # We want to maximize validation accuracy
            max_jobs=kwargs.get("max_jobs", 4),  # Total number of training jobs to run
            max_parallel_jobs=kwargs.get("max_parallel_jobs", 2)  # Number of parallel jobs
        )
        
        # Launch hyperparameter tuning job
        job_name = f"sports-pitch-classifier-hpo-{int(time.time())}"
        tuner.fit(inputs=data_channels, job_name=job_name)
        
        # Get best training job
        best_training_job = tuner.best_training_job()
        print(f"Best training job: {best_training_job}")
        
        # Deploy best model if requested
        if kwargs.get("deploy", False):
            predictor = tuner.deploy(
                initial_instance_count=1,
                instance_type=kwargs.get("deployment_instance_type", "ml.t2.medium")
            )
            return predictor
    else:
        # Launch regular training job
        job_name = f"sports-pitch-classifier-{int(time.time())}"
        estimator.fit(inputs=data_channels, job_name=job_name)
        
        # Deploy model if requested
        if kwargs.get("deploy", False):
            predictor = estimator.deploy(
                initial_instance_count=1,
                instance_type=kwargs.get("deployment_instance_type", "ml.t2.medium")
            )
            return predictor

def main():
    parser = argparse.ArgumentParser(description="Run SageMaker training for Sports Pitch Classifier")
    parser.add_argument("--bucket_name", type=str, required=True, 
                        help="S3 bucket name for data and model artifacts")
    parser.add_argument("--data_dir", type=str, default="data",
                        help="Local directory containing train, val, and test data")
    parser.add_argument("--hpo", action="store_true",
                        help="Use hyperparameter optimization")
    parser.add_argument("--epochs", type=int, default=10,
                        help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=0.001,
                        help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size")
    parser.add_argument("--instance_type", type=str, default="ml.g4dn.xlarge",
                        help="SageMaker instance type for training")
    parser.add_argument("--deploy", action="store_true",
                        help="Deploy the model after training")
    
    args = parser.parse_args()
    
    # Upload data to S3
    s3_data_paths = upload_data_to_s3(args.bucket_name, args.data_dir)
    
    # Start SageMaker training
    setup_sagemaker_training(
        args.bucket_name,
        s3_data_paths,
        hyperparameter_tuning=args.hpo,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        instance_type=args.instance_type,
        deploy=args.deploy
    )

if __name__ == "__main__":
    main()
