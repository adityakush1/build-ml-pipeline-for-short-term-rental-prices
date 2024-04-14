#!/usr/bin/env python
"""
Performs basic cleaning on the data and save the results in Weights & Biases
"""
import argparse
import logging
import wandb

import pandas as pd
import tempfile
import os


logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


def go(args):

    run = wandb.init(job_type="basic_cleaning")
    run.config.update(args)

    # Download input artifact. This will also log that this script is using this
    # particular version of the artifact
    # artifact_local_path = run.use_artifact(args.input_artifact).file()

    ######################
    # YOUR CODE HERE     #
    ######################

    logger.info("Downloading artifact : %s",args.input_artifact)
    artifact = run.use_artifact(args.input_artifact)
    artifact_path = artifact.file()
    
    logger.debug("Loading artifact: %s", artifact_path)
    df = pd.read_csv(artifact_path)
    
    logger.info("Cleaning artifact : ")
    logger.debug("Cleaning Step 1 - Remove Outliers : Select rows only if price is withing min & max")
    df = df[df['price'].between(args.min_price, args.max_price)].copy()
    
    logger.debug("Cleaning Step 2 - Dropping duplicates : Drop duplicate rows")
    df = df.drop_duplicates().reset_index(drop=True)
    
    # logger.debug("Cleaning Step 3 - Dropping NA : Drop NA")
    # df = df.dropna().reset_index(drop=True)
    # We will handle this in inference pipeline
    
    #location
    idx = df['longitude'].between(-74.25, -73.50) & df['latitude'].between(40.5, 41.2)
    df = df[idx].copy()
    
    logger.info("Cleaning artifact Completed ")
    
    logger.debug("Saving file ")
    
    temp_file_name = 'clean_sample.csv'
    temp_dir = tempfile.mkdtemp(dir='./')
    df.to_csv(os.path.join(temp_dir,temp_file_name), header=True, index=False)
    
    artifact = wandb.Artifact(
        name=args.output_artifact,
        type=args.output_type,
        description=args.output_description
    )
    artifact.add_file(os.path.join(temp_dir,temp_file_name))
    
    run.log_artifact(artifact)
    
    logger.info("File added as artifact : %s",args.output_artifact)
    
    
    os.remove(os.path.join(temp_dir,temp_file_name))
    logger.debug("Removed temp file")
    
    

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="This steps cleans the data")


    parser.add_argument(
        "--input_artifact", 
        type=str,
        help="Filename for input artifact (i.e. downloaded dataset)",
        required=True
    )

    parser.add_argument(
        "--output_artifact", 
        type=str,
        help="Filename for output artifact (i.e cleaned data)",
        required=True
    )

    parser.add_argument(
        "--output_type", 
        type=str,
        help="Type of Output artifact",
        required=True
    )

    parser.add_argument(
        "--output_description", 
        type=str,
        help="Description of output artifact",
        required=True
    )

    parser.add_argument(
        "--min_price", 
        type=float,
        help="Min cut off price limit in USD",
        required=True
    )

    parser.add_argument(
        "--max_price", 
        type=float,
        help="Max cut off price limit (in USD)",
        required=True
    )
    
    
    args = parser.parse_args()

    go(args)
