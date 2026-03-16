"""AWS S3/DynamoDB integration for camera trap image pipeline.

Manages the cloud-side workflow:
    1. Upload raw images from field cameras to S3 (organized by camera/date)
    2. Trigger Lambda-based classification on upload (via S3 event notification)
    3. Store classification results and metadata in DynamoDB
    4. Configure S3 lifecycle rules for cost-effective long-term archival

This module assumes the AWS infrastructure is provisioned via the
CloudFormation template in infrastructure/cloudformation.yaml.
"""

from __future__ import annotations

import json
import logging
import mimetypes
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

import boto3
from botocore.exceptions import ClientError

logger = logging.getLogger(__name__)

# Default AWS resource names (override via environment or config).
DEFAULT_BUCKET = "wildeye-camera-trap-images"
DEFAULT_TABLE = "wildeye-classifications"
DEFAULT_REGION = "us-west-2"


class S3ImageStore:
    """Manages camera trap image storage in Amazon S3.

    Images are organized by camera station and date for efficient querying:
        s3://{bucket}/raw/{camera_id}/{YYYY-MM-DD}/{image_uuid}.jpg

    Processed/classified images can be moved to a separate prefix:
        s3://{bucket}/classified/{camera_id}/{YYYY-MM-DD}/{image_uuid}.jpg

    Attributes:
        bucket_name: S3 bucket for image storage.
        region: AWS region.
    """

    def __init__(
        self,
        bucket_name: str = DEFAULT_BUCKET,
        region: str = DEFAULT_REGION,
    ) -> None:
        self.bucket_name = bucket_name
        self.region = region
        self._s3 = boto3.client("s3", region_name=region)

    def upload_image(
        self,
        local_path: Path,
        camera_id: str,
        capture_time: datetime | None = None,
        image_id: str | None = None,
    ) -> str:
        """Upload a camera trap image to S3.

        Args:
            local_path: Path to the image file on disk.
            camera_id: Camera station identifier.
            capture_time: Image capture timestamp. Defaults to now.
            image_id: Unique image identifier. Auto-generated if None.

        Returns:
            S3 key of the uploaded object.
        """
        if capture_time is None:
            capture_time = datetime.utcnow()
        if image_id is None:
            image_id = str(uuid.uuid4())

        date_str = capture_time.strftime("%Y-%m-%d")
        extension = local_path.suffix.lower() or ".jpg"
        s3_key = f"raw/{camera_id}/{date_str}/{image_id}{extension}"

        content_type = mimetypes.guess_type(str(local_path))[0] or "image/jpeg"

        self._s3.upload_file(
            Filename=str(local_path),
            Bucket=self.bucket_name,
            Key=s3_key,
            ExtraArgs={
                "ContentType": content_type,
                "Metadata": {
                    "camera_id": camera_id,
                    "capture_time": capture_time.isoformat(),
                    "image_id": image_id,
                },
            },
        )
        logger.info("Uploaded %s -> s3://%s/%s", local_path, self.bucket_name, s3_key)
        return s3_key

    def upload_directory(
        self,
        directory: Path,
        camera_id: str,
        extensions: set[str] | None = None,
    ) -> list[str]:
        """Upload all images from a directory (e.g., SD card dump) to S3.

        Args:
            directory: Local directory containing images.
            camera_id: Camera station identifier.
            extensions: Allowed file extensions. Defaults to common image types.

        Returns:
            List of S3 keys for uploaded objects.
        """
        if extensions is None:
            extensions = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}

        image_files = sorted(
            f for f in Path(directory).rglob("*") if f.suffix.lower() in extensions
        )

        keys: list[str] = []
        for image_path in image_files:
            try:
                s3_key = self.upload_image(
                    local_path=image_path,
                    camera_id=camera_id,
                )
                keys.append(s3_key)
            except ClientError:
                logger.exception("Failed to upload %s", image_path)

        logger.info(
            "Uploaded %d/%d images from %s",
            len(keys),
            len(image_files),
            directory,
        )
        return keys

    def download_image(self, s3_key: str, local_path: Path) -> Path:
        """Download a single image from S3.

        Args:
            s3_key: S3 object key.
            local_path: Local destination path.

        Returns:
            Path to the downloaded file.
        """
        local_path.parent.mkdir(parents=True, exist_ok=True)
        self._s3.download_file(self.bucket_name, s3_key, str(local_path))
        logger.debug("Downloaded s3://%s/%s -> %s", self.bucket_name, s3_key, local_path)
        return local_path

    def list_images(
        self,
        camera_id: str | None = None,
        date: str | None = None,
        prefix: str = "raw",
    ) -> list[str]:
        """List image keys in S3, optionally filtered by camera and date.

        Args:
            camera_id: Filter to a specific camera station.
            date: Filter to a specific date (YYYY-MM-DD format).
            prefix: S3 key prefix ('raw' or 'classified').

        Returns:
            List of S3 object keys.
        """
        search_prefix = prefix
        if camera_id:
            search_prefix = f"{prefix}/{camera_id}"
            if date:
                search_prefix = f"{prefix}/{camera_id}/{date}"

        keys: list[str] = []
        paginator = self._s3.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=self.bucket_name, Prefix=search_prefix):
            for obj in page.get("Contents", []):
                keys.append(obj["Key"])

        return keys

    def generate_presigned_url(self, s3_key: str, expiration: int = 3600) -> str:
        """Generate a presigned URL for temporary image access.

        Args:
            s3_key: S3 object key.
            expiration: URL validity in seconds (default 1 hour).

        Returns:
            Presigned URL string.
        """
        url: str = self._s3.generate_presigned_url(
            "get_object",
            Params={"Bucket": self.bucket_name, "Key": s3_key},
            ExpiresIn=expiration,
        )
        return url

    def configure_lifecycle_rules(self) -> None:
        """Configure S3 lifecycle rules for cost optimization.

        Lifecycle policy:
            - Raw images: Move to S3 Intelligent-Tiering after 30 days
            - Raw images: Move to S3 Glacier after 180 days
            - Raw images: Expire after 5 years (configurable by study)
            - Classified images: Keep in Standard for 90 days, then IA
        """
        lifecycle_config = {
            "Rules": [
                {
                    "ID": "raw-to-intelligent-tiering",
                    "Status": "Enabled",
                    "Filter": {"Prefix": "raw/"},
                    "Transitions": [
                        {
                            "Days": 30,
                            "StorageClass": "INTELLIGENT_TIERING",
                        },
                        {
                            "Days": 180,
                            "StorageClass": "GLACIER",
                        },
                    ],
                    "Expiration": {"Days": 1825},  # ~5 years
                },
                {
                    "ID": "classified-to-ia",
                    "Status": "Enabled",
                    "Filter": {"Prefix": "classified/"},
                    "Transitions": [
                        {
                            "Days": 90,
                            "StorageClass": "STANDARD_IA",
                        },
                    ],
                },
            ]
        }

        self._s3.put_bucket_lifecycle_configuration(
            Bucket=self.bucket_name,
            LifecycleConfiguration=lifecycle_config,
        )
        logger.info("Configured lifecycle rules for bucket %s", self.bucket_name)


class DynamoDBResultStore:
    """Stores classification results in Amazon DynamoDB.

    Table schema:
        Partition key: camera_id (String)
        Sort key: timestamp (String, ISO 8601)

    Each item stores the classification result, image metadata, and
    a reference to the S3 object for the source image.
    """

    def __init__(
        self,
        table_name: str = DEFAULT_TABLE,
        region: str = DEFAULT_REGION,
    ) -> None:
        self.table_name = table_name
        self.region = region
        self._dynamodb = boto3.resource("dynamodb", region_name=region)
        self._table = self._dynamodb.Table(table_name)

    def store_classification(
        self,
        camera_id: str,
        timestamp: datetime,
        species: list[str],
        probabilities: dict[str, float],
        s3_key: str,
        latitude: float | None = None,
        longitude: float | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Store a classification result in DynamoDB.

        Args:
            camera_id: Camera station identifier.
            timestamp: Detection timestamp.
            species: List of detected species labels.
            probabilities: Per-species confidence scores.
            s3_key: S3 key of the source image.
            latitude: Camera GPS latitude.
            longitude: Camera GPS longitude.
            metadata: Additional metadata (EXIF, processing info).

        Returns:
            The detection_id (UUID) of the stored record.
        """
        detection_id = str(uuid.uuid4())

        item: dict[str, Any] = {
            "camera_id": camera_id,
            "timestamp": timestamp.isoformat(),
            "detection_id": detection_id,
            "species": species,
            "top_species": species[0] if species else "empty",
            "probabilities": json.loads(json.dumps(probabilities)),
            "s3_key": s3_key,
            "created_at": datetime.utcnow().isoformat(),
        }

        if latitude is not None:
            item["latitude"] = str(latitude)
        if longitude is not None:
            item["longitude"] = str(longitude)
        if metadata:
            item["metadata"] = json.loads(json.dumps(metadata, default=str))

        self._table.put_item(Item=item)
        logger.info(
            "Stored classification %s: camera=%s, species=%s",
            detection_id,
            camera_id,
            species,
        )
        return detection_id

    def query_by_camera(
        self,
        camera_id: str,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """Query classification results for a specific camera station.

        Args:
            camera_id: Camera station identifier.
            start_time: Inclusive start of time range.
            end_time: Inclusive end of time range.
            limit: Maximum number of results to return.

        Returns:
            List of DynamoDB items matching the query.
        """
        key_condition = boto3.dynamodb.conditions.Key("camera_id").eq(camera_id)

        if start_time and end_time:
            key_condition &= boto3.dynamodb.conditions.Key("timestamp").between(
                start_time.isoformat(), end_time.isoformat()
            )
        elif start_time:
            key_condition &= boto3.dynamodb.conditions.Key("timestamp").gte(start_time.isoformat())
        elif end_time:
            key_condition &= boto3.dynamodb.conditions.Key("timestamp").lte(end_time.isoformat())

        response = self._table.query(
            KeyConditionExpression=key_condition,
            Limit=limit,
            ScanIndexForward=False,  # Most recent first.
        )
        items: list[dict[str, Any]] = response.get("Items", [])
        return items

    def query_by_species(
        self,
        species: str,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """Query all detections of a specific species across cameras.

        Requires a Global Secondary Index (GSI) on top_species.

        Args:
            species: Species label to query.
            limit: Maximum number of results.

        Returns:
            List of DynamoDB items.
        """
        response = self._table.query(
            IndexName="species-index",
            KeyConditionExpression=boto3.dynamodb.conditions.Key("top_species").eq(species),
            Limit=limit,
            ScanIndexForward=False,
        )
        items: list[dict[str, Any]] = response.get("Items", [])
        return items

    def get_recent_sightings(self, limit: int = 50) -> list[dict[str, Any]]:
        """Retrieve the most recent sightings across all cameras.

        Uses a table scan (expensive) -- for production, consider a GSI
        on a date partition or maintain a 'latest sightings' table.

        Args:
            limit: Maximum number of sightings to return.

        Returns:
            List of DynamoDB items sorted by timestamp descending.
        """
        response = self._table.scan(Limit=limit)
        items: list[dict[str, Any]] = response.get("Items", [])
        items.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
        return items[:limit]


def lambda_handler(event: dict[str, Any], context: Any) -> dict[str, Any]:
    """AWS Lambda handler for S3-triggered classification.

    Triggered by S3 PutObject events in the raw/ prefix. Downloads
    the uploaded image, runs the classifier, and stores results in
    DynamoDB.

    Environment variables expected:
        MODEL_PATH: S3 key or local path to the ONNX model.
        TABLE_NAME: DynamoDB table name.
        BUCKET_NAME: S3 bucket name.

    Args:
        event: S3 event notification payload.
        context: Lambda execution context.

    Returns:
        Classification results for each processed image.
    """
    import os
    import tempfile

    bucket = os.environ.get("BUCKET_NAME", DEFAULT_BUCKET)
    table = os.environ.get("TABLE_NAME", DEFAULT_TABLE)

    s3_store = S3ImageStore(bucket_name=bucket)
    db_store = DynamoDBResultStore(table_name=table)

    results: list[dict[str, Any]] = []

    for record in event.get("Records", []):
        s3_info = record.get("s3", {})
        s3_key = s3_info.get("object", {}).get("key", "")

        if not s3_key.startswith("raw/"):
            logger.info("Skipping non-raw object: %s", s3_key)
            continue

        # Parse camera_id from S3 key: raw/{camera_id}/{date}/{filename}
        parts = s3_key.split("/")
        camera_id = parts[1] if len(parts) >= 3 else "unknown"

        logger.info("Processing %s from camera %s", s3_key, camera_id)

        try:
            # Download image to temp file for classification.
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
                tmp_path = Path(tmp.name)

            s3_store.download_image(s3_key, tmp_path)

            # Run classification using ONNX runtime for Lambda efficiency.
            classification = _classify_with_onnx(tmp_path)

            # Store result in DynamoDB.
            detection_id = db_store.store_classification(
                camera_id=camera_id,
                timestamp=datetime.utcnow(),
                species=classification["species"],
                probabilities=classification["probabilities"],
                s3_key=s3_key,
            )

            results.append(
                {
                    "s3_key": s3_key,
                    "camera_id": camera_id,
                    "detection_id": detection_id,
                    "species": classification["species"],
                }
            )

        except Exception:
            logger.exception("Failed to process %s", s3_key)
            results.append(
                {
                    "s3_key": s3_key,
                    "camera_id": camera_id,
                    "error": "classification_failed",
                }
            )
        finally:
            if tmp_path.exists():
                tmp_path.unlink()

    return {
        "statusCode": 200,
        "body": json.dumps({"processed": len(results), "results": results}),
    }


def _classify_with_onnx(image_path: Path) -> dict[str, Any]:
    """Classify an image using the ONNX Runtime model.

    This is used in the Lambda function where a full PyTorch installation
    is not available. ONNX Runtime is much lighter (~50MB vs ~2GB).

    Args:
        image_path: Path to the image file.

    Returns:
        Dictionary with 'species' (list) and 'probabilities' (dict).
    """
    import numpy as np
    import onnxruntime as ort
    from PIL import Image

    from wild_eye import SPECIES_LABELS
    from wild_eye.models.species_classifier import (
        IMAGENET_MEAN,
        IMAGENET_STD,
        INPUT_SIZE,
    )

    # Load and preprocess image.
    image = Image.open(image_path).convert("RGB")
    image = image.resize((INPUT_SIZE, INPUT_SIZE))
    img_array = np.array(image, dtype=np.float32) / 255.0

    # Normalize with ImageNet statistics.
    mean = np.array(IMAGENET_MEAN, dtype=np.float32)
    std = np.array(IMAGENET_STD, dtype=np.float32)
    img_array = (img_array - mean) / std

    # Convert to NCHW format (batch, channels, height, width).
    img_array = np.transpose(img_array, (2, 0, 1))
    img_array = np.expand_dims(img_array, axis=0)

    # Run ONNX inference.
    import os

    model_path = os.environ.get("MODEL_PATH", "/opt/model/wildeye.onnx")
    session = ort.InferenceSession(model_path)
    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: img_array})
    logits = outputs[0][0]

    # Apply sigmoid for multi-label probabilities.
    probs = 1.0 / (1.0 + np.exp(-logits))

    prob_dict = {SPECIES_LABELS[i]: float(probs[i]) for i in range(len(SPECIES_LABELS))}

    detected = [
        label
        for label, prob in prob_dict.items()
        if prob >= 0.5 and label not in ("empty", "human")
    ]

    return {"species": detected, "probabilities": prob_dict}
