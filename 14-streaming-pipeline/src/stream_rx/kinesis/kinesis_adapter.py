"""
AWS Kinesis integration for the StreamRx pipeline.

Provides adapters for:
- Kinesis Data Streams (put records, enhanced fan-out consumption)
- Kinesis Data Firehose (S3 delivery)
- DynamoDB checkpoint management for consumer state
"""

from __future__ import annotations

import hashlib
import time
from datetime import datetime
from typing import Any

import boto3
import orjson
from botocore.exceptions import ClientError

from stream_rx.config import KinesisConfig, get_config
from stream_rx.logging_setup import get_logger

logger = get_logger(__name__)


class KinesisProducer:
    """
    Puts records to AWS Kinesis Data Streams with batching and retry.

    Supports single and batch record submission with automatic partition
    key hashing and exponential backoff on throttling.
    """

    def __init__(
        self,
        kinesis_config: KinesisConfig | None = None,
        boto_session: boto3.Session | None = None,
    ) -> None:
        cfg = kinesis_config or get_config().kinesis
        self._config = cfg
        session = boto_session or boto3.Session(region_name=cfg.region)
        self._client = session.client("kinesis")
        self._total_put = 0
        self._total_errors = 0
        logger.info(
            "kinesis_producer_initialized",
            region=cfg.region,
            stream=cfg.prescription_stream,
        )

    def put_record(
        self,
        stream_name: str,
        data: bytes | dict[str, Any],
        partition_key: str,
        explicit_hash_key: str | None = None,
    ) -> dict[str, Any]:
        """
        Put a single record to a Kinesis Data Stream.

        Args:
            stream_name: Target Kinesis stream name.
            data: Record payload (bytes or dict to be JSON-serialized).
            partition_key: Key for shard distribution.
            explicit_hash_key: Optional explicit hash for deterministic routing.

        Returns:
            Kinesis PutRecord response.
        """
        if isinstance(data, dict):
            payload = orjson.dumps(data, default=str)
        else:
            payload = data

        params: dict[str, Any] = {
            "StreamName": stream_name,
            "Data": payload,
            "PartitionKey": partition_key,
        }
        if explicit_hash_key:
            params["ExplicitHashKey"] = explicit_hash_key

        try:
            response = self._client.put_record(**params)
            self._total_put += 1
            if self._total_put % 5000 == 0:
                logger.info(
                    "kinesis_put_progress",
                    total_put=self._total_put,
                    shard_id=response.get("ShardId"),
                )
            return response
        except ClientError as exc:
            self._total_errors += 1
            error_code = exc.response["Error"]["Code"]
            if error_code == "ProvisionedThroughputExceededException":
                logger.warning("kinesis_throttled", stream=stream_name)
                raise
            logger.error("kinesis_put_failed", error=str(exc), stream=stream_name)
            raise

    def put_records_batch(
        self,
        stream_name: str,
        records: list[dict[str, Any]],
        max_retries: int = 3,
    ) -> dict[str, Any]:
        """
        Put multiple records with automatic retry for failed items.

        Each record dict must have 'data' (bytes/dict) and 'partition_key' (str).

        Args:
            stream_name: Target Kinesis stream name.
            records: List of records to submit.
            max_retries: Maximum retry attempts for failed records.

        Returns:
            Summary of batch operation results.
        """
        kinesis_records = []
        for rec in records:
            data = rec["data"]
            if isinstance(data, dict):
                data = orjson.dumps(data, default=str)
            kinesis_records.append(
                {
                    "Data": data,
                    "PartitionKey": rec["partition_key"],
                }
            )

        total_success = 0
        total_failed = 0
        attempt = 0

        while kinesis_records and attempt < max_retries:
            try:
                response = self._client.put_records(
                    StreamName=stream_name,
                    Records=kinesis_records,
                )
            except ClientError as exc:
                logger.error("kinesis_batch_put_failed", error=str(exc), attempt=attempt)
                attempt += 1
                time.sleep(2**attempt * 0.1)
                continue

            failed_count = response.get("FailedRecordCount", 0)
            total_success += len(kinesis_records) - failed_count
            total_failed += failed_count
            self._total_put += len(kinesis_records) - failed_count

            if failed_count == 0:
                break

            # Collect failed records for retry
            retry_records = []
            for i, result in enumerate(response["Records"]):
                if "ErrorCode" in result:
                    retry_records.append(kinesis_records[i])

            kinesis_records = retry_records
            attempt += 1
            backoff = 2**attempt * 0.1
            logger.warning(
                "kinesis_batch_retry",
                failed=failed_count,
                attempt=attempt,
                backoff_sec=backoff,
            )
            time.sleep(backoff)

        if kinesis_records:
            total_failed += len(kinesis_records)
            self._total_errors += len(kinesis_records)
            logger.error(
                "kinesis_batch_records_dropped",
                dropped=len(kinesis_records),
                max_retries=max_retries,
            )

        return {
            "total_success": total_success,
            "total_failed": total_failed,
            "attempts": attempt + 1,
        }

    @property
    def stats(self) -> dict[str, int]:
        return {"total_put": self._total_put, "total_errors": self._total_errors}


class KinesisFirehoseDelivery:
    """
    Manages Kinesis Data Firehose delivery to S3.

    Wraps the Firehose API to submit records for automatic S3 delivery
    with buffering, compression, and format conversion handled by the
    Firehose delivery stream configuration.
    """

    def __init__(
        self,
        kinesis_config: KinesisConfig | None = None,
        boto_session: boto3.Session | None = None,
    ) -> None:
        cfg = kinesis_config or get_config().kinesis
        self._config = cfg
        session = boto_session or boto3.Session(region_name=cfg.region)
        self._client = session.client("firehose")
        self._total_delivered = 0
        logger.info(
            "firehose_delivery_initialized",
            delivery_stream=cfg.firehose_delivery_stream,
        )

    def put_record(self, data: bytes | dict[str, Any]) -> dict[str, Any]:
        """
        Put a single record to the Firehose delivery stream.

        Args:
            data: Record payload.

        Returns:
            Firehose PutRecord response.
        """
        if isinstance(data, dict):
            payload = orjson.dumps(data, default=str) + b"\n"
        else:
            payload = data + b"\n" if not data.endswith(b"\n") else data

        try:
            response = self._client.put_record(
                DeliveryStreamName=self._config.firehose_delivery_stream,
                Record={"Data": payload},
            )
            self._total_delivered += 1
            return response
        except ClientError as exc:
            logger.error("firehose_put_failed", error=str(exc))
            raise

    def put_records_batch(self, records: list[bytes | dict[str, Any]]) -> dict[str, Any]:
        """
        Put a batch of records to Firehose (max 500 per API call).

        Args:
            records: List of record payloads.

        Returns:
            Batch submission summary.
        """
        firehose_records = []
        for rec in records:
            if isinstance(rec, dict):
                payload = orjson.dumps(rec, default=str) + b"\n"
            else:
                payload = rec + b"\n" if not rec.endswith(b"\n") else rec
            firehose_records.append({"Data": payload})

        total_success = 0
        total_failed = 0

        # Firehose limit: 500 records per batch
        for i in range(0, len(firehose_records), 500):
            batch = firehose_records[i : i + 500]
            try:
                response = self._client.put_record_batch(
                    DeliveryStreamName=self._config.firehose_delivery_stream,
                    Records=batch,
                )
                failed = response.get("FailedPutCount", 0)
                total_success += len(batch) - failed
                total_failed += failed
                self._total_delivered += len(batch) - failed
            except ClientError as exc:
                total_failed += len(batch)
                logger.error("firehose_batch_failed", error=str(exc), batch_size=len(batch))

        return {"total_success": total_success, "total_failed": total_failed}


class EnhancedFanOutConsumer:
    """
    Kinesis Enhanced Fan-Out (EFO) consumer.

    Registers as a stream consumer and receives data via SubscribeToShard,
    providing dedicated throughput of 2 MB/sec per shard per consumer.
    """

    def __init__(
        self,
        stream_name: str,
        consumer_name: str | None = None,
        kinesis_config: KinesisConfig | None = None,
        boto_session: boto3.Session | None = None,
    ) -> None:
        cfg = kinesis_config or get_config().kinesis
        self._config = cfg
        self._stream_name = stream_name
        self._consumer_name = consumer_name or cfg.enhanced_fanout_consumer
        session = boto_session or boto3.Session(region_name=cfg.region)
        self._client = session.client("kinesis")
        self._consumer_arn: str | None = None
        self._total_received = 0
        logger.info(
            "efo_consumer_initializing",
            stream=stream_name,
            consumer=self._consumer_name,
        )

    def register(self) -> str:
        """
        Register the enhanced fan-out consumer with the stream.

        Returns:
            Consumer ARN.

        Raises:
            ClientError if registration fails.
        """
        try:
            # First try to describe existing consumer
            stream_arn = self._get_stream_arn()
            try:
                response = self._client.describe_stream_consumer(
                    StreamARN=stream_arn,
                    ConsumerName=self._consumer_name,
                )
                self._consumer_arn = response["ConsumerDescription"]["ConsumerARN"]
                logger.info("efo_consumer_exists", arn=self._consumer_arn)
                return self._consumer_arn
            except ClientError:
                pass

            response = self._client.register_stream_consumer(
                StreamARN=stream_arn,
                ConsumerName=self._consumer_name,
            )
            self._consumer_arn = response["Consumer"]["ConsumerARN"]
            logger.info("efo_consumer_registered", arn=self._consumer_arn)

            # Wait for consumer to become ACTIVE
            self._wait_for_consumer_active()
            return self._consumer_arn

        except ClientError as exc:
            logger.error("efo_registration_failed", error=str(exc))
            raise

    def _get_stream_arn(self) -> str:
        """Retrieve the stream ARN."""
        response = self._client.describe_stream(StreamName=self._stream_name)
        return response["StreamDescription"]["StreamARN"]

    def _wait_for_consumer_active(self, timeout: int = 60) -> None:
        """Poll until the consumer reaches ACTIVE status."""
        if not self._consumer_arn:
            return
        start = time.time()
        while time.time() - start < timeout:
            try:
                response = self._client.describe_stream_consumer(
                    StreamARN=self._get_stream_arn(),
                    ConsumerName=self._consumer_name,
                )
                status = response["ConsumerDescription"]["ConsumerStatus"]
                if status == "ACTIVE":
                    logger.info("efo_consumer_active", arn=self._consumer_arn)
                    return
                logger.debug("efo_consumer_status", status=status)
            except ClientError:
                pass
            time.sleep(2)
        logger.warning("efo_consumer_activation_timeout", timeout=timeout)

    def subscribe_to_shard(
        self,
        shard_id: str,
        starting_position: dict[str, Any] | None = None,
    ) -> Any:
        """
        Subscribe to a shard for push-based record delivery.

        Args:
            shard_id: The shard to subscribe to.
            starting_position: Where in the shard to start reading.

        Returns:
            EventStream with record batches.
        """
        if not self._consumer_arn:
            raise RuntimeError("Consumer not registered. Call register() first.")

        position = starting_position or {
            "Type": self._config.iterator_type,
        }

        try:
            response = self._client.subscribe_to_shard(
                ConsumerARN=self._consumer_arn,
                ShardId=shard_id,
                StartingPosition=position,
            )
            return response["EventStream"]
        except ClientError as exc:
            logger.error(
                "efo_subscribe_failed",
                shard_id=shard_id,
                error=str(exc),
            )
            raise

    def deregister(self) -> None:
        """Deregister the enhanced fan-out consumer."""
        if not self._consumer_arn:
            return
        try:
            stream_arn = self._get_stream_arn()
            self._client.deregister_stream_consumer(
                StreamARN=stream_arn,
                ConsumerName=self._consumer_name,
            )
            logger.info("efo_consumer_deregistered", arn=self._consumer_arn)
            self._consumer_arn = None
        except ClientError as exc:
            logger.error("efo_deregister_failed", error=str(exc))


class DynamoDBCheckpointStore:
    """
    Checkpoint management using DynamoDB for Kinesis consumer state.

    Stores the last processed sequence number per shard, enabling
    consumers to resume from their last checkpoint after restart.
    """

    def __init__(
        self,
        table_name: str | None = None,
        kinesis_config: KinesisConfig | None = None,
        boto_session: boto3.Session | None = None,
    ) -> None:
        cfg = kinesis_config or get_config().kinesis
        self._table_name = table_name or cfg.checkpoint_table
        session = boto_session or boto3.Session(region_name=cfg.region)
        self._dynamodb = session.resource("dynamodb")
        self._table = self._dynamodb.Table(self._table_name)
        logger.info("checkpoint_store_initialized", table=self._table_name)

    def save_checkpoint(
        self,
        stream_name: str,
        shard_id: str,
        sequence_number: str,
        consumer_name: str = "default",
    ) -> None:
        """
        Save a checkpoint for a specific shard.

        Args:
            stream_name: Kinesis stream name.
            shard_id: Shard identifier.
            sequence_number: Last processed sequence number.
            consumer_name: Name of the consumer group.
        """
        checkpoint_key = f"{stream_name}:{consumer_name}:{shard_id}"
        try:
            self._table.put_item(
                Item={
                    "checkpoint_key": checkpoint_key,
                    "stream_name": stream_name,
                    "shard_id": shard_id,
                    "consumer_name": consumer_name,
                    "sequence_number": sequence_number,
                    "last_updated": datetime.utcnow().isoformat(),
                    "checksum": hashlib.sha256(
                        f"{checkpoint_key}:{sequence_number}".encode()
                    ).hexdigest(),
                }
            )
            logger.debug(
                "checkpoint_saved",
                shard_id=shard_id,
                sequence_number=sequence_number,
            )
        except ClientError as exc:
            logger.error("checkpoint_save_failed", error=str(exc), shard_id=shard_id)
            raise

    def get_checkpoint(
        self,
        stream_name: str,
        shard_id: str,
        consumer_name: str = "default",
    ) -> str | None:
        """
        Retrieve the last checkpoint for a shard.

        Returns:
            The last sequence number, or None if no checkpoint exists.
        """
        checkpoint_key = f"{stream_name}:{consumer_name}:{shard_id}"
        try:
            response = self._table.get_item(Key={"checkpoint_key": checkpoint_key})
            item = response.get("Item")
            if item:
                return item["sequence_number"]
            return None
        except ClientError as exc:
            logger.error("checkpoint_get_failed", error=str(exc), shard_id=shard_id)
            return None

    def delete_checkpoint(
        self,
        stream_name: str,
        shard_id: str,
        consumer_name: str = "default",
    ) -> None:
        """Remove a checkpoint entry."""
        checkpoint_key = f"{stream_name}:{consumer_name}:{shard_id}"
        try:
            self._table.delete_item(Key={"checkpoint_key": checkpoint_key})
            logger.info("checkpoint_deleted", shard_id=shard_id)
        except ClientError as exc:
            logger.error("checkpoint_delete_failed", error=str(exc))

    def list_checkpoints(
        self,
        stream_name: str,
        consumer_name: str = "default",
    ) -> list[dict[str, Any]]:
        """
        List all checkpoints for a stream/consumer combination.

        Returns:
            List of checkpoint items.
        """
        try:
            response = self._table.scan(
                FilterExpression="stream_name = :sn AND consumer_name = :cn",
                ExpressionAttributeValues={
                    ":sn": stream_name,
                    ":cn": consumer_name,
                },
            )
            return response.get("Items", [])
        except ClientError as exc:
            logger.error("checkpoint_list_failed", error=str(exc))
            return []

    def ensure_table_exists(self) -> None:
        """Create the checkpoint table if it does not exist."""
        try:
            self._table.table_status  # noqa: B018 - triggers DescribeTable
            logger.info("checkpoint_table_exists", table=self._table_name)
        except ClientError:
            logger.info("creating_checkpoint_table", table=self._table_name)
            self._dynamodb.create_table(
                TableName=self._table_name,
                KeySchema=[
                    {"AttributeName": "checkpoint_key", "KeyType": "HASH"},
                ],
                AttributeDefinitions=[
                    {"AttributeName": "checkpoint_key", "AttributeType": "S"},
                ],
                BillingMode="PAY_PER_REQUEST",
            )
            waiter = self._dynamodb.meta.client.get_waiter("table_exists")
            waiter.wait(TableName=self._table_name)
            logger.info("checkpoint_table_created", table=self._table_name)
