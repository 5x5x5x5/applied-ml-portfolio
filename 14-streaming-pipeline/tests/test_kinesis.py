"""
Tests for AWS Kinesis integration using moto for AWS service mocking.

Covers KinesisProducer, KinesisFirehoseDelivery, and DynamoDBCheckpointStore.
"""

from __future__ import annotations

import json
from collections.abc import Generator

import boto3
import pytest
from moto import mock_aws

from stream_rx.config import KinesisConfig
from stream_rx.kinesis.kinesis_adapter import (
    DynamoDBCheckpointStore,
    KinesisProducer,
)

# ---------------------------------------------------------------------------
# AWS moto fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def aws_credentials() -> None:
    """Ensure moto credentials are set."""
    import os

    os.environ["AWS_DEFAULT_REGION"] = "us-east-1"
    os.environ["AWS_ACCESS_KEY_ID"] = "testing"
    os.environ["AWS_SECRET_ACCESS_KEY"] = "testing"


@pytest.fixture
def kinesis_stream(aws_credentials: None) -> Generator[tuple[boto3.Session, str], None, None]:
    """Create a mock Kinesis stream."""
    with mock_aws():
        session = boto3.Session(region_name="us-east-1")
        client = session.client("kinesis")
        stream_name = "test-stream"
        client.create_stream(StreamName=stream_name, ShardCount=2)

        # Wait for stream to become ACTIVE
        waiter = client.get_waiter("stream_exists")
        waiter.wait(StreamName=stream_name)

        yield session, stream_name


@pytest.fixture
def dynamodb_table(aws_credentials: None) -> Generator[tuple[boto3.Session, str], None, None]:
    """Create a mock DynamoDB checkpoint table."""
    with mock_aws():
        session = boto3.Session(region_name="us-east-1")
        dynamodb = session.resource("dynamodb")
        table_name = "test-checkpoints"

        dynamodb.create_table(
            TableName=table_name,
            KeySchema=[
                {"AttributeName": "checkpoint_key", "KeyType": "HASH"},
            ],
            AttributeDefinitions=[
                {"AttributeName": "checkpoint_key", "AttributeType": "S"},
            ],
            BillingMode="PAY_PER_REQUEST",
        )

        yield session, table_name


# ---------------------------------------------------------------------------
# KinesisProducer tests
# ---------------------------------------------------------------------------


class TestKinesisProducer:
    """Test Kinesis Data Streams producer."""

    def test_put_single_record(self, kinesis_stream: tuple[boto3.Session, str]) -> None:
        session, stream_name = kinesis_stream
        config = KinesisConfig()
        producer = KinesisProducer(kinesis_config=config, boto_session=session)

        response = producer.put_record(
            stream_name=stream_name,
            data={"event_id": "test-001", "drug": "Atorvastatin"},
            partition_key="patient-123",
        )

        assert "ShardId" in response
        assert "SequenceNumber" in response
        assert producer.stats["total_put"] == 1
        assert producer.stats["total_errors"] == 0

    def test_put_record_bytes(self, kinesis_stream: tuple[boto3.Session, str]) -> None:
        session, stream_name = kinesis_stream
        config = KinesisConfig()
        producer = KinesisProducer(kinesis_config=config, boto_session=session)

        data = json.dumps({"test": True}).encode()
        response = producer.put_record(
            stream_name=stream_name,
            data=data,
            partition_key="key-1",
        )
        assert "SequenceNumber" in response

    def test_put_records_batch(self, kinesis_stream: tuple[boto3.Session, str]) -> None:
        session, stream_name = kinesis_stream
        config = KinesisConfig()
        producer = KinesisProducer(kinesis_config=config, boto_session=session)

        records = [
            {
                "data": {"event_id": f"evt-{i}", "drug": "Test"},
                "partition_key": f"patient-{i}",
            }
            for i in range(10)
        ]

        result = producer.put_records_batch(stream_name=stream_name, records=records)

        assert result["total_success"] == 10
        assert result["total_failed"] == 0

    def test_put_records_batch_empty(self, kinesis_stream: tuple[boto3.Session, str]) -> None:
        session, stream_name = kinesis_stream
        config = KinesisConfig()
        producer = KinesisProducer(kinesis_config=config, boto_session=session)

        result = producer.put_records_batch(stream_name=stream_name, records=[])
        assert result["total_success"] == 0


# ---------------------------------------------------------------------------
# DynamoDB Checkpoint tests
# ---------------------------------------------------------------------------


class TestDynamoDBCheckpointStore:
    """Test DynamoDB-backed checkpoint management."""

    def test_save_and_retrieve_checkpoint(self, dynamodb_table: tuple[boto3.Session, str]) -> None:
        session, table_name = dynamodb_table
        store = DynamoDBCheckpointStore(
            table_name=table_name,
            boto_session=session,
        )

        store.save_checkpoint(
            stream_name="test-stream",
            shard_id="shardId-000000000001",
            sequence_number="49590338271490256608559692540925702759818972116337590274",
            consumer_name="test-consumer",
        )

        seq = store.get_checkpoint(
            stream_name="test-stream",
            shard_id="shardId-000000000001",
            consumer_name="test-consumer",
        )
        assert seq == "49590338271490256608559692540925702759818972116337590274"

    def test_get_nonexistent_checkpoint(self, dynamodb_table: tuple[boto3.Session, str]) -> None:
        session, table_name = dynamodb_table
        store = DynamoDBCheckpointStore(
            table_name=table_name,
            boto_session=session,
        )

        seq = store.get_checkpoint(
            stream_name="test-stream",
            shard_id="shardId-nonexistent",
            consumer_name="test-consumer",
        )
        assert seq is None

    def test_update_checkpoint(self, dynamodb_table: tuple[boto3.Session, str]) -> None:
        session, table_name = dynamodb_table
        store = DynamoDBCheckpointStore(
            table_name=table_name,
            boto_session=session,
        )

        # Save initial
        store.save_checkpoint("stream", "shard-0", "seq-100", "consumer")
        assert store.get_checkpoint("stream", "shard-0", "consumer") == "seq-100"

        # Update
        store.save_checkpoint("stream", "shard-0", "seq-200", "consumer")
        assert store.get_checkpoint("stream", "shard-0", "consumer") == "seq-200"

    def test_delete_checkpoint(self, dynamodb_table: tuple[boto3.Session, str]) -> None:
        session, table_name = dynamodb_table
        store = DynamoDBCheckpointStore(
            table_name=table_name,
            boto_session=session,
        )

        store.save_checkpoint("stream", "shard-0", "seq-100", "consumer")
        store.delete_checkpoint("stream", "shard-0", "consumer")

        seq = store.get_checkpoint("stream", "shard-0", "consumer")
        assert seq is None

    def test_list_checkpoints(self, dynamodb_table: tuple[boto3.Session, str]) -> None:
        session, table_name = dynamodb_table
        store = DynamoDBCheckpointStore(
            table_name=table_name,
            boto_session=session,
        )

        for i in range(3):
            store.save_checkpoint("stream", f"shard-{i}", f"seq-{i}", "consumer")

        checkpoints = store.list_checkpoints("stream", "consumer")
        assert len(checkpoints) == 3

    def test_ensure_table_exists_creates_table(self, aws_credentials: None) -> None:
        with mock_aws():
            session = boto3.Session(region_name="us-east-1")
            store = DynamoDBCheckpointStore(
                table_name="auto-created-table",
                boto_session=session,
            )
            store.ensure_table_exists()

            # Verify table was created
            dynamodb = session.resource("dynamodb")
            table = dynamodb.Table("auto-created-table")
            assert table.table_status == "ACTIVE"
