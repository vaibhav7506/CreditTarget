"""SparkSession factory with Delta Lake, AQE, UTC timezone, and conditional S3A support."""

from __future__ import annotations

import os
import sys

from delta import configure_spark_with_delta_pip
from pyspark.sql import SparkSession

from credittarget.config import AppConfig


def get_spark_session(config: AppConfig | None = None) -> SparkSession:
    """Creates or retrieves a configured SparkSession."""
    # Ensure JAVA_HOME is configured for local JVM execution if not in environment
    if "JAVA_HOME" not in os.environ or not os.path.isdir(os.environ["JAVA_HOME"]):
        adoptium = "C:\\Program Files\\Eclipse Adoptium\\jdk-17.0.21.10002-hotspot"
        if os.path.isdir(adoptium):
            os.environ["JAVA_HOME"] = adoptium
            os.environ["PATH"] = f"{adoptium}\\bin;{os.environ.get('PATH', '')}"

    # Ensure HADOOP_HOME is configured on Windows
    if "HADOOP_HOME" not in os.environ or not os.path.isdir(os.environ["HADOOP_HOME"]):
        hadoop_dir = "C:\\hadoop"
        if os.path.isdir(hadoop_dir):
            os.environ["HADOOP_HOME"] = hadoop_dir
            os.environ["PATH"] = f"{hadoop_dir}\\bin;{os.environ.get('PATH', '')}"

    # Ensure driver and workers use the identical active python executable
    os.environ["PYSPARK_PYTHON"] = sys.executable
    os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

    master = config.spark.master if config else "local[2]"
    app_name = config.spark.app_name if config else "CreditTarget"
    shuffle_partitions = str(config.spark.shuffle_partitions if config else 4)
    driver_mem = config.spark.driver_memory if config else "2g"
    storage_backend = config.storage.backend if config else "local"

    builder = (
        SparkSession.builder.appName(app_name)
        .master(master)
        .config("spark.driver.memory", driver_mem)
        .config("spark.driver.host", "127.0.0.1")
        .config("spark.driver.bindAddress", "127.0.0.1")
        .config("spark.sql.session.timeZone", "UTC")
        .config("spark.sql.adaptive.enabled", "true")
        .config("spark.sql.adaptive.coalescePartitions.enabled", "true")
        .config("spark.sql.adaptive.skewJoin.enabled", "true")
        .config("spark.sql.shuffle.partitions", shuffle_partitions)
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
        .config(
            "spark.sql.catalog.spark_catalog",
            "org.apache.spark.sql.delta.catalog.DeltaCatalog",
        )
        .config("spark.databricks.delta.retentionDurationCheck.enabled", "false")
    )

    extra_packages: list[str] = []
    if storage_backend == "s3":
        extra_packages.append("org.apache.hadoop:hadoop-aws:3.3.4")
        builder = (
            builder.config(
                "spark.hadoop.fs.s3a.impl",
                "org.apache.hadoop.fs.s3a.S3AFileSystem",
            )
            .config("spark.hadoop.fs.s3a.path.style.access", "true")
            .config("spark.hadoop.fs.s3a.fast.upload", "true")
        )

    spark = configure_spark_with_delta_pip(builder, extra_packages=extra_packages).getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")
    return spark
