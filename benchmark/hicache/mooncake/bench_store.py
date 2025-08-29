#!/usr/bin/env python3
"""
Simple direct benchmark for MooncakeDistributedStore to test batch operations.
This benchmark directly uses mooncake.store.MooncakeDistributedStore without the MooncakeStore wrapper.
"""

import logging
import os
import random
import string
import time

import torch

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Constants
BUFFER_SIZE_GB = 4
BUFFER_SIZE_BYTES = BUFFER_SIZE_GB * 1024 * 1024 * 1024
SLICE_SIZE_BYTES = 16 * 1024 * 1024  # 16MB slice


def create_buffer_tensor(buffer_size_bytes: int, name: str) -> torch.Tensor:
    """Create a buffer tensor of specified size."""
    # For float32, each element is 4 bytes
    num_elements = buffer_size_bytes // 4
    tensor = torch.zeros(num_elements, dtype=torch.float32)

    # Move to GPU if available
    if torch.cuda.is_available():
        tensor = tensor.cuda()
        logger.info(f"Created {name} buffer tensor on GPU: {tensor.device}")
    else:
        logger.info(f"Created {name} buffer tensor on CPU")

    return tensor


def verify_data_integrity(put_buffer: torch.Tensor, get_buffer: torch.Tensor) -> bool:
    """Verify that put and get buffers contain identical data."""
    logger.info("Verifying data integrity by comparing buffers...")

    # Simple comparison
    if torch.allclose(put_buffer, get_buffer, atol=1e-6):
        logger.info("✅ Data integrity verified - all data matches exactly")
        return True
    else:
        logger.error("❌ Data integrity check failed")
        return False


def run_mooncake_benchmark():
    """Run the main MooncakeDistributedStore benchmark."""

    logger.info("Starting MooncakeDistributedStore direct benchmark")
    logger.info(f"Buffer size: {BUFFER_SIZE_GB} GB")
    logger.info(f"Slice size: {SLICE_SIZE_BYTES / (1024 * 1024):.2f} MB")

    try:
        # Import MooncakeDistributedStore directly
        from mooncake.store import MooncakeDistributedStore

        # Initialize MooncakeDistributedStore
        logger.info("Initializing MooncakeDistributedStore...")
        store = MooncakeDistributedStore()

        # Load configuration from environment
        local_hostname = os.getenv("LOCAL_HOSTNAME", "localhost")
        metadata_server = os.getenv("MOONCAKE_TE_META_DATA_SERVER", "P2PHANDSHAKE")
        global_segment_size = int(
            os.getenv("MOONCAKE_GLOBAL_SEGMENT_SIZE", 4 * 1024 * 1024 * 1024)
        )
        local_buffer_size = int(
            os.getenv("MOONCAKE_LOCAL_BUFFER_SIZE", 16 * 1024 * 1024)
        )
        protocol = os.getenv("MOONCAKE_PROTOCOL", "tcp")
        device_name = os.getenv("MOONCAKE_DEVICE", "auto")
        master_server_address = os.getenv("MOONCAKE_MASTER")

        if not master_server_address:
            raise ValueError("The environment variable 'MOONCAKE_MASTER' is not set.")

        # Set auto discovery environment variables if needed
        if device_name == "auto":
            os.environ["MC_MS_AUTO_DISC"] = "1"
            os.environ["MC_MS_FILTERS"] = (
                "mlx5_bond_0, mlx5_bond_1, mlx5_bond_2, mlx5_bond_3"
            )

        logger.info("Mooncake Configuration loaded from env successfully.")

        # Setup the store
        ret_code = store.setup(
            local_hostname,
            metadata_server,
            global_segment_size,
            local_buffer_size,
            protocol,
            device_name,
            master_server_address,
        )

        if ret_code:
            logger.error(f"Failed to setup mooncake store, error code: {ret_code}")
            return False

        logger.info("Connected to Mooncake store successfully.")

        # Create put and get buffers
        logger.info(f"Creating {BUFFER_SIZE_GB} GB put and get buffers...")
        put_buffer = create_buffer_tensor(BUFFER_SIZE_BYTES, "put")
        get_buffer = create_buffer_tensor(BUFFER_SIZE_BYTES, "get")

        buffer_size_mb = put_buffer.numel() * put_buffer.element_size() / (1024 * 1024)
        logger.info(f"Each buffer size: {buffer_size_mb:.2f} MB")

        # Register both buffers with MooncakeDistributedStore
        logger.info("Registering put buffer with MooncakeDistributedStore...")
        put_buffer_ptr = put_buffer.data_ptr()
        put_buffer_size = put_buffer.numel() * put_buffer.element_size()
        ret_code = store.register_buffer(put_buffer_ptr, put_buffer_size)
        if ret_code:
            logger.error(f"Failed to register put buffer, error code: {ret_code}")
            return False

        logger.info("Registering get buffer with MooncakeDistributedStore...")
        get_buffer_ptr = get_buffer.data_ptr()
        get_buffer_size = get_buffer.numel() * get_buffer.element_size()
        ret_code = store.register_buffer(get_buffer_ptr, get_buffer_size)
        if ret_code:
            logger.error(f"Failed to register get buffer, error code: {ret_code}")
            return False

        logger.info("Both buffers registered successfully")

        # Generate random data for put buffer
        logger.info("Filling put buffer with random data...")
        put_buffer.random_()
        logger.info("Put buffer filled with random data")

        # Split buffer into slices
        slice_elements = SLICE_SIZE_BYTES // 4  # For float32
        total_slices = BUFFER_SIZE_BYTES // SLICE_SIZE_BYTES

        logger.info(f"Buffer split into {total_slices} slices")
        logger.info(f"Each slice has {slice_elements} elements")

        # Generate random keys for all slices
        keys = []
        for i in range(total_slices):
            key = "".join(random.choices(string.ascii_letters + string.digits, k=16))
            keys.append(key)

        logger.info(f"Generated {total_slices} random keys")

        # Prepare pointers and sizes for all slices
        put_ptrs = []
        put_sizes = []
        for i in range(total_slices):
            slice_start = i * slice_elements
            slice_end = slice_start + slice_elements
            slice_ptr = put_buffer[slice_start:slice_end].data_ptr()
            put_ptrs.append(slice_ptr)
            put_sizes.append(SLICE_SIZE_BYTES)

        # Perform batch put operation
        logger.info("Starting batch put operation...")
        put_start_time = time.time()

        result = store.batch_put_from(keys, put_ptrs, put_sizes)

        put_time = time.time() - put_start_time
        logger.info(f"Batch put operation completed in {put_time:.4f} seconds")

        # Check if all operations succeeded (result should be list of 0s for success)
        if not all(r == 0 for r in result):
            logger.error(f"Batch put failed, result: {result}")
            return False

        # Verify data exists
        logger.info("Verifying data exists...")
        exists_start_time = time.time()

        exists_result = store.batch_is_exist(keys)
        if not all(r == 1 for r in exists_result):
            logger.error("Data existence check failed")
            return False

        exists_time = time.time() - exists_start_time
        logger.info(
            f"Data existence verification completed in {exists_time:.4f} seconds"
        )

        # Prepare pointers and sizes for get operation
        get_ptrs = []
        get_sizes = []
        for i in range(total_slices):
            slice_start = i * slice_elements
            slice_end = slice_start + slice_elements
            slice_ptr = get_buffer[slice_start:slice_end].data_ptr()
            get_ptrs.append(slice_ptr)
            get_sizes.append(SLICE_SIZE_BYTES)

        # Perform batch get operation
        logger.info("Starting batch get operation...")
        get_start_time = time.time()

        result = store.batch_get_into(keys, get_ptrs, get_sizes)

        get_time = time.time() - get_start_time
        logger.info(f"Batch get operation completed in {get_time:.4f} seconds")

        # Check if all operations succeeded (result should be list of 0s for success)
        if not all(r >= 0 for r in result):
            logger.error(f"Batch get failed, result: {result}")
            return False

        # Verify data integrity by comparing all slices
        logger.info("Verifying data integrity...")
        integrity_verified = True
        for i in range(total_slices):
            slice_start = i * slice_elements
            slice_end = slice_start + slice_elements
            put_slice = put_buffer[slice_start:slice_end]
            get_slice = get_buffer[slice_start:slice_end]

            if not torch.allclose(put_slice, get_slice, atol=1e-6):
                logger.error(f"❌ Data integrity check failed for slice {i}")
                integrity_verified = False
                break

        if integrity_verified:
            logger.info("✅ Data integrity verified - all slice data matches exactly")
        else:
            logger.error("❌ Data integrity check failed")
            return False

        # Calculate performance metrics
        total_data_size_mb = (total_slices * SLICE_SIZE_BYTES) / (1024 * 1024)
        put_throughput = total_data_size_mb / put_time
        get_throughput = total_data_size_mb / get_time

        # Print results
        logger.info("=" * 60)
        logger.info("BENCHMARK RESULTS")
        logger.info("=" * 60)
        logger.info(f"Buffer size: {BUFFER_SIZE_GB} GB")
        logger.info(f"Total data size: {total_data_size_mb:.2f} MB")
        logger.info(f"Number of slices: {total_slices}")
        logger.info(f"Slice size: {SLICE_SIZE_BYTES / (1024 * 1024):.2f} MB")
        logger.info(f"Batch put time: {put_time:.4f} seconds")
        logger.info(f"Batch get time: {get_time:.4f} seconds")
        logger.info(f"Data verification time: {exists_time:.4f} seconds")
        logger.info(f"Batch put throughput: {put_throughput:.2f} MB/s")
        logger.info(f"Batch get throughput: {get_throughput:.2f} MB/s")
        logger.info("=" * 60)

        return True

    except ImportError as e:
        logger.error(f"Failed to import MooncakeDistributedStore: {e}")
        logger.error("Please install mooncake by following the instructions at:")
        logger.error("https://kvcache-ai.github.io/Mooncake/getting_started/build.html")
        return False
    except Exception as e:
        logger.error(f"Benchmark failed with error: {e}")
        return False


def main():
    """Main function to run the benchmark."""
    logger.info("MooncakeDistributedStore Direct Benchmark")
    logger.info("=" * 50)

    # Check environment variables
    required_env_vars = ["MOONCAKE_MASTER", "MOONCAKE_TE_META_DATA_SERVER"]

    missing_vars = []
    for var in required_env_vars:
        if not os.getenv(var):
            missing_vars.append(var)

    if missing_vars:
        logger.error(f"Missing required environment variables: {missing_vars}")
        logger.error("Please set the following environment variables:")
        logger.error("export MOONCAKE_MASTER=<master_server_address>")
        logger.error("export MOONCAKE_TE_META_DATA_SERVER=<metadata_server>")
        return 1

    # Run benchmark
    success = run_mooncake_benchmark()

    if success:
        logger.info("Benchmark completed successfully!")
        return 0
    else:
        logger.error("Benchmark failed!")
        return 1


if __name__ == "__main__":
    exit(main())
