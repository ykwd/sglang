#!/usr/bin/env python3
"""
Benchmark client for MooncakeStore to set and get 16MB of data.
"""

import logging
import os
import time

import torch

from sglang.srt.mem_cache.storage.mooncake_store.mooncake_store import MooncakeStore

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def create_4gb_buffer_tensor() -> torch.Tensor:
    """Create a 4GB tensor for buffer registration."""
    # 4GB = 4 * 1024 * 1024 * 1024 bytes
    # For float32, each element is 4 bytes, so we need 1024 * 1024 * 1024 elements
    num_elements = 1024 * 1024 * 1024  # 4GB / 4 bytes per float32
    tensor = torch.zeros(num_elements, dtype=torch.float32)

    # Move to GPU if available
    if torch.cuda.is_available():
        tensor = tensor.cuda()
        logger.info(f"Created 4GB buffer tensor on GPU: {tensor.device}")
    else:
        logger.info("Created 4GB buffer tensor on CPU")

    return tensor


def create_16mb_tensor() -> torch.Tensor:
    """Create a tensor with 16MB of data."""
    # 16MB = 16 * 1024 * 1024 bytes
    # For float32, each element is 4 bytes, so we need 4 * 1024 * 1024 elements
    num_elements = 4 * 1024 * 1024  # 16MB / 4 bytes per float32
    tensor = torch.randn(num_elements, dtype=torch.float32)

    # Move to GPU if available
    if torch.cuda.is_available():
        tensor = tensor.cuda()
        logger.info(f"Created 16MB tensor on GPU: {tensor.device}")
    else:
        logger.info("Created 16MB tensor on CPU")

    return tensor


def run_benchmark_with_slice_size(
    store, set_buffer_tensor, get_buffer_tensor, slice_size_bytes: int
):
    """
    Run benchmark with a specific slice size.

    Args:
        store: MooncakeStore instance
        set_buffer_tensor: 4GB buffer for set operations
        get_buffer_tensor: 4GB buffer for get operations
        slice_size_bytes: Size of each slice in bytes

    Returns:
        dict: Benchmark results
    """
    # Calculate parameters based on slice size
    slice_elements = slice_size_bytes // 4  # For float32
    batch_size_bytes = 128 * 1024 * 1024  # 128MB batch size
    slices_per_batch = batch_size_bytes // slice_size_bytes
    total_slices = (4 * 1024 * 1024 * 1024) // slice_size_bytes  # 4GB / slice_size
    num_batches = total_slices // slices_per_batch

    slice_size_mb = slice_size_bytes / (1024 * 1024)

    logger.info(f"Running benchmark with slice size: {slice_size_mb:.2f} MB")
    logger.info(f"Slices per batch: {slices_per_batch}")
    logger.info(f"Total slices: {total_slices}")
    logger.info(f"Number of batches: {num_batches}")

    # Fill set buffer with random data
    logger.info("Filling set buffer with random data...")
    set_buffer_tensor.random_()
    logger.info("Set buffer filled with random data")

    # Generate unique keys for all slices
    logger.info("Generating unique keys for all slices...")
    all_keys = [
        f"benchmark_slice_{slice_size_bytes}_{i:04d}" for i in range(total_slices)
    ]

    # Prepare batch set operations
    logger.info("Starting batch set operations...")
    set_start_time = time.time()

    for batch_idx in range(num_batches):
        batch_start = batch_idx * slices_per_batch
        batch_end = batch_start + slices_per_batch

        # Get keys for this batch
        batch_keys = all_keys[batch_start:batch_end]

        # Prepare pointers and sizes for this batch
        batch_ptrs = []
        batch_sizes = []

        for slice_idx in range(slices_per_batch):
            global_slice_idx = batch_start + slice_idx
            slice_start = global_slice_idx * slice_elements
            slice_end = slice_start + slice_elements

            # Get pointer to this slice in the set buffer
            slice_ptr = set_buffer_tensor[slice_start:slice_end].data_ptr()
            batch_ptrs.append(slice_ptr)
            batch_sizes.append(slice_size_bytes)

        # Perform batch set
        store.batch_set(
            batch_keys, target_location=batch_ptrs, target_sizes=batch_sizes
        )

        if (batch_idx + 1) % 8 == 0:  # Log progress every 8 batches
            logger.info(f"Completed {batch_idx + 1}/{num_batches} batch set operations")

    set_time = time.time() - set_start_time
    logger.info(f"All batch set operations completed in {set_time:.4f} seconds")

    # Measure exists latency for each batch
    logger.info("Measuring exists latency for each batch...")
    exists_latencies = []

    for batch_idx in range(num_batches):
        batch_start = batch_idx * slices_per_batch
        batch_end = batch_start + slices_per_batch

        # Get keys for this batch
        batch_keys = all_keys[batch_start:batch_end]

        # Measure exists latency for this batch
        exists_start_time = time.time()
        exists_result = store.exists(batch_keys)
        exists_end_time = time.time()

        exists_latency = (
            exists_end_time - exists_start_time
        ) * 1000  # Convert to milliseconds
        exists_latencies.append(exists_latency)

        if (batch_idx + 1) % 8 == 0:  # Log progress every 8 batches
            logger.info(f"Completed {batch_idx + 1}/{num_batches} exists operations")

    avg_exists_latency = sum(exists_latencies) / len(exists_latencies)
    logger.info(f"Average exists latency per batch: {avg_exists_latency:.2f} ms")
    logger.info(f"Exists latency for {slices_per_batch} slices per batch")

    # Prepare batch get operations
    logger.info("Starting batch get operations...")
    get_start_time = time.time()

    for batch_idx in range(num_batches):
        batch_start = batch_idx * slices_per_batch
        batch_end = batch_start + slices_per_batch

        # Get keys for this batch
        batch_keys = all_keys[batch_start:batch_end]

        # Prepare pointers and sizes for this batch
        batch_ptrs = []
        batch_sizes = []

        for slice_idx in range(slices_per_batch):
            global_slice_idx = batch_start + slice_idx
            slice_start = global_slice_idx * slice_elements
            slice_end = slice_start + slice_elements

            # Get pointer to this slice in the get buffer
            slice_ptr = get_buffer_tensor[slice_start:slice_end].data_ptr()
            batch_ptrs.append(slice_ptr)
            batch_sizes.append(slice_size_bytes)

        # Perform batch get
        store.batch_get(
            batch_keys, target_location=batch_ptrs, target_sizes=batch_sizes
        )

        if (batch_idx + 1) % 8 == 0:  # Log progress every 8 batches
            logger.info(f"Completed {batch_idx + 1}/{num_batches} batch get operations")

    get_time = time.time() - get_start_time
    logger.info(f"All batch get operations completed in {get_time:.4f} seconds")

    # Verify data integrity for a few sample slices
    logger.info("Verifying data integrity for sample slices...")
    sample_indices = [0, 100, 1000, 2000, 3000]  # Sample different parts of the data

    integrity_verified = True
    for sample_idx in sample_indices:
        if sample_idx >= total_slices:
            continue

        # Get original data from set buffer
        slice_start = sample_idx * slice_elements
        slice_end = slice_start + slice_elements
        original_data = set_buffer_tensor[slice_start:slice_end]

        # Get retrieved data from get buffer
        retrieved_data = get_buffer_tensor[slice_start:slice_end]

        # Compare
        if not torch.allclose(original_data, retrieved_data, atol=1e-6):
            logger.error(f"❌ Data integrity check failed for slice {sample_idx}")
            diff = torch.abs(original_data - retrieved_data)
            logger.error(f"Max difference: {torch.max(diff):.6f}")
            logger.error(f"Mean difference: {torch.mean(diff):.6f}")
            integrity_verified = False
        else:
            logger.info(f"✅ Slice {sample_idx} integrity verified")

    if integrity_verified:
        logger.info("✅ All sample slices integrity verified")
    else:
        logger.error("❌ Some sample slices failed integrity check")

    # Performance metrics
    total_data_size_gb = (total_slices * slice_size_bytes) / (1024 * 1024 * 1024)
    set_throughput = total_data_size_gb / set_time
    get_throughput = total_data_size_gb / get_time

    logger.info("=" * 60)
    logger.info(f"BENCHMARK RESULTS - Slice Size: {slice_size_mb:.2f} MB")
    logger.info("=" * 60)
    logger.info(f"Total data size: {total_data_size_gb:.2f} GB")
    logger.info(f"Number of batches: {num_batches}")
    logger.info(f"Slices per batch: {slices_per_batch}")
    logger.info(f"Slice size: {slice_size_mb:.2f} MB")
    logger.info(f"Total batch set time: {set_time:.4f} seconds")
    logger.info(f"Total batch get time: {get_time:.4f} seconds")
    logger.info(f"Batch set throughput: {set_throughput:.2f} GB/s")
    logger.info(f"Batch get throughput: {get_throughput:.2f} GB/s")
    logger.info("=" * 60)

    return {
        "slice_size_mb": slice_size_mb,
        "total_data_size_gb": total_data_size_gb,
        "num_batches": num_batches,
        "slices_per_batch": slices_per_batch,
        "set_time": set_time,
        "get_time": get_time,
        "set_throughput": set_throughput,
        "get_throughput": get_throughput,
        "avg_exists_latency": avg_exists_latency,
        "success": True,
    }


def benchmark_mooncake_store():
    """Benchmark MooncakeStore with multiple slice sizes using two 4GB buffers."""

    logger.info(
        "Starting enhanced MooncakeStore benchmark with configurable slice sizes"
    )

    try:
        # Initialize MooncakeStore
        logger.info("Initializing MooncakeStore...")
        store = MooncakeStore()
        logger.info("MooncakeStore initialized successfully")

        # Create two 4GB buffer tensors - one for set, one for get
        logger.info("Creating two 4GB buffer tensors...")
        set_buffer_tensor = create_4gb_buffer_tensor()
        get_buffer_tensor = create_4gb_buffer_tensor()
        buffer_size = set_buffer_tensor.numel() * set_buffer_tensor.element_size()
        logger.info(
            f"Each buffer tensor size: {buffer_size / (1024 * 1024 * 1024):.2f} GB"
        )

        # Register both buffers with MooncakeStore
        logger.info("Registering set buffer with MooncakeStore...")
        store.register_buffer(set_buffer_tensor)
        logger.info("Registering get buffer with MooncakeStore...")
        store.register_buffer(get_buffer_tensor)
        logger.info("Both 4GB buffers registered successfully")

        # Define slice sizes to test (in bytes)
        slice_sizes = [
            64 * 1024,  # 64KB
            256 * 1024,  # 256KB
            1024 * 1024,  # 1MB
            4 * 1024 * 1024,  # 4MB
            16 * 1024 * 1024,  # 16MB
        ]

        all_results = []

        # Run benchmark for each slice size
        for slice_size_bytes in slice_sizes:
            logger.info("=" * 80)
            logger.info(
                f"TESTING SLICE SIZE: {slice_size_bytes / (1024 * 1024):.2f} MB"
            )
            logger.info("=" * 80)

            result = run_benchmark_with_slice_size(
                store, set_buffer_tensor, get_buffer_tensor, slice_size_bytes
            )

            if result["success"]:
                all_results.append(result)
                logger.info(
                    f"✅ Benchmark completed for {result['slice_size_mb']:.2f} MB slices"
                )
            else:
                logger.error(
                    f"❌ Benchmark failed for {slice_size_bytes / (1024 * 1024):.2f} MB slices"
                )

        # Print summary of all results
        logger.info("=" * 100)
        logger.info("SUMMARY OF ALL BENCHMARK RESULTS")
        logger.info("=" * 100)
        logger.info(
            f"{'Slice Size (MB)':<15} {'Set Throughput (GB/s)':<20} {'Get Throughput (GB/s)':<20} {'Exists Latency (ms)':<20}"
        )
        logger.info("-" * 100)

        for result in all_results:
            logger.info(
                f"{result['slice_size_mb']:<15.2f} {result['set_throughput']:<20.2f} {result['get_throughput']:<20.2f} {result['avg_exists_latency']:<20.2f}"
            )

        logger.info("=" * 100)

        return {
            "all_results": all_results,
            "success": len(all_results) == len(slice_sizes),
        }

    except Exception as e:
        logger.error(f"Benchmark failed with error: {e}")
        return {"success": False, "error": str(e)}


def main():
    """Main function to run the benchmark."""
    logger.info("MooncakeStore 16MB Benchmark Client")
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
    result = benchmark_mooncake_store()

    if result["success"]:
        logger.info("Benchmark completed successfully!")
        return 0
    else:
        logger.error("Benchmark failed!")
        return 1


if __name__ == "__main__":
    exit(main())
