"""Lazy CUDA extension for one-launch ancestral probability-tree sampling."""
from __future__ import annotations

from functools import lru_cache
import math

import torch


CPP_SOURCE = r"""
torch::Tensor fused_tree_sample_cuda(
    torch::Tensor probabilities,
    torch::Tensor edge_parents,
    torch::Tensor edge_tokens,
    torch::Tensor uniforms,
    int64_t max_depth);

torch::Tensor fused_tree_sample_parallel_cuda(
    torch::Tensor probabilities,
    torch::Tensor edge_parents,
    torch::Tensor edge_tokens,
    torch::Tensor uniforms,
    int64_t max_depth);

torch::Tensor fused_tree_sample_scan_cuda(
    torch::Tensor probabilities,
    torch::Tensor edge_parents,
    torch::Tensor edge_tokens,
    torch::Tensor uniforms,
    int64_t max_depth);

torch::Tensor fused_tree_sample_logits_scan_cuda(
    torch::Tensor logits,
    torch::Tensor edge_parents,
    torch::Tensor edge_tokens,
    torch::Tensor uniforms,
    double temperature,
    int64_t max_depth);

torch::Tensor fused_internal_tree_sample_scan_cuda(
    torch::Tensor internal_probabilities,
    torch::Tensor internal_rows,
    torch::Tensor edge_parents,
    torch::Tensor edge_tokens,
    torch::Tensor uniforms,
    int64_t max_depth);
"""


CUDA_SOURCE = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <cub/block/block_reduce.cuh>
#include <cub/block/block_scan.cuh>
#include <algorithm>
#include <climits>
#include <cmath>

namespace cg = cooperative_groups;

template <typename scalar_t>
__global__ void fused_tree_sample_kernel(
    const scalar_t* __restrict__ probabilities,
    const int64_t* __restrict__ edge_parents,
    const int64_t* __restrict__ edge_tokens,
    const double* __restrict__ uniforms,
    int64_t* __restrict__ output,
    int node_count,
    int vocabulary,
    int max_depth) {
  extern __shared__ double chunk_sums[];
  __shared__ int current_node;
  __shared__ int finished;
  __shared__ int accepted_count;

  const int thread = threadIdx.x;
  const int chunk = (vocabulary + blockDim.x - 1) / blockDim.x;
  if (thread == 0) {
    current_node = 0;
    finished = 0;
    accepted_count = 0;
  }
  __syncthreads();

  for (int depth = 0; depth <= max_depth; ++depth) {
    if (finished) {
      break;
    }
    const int row = current_node;
    const int begin = thread * chunk;
    const int end = min(begin + chunk, vocabulary);
    double local_sum = 0.0;
    for (int token = begin; token < end; ++token) {
      local_sum += static_cast<double>(probabilities[row * vocabulary + token]);
    }
    chunk_sums[thread] = local_sum;
    __syncthreads();

    if (thread == 0) {
      double row_total = 0.0;
      for (int lane = 0; lane < blockDim.x; ++lane) {
        row_total += chunk_sums[lane];
      }
      double threshold = uniforms[depth] * row_total;
      double prefix = 0.0;
      int selected_lane = blockDim.x - 1;
      for (int lane = 0; lane < blockDim.x; ++lane) {
        const double next = prefix + chunk_sums[lane];
        if (threshold < next || lane == blockDim.x - 1) {
          selected_lane = lane;
          break;
        }
        prefix = next;
      }
      int selected_token = min(selected_lane * chunk, vocabulary - 1);
      const int selected_end = min(selected_token + chunk, vocabulary);
      for (int token = selected_token; token < selected_end; ++token) {
        prefix += static_cast<double>(probabilities[row * vocabulary + token]);
        selected_token = token;
        if (threshold < prefix) {
          break;
        }
      }

      int child = -1;
      for (int edge = 0; edge < node_count - 1; ++edge) {
        if (edge_parents[edge] == row && edge_tokens[edge] == selected_token) {
          child = edge + 1;
          break;
        }
      }
      if (child >= 0 && depth < max_depth) {
        output[accepted_count] = child;
        ++accepted_count;
        current_node = child;
      } else {
        output[max_depth] = accepted_count;
        output[max_depth + 1] = selected_token;
        finished = 1;
      }
    }
    __syncthreads();
  }
}

torch::Tensor fused_tree_sample_cuda(
    torch::Tensor probabilities,
    torch::Tensor edge_parents,
    torch::Tensor edge_tokens,
    torch::Tensor uniforms,
    int64_t max_depth) {
  TORCH_CHECK(probabilities.is_cuda(), "probabilities must be CUDA");
  TORCH_CHECK(probabilities.is_contiguous(), "probabilities must be contiguous");
  TORCH_CHECK(probabilities.dim() == 2, "probabilities must have rank two");
  TORCH_CHECK(edge_parents.is_cuda() && edge_tokens.is_cuda(), "tree edges must be CUDA");
  TORCH_CHECK(edge_parents.scalar_type() == torch::kLong && edge_tokens.scalar_type() == torch::kLong,
              "tree edges must use torch.long");
  TORCH_CHECK(edge_parents.is_contiguous() && edge_tokens.is_contiguous(), "tree edges must be contiguous");
  TORCH_CHECK(uniforms.is_cuda() && uniforms.scalar_type() == torch::kFloat64,
              "uniforms must be CUDA float64");
  TORCH_CHECK(max_depth >= 0 && uniforms.numel() >= max_depth + 1,
              "not enough uniforms for the tree depth");
  const int64_t node_count = probabilities.size(0);
  const int64_t vocabulary = probabilities.size(1);
  TORCH_CHECK(edge_parents.numel() == node_count - 1 && edge_tokens.numel() == node_count - 1,
              "tree edge count mismatch");
  TORCH_CHECK(node_count > 0 && vocabulary > 0, "empty probability tree");
  TORCH_CHECK(node_count <= INT_MAX && vocabulary <= INT_MAX && max_depth <= INT_MAX,
              "tree dimensions exceed CUDA kernel limits");

  c10::cuda::CUDAGuard device_guard(probabilities.device());
  auto output = torch::full(
      {max_depth + 2}, -1,
      torch::TensorOptions().dtype(torch::kLong).device(probabilities.device()));
  constexpr int threads = 1024;
  const size_t shared_bytes = threads * sizeof(double);
  auto stream = at::cuda::getCurrentCUDAStream();
  AT_DISPATCH_FLOATING_TYPES(probabilities.scalar_type(), "fused_tree_sample_cuda", [&] {
    fused_tree_sample_kernel<scalar_t><<<1, threads, shared_bytes, stream>>>(
        probabilities.data_ptr<scalar_t>(),
        edge_parents.data_ptr<int64_t>(),
        edge_tokens.data_ptr<int64_t>(),
        uniforms.data_ptr<double>(),
        output.data_ptr<int64_t>(),
        static_cast<int>(node_count),
        static_cast<int>(vocabulary),
        static_cast<int>(max_depth));
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

template <typename scalar_t, int BLOCK_THREADS>
__global__ void fused_tree_sample_scan_kernel(
    const scalar_t* __restrict__ probabilities,
    const int64_t* __restrict__ edge_parents,
    const int64_t* __restrict__ edge_tokens,
    const double* __restrict__ uniforms,
    int64_t* __restrict__ output,
    int node_count,
    int vocabulary,
    int max_depth) {
  using BlockScan = cub::BlockScan<double, BLOCK_THREADS>;
  __shared__ typename BlockScan::TempStorage scan_storage;
  __shared__ int current_node;
  __shared__ int finished;
  __shared__ int accepted_count;
  __shared__ int selected_token;

  const int thread = threadIdx.x;
  const int chunk = (vocabulary + BLOCK_THREADS - 1) / BLOCK_THREADS;
  if (thread == 0) {
    current_node = 0;
    finished = 0;
    accepted_count = 0;
  }
  __syncthreads();

  for (int depth = 0; depth <= max_depth; ++depth) {
    if (finished) {
      break;
    }
    const int row = current_node;
    const int begin = thread * chunk;
    const int end = min(begin + chunk, vocabulary);
    double local_sum = 0.0;
    for (int token = begin; token < end; ++token) {
      local_sum += static_cast<double>(
          probabilities[row * vocabulary + token]);
    }

    double exclusive_prefix = 0.0;
    double row_total = 0.0;
    BlockScan(scan_storage).ExclusiveSum(
        local_sum, exclusive_prefix, row_total);
    if (thread == 0) {
      selected_token = -1;
    }
    __syncthreads();

    const double threshold = uniforms[depth] * row_total;
    if (begin < end && threshold >= exclusive_prefix
        && threshold < exclusive_prefix + local_sum) {
      double prefix = exclusive_prefix;
      for (int token = begin; token < end; ++token) {
        prefix += static_cast<double>(
            probabilities[row * vocabulary + token]);
        if (threshold < prefix) {
          atomicCAS(&selected_token, -1, token);
          break;
        }
      }
    }
    __syncthreads();

    if (thread == 0) {
      // torch.rand is strictly below one.  This fallback only covers a final
      // rounding gap in a non-normalized but otherwise positive input row.
      if (selected_token < 0) {
        selected_token = vocabulary - 1;
      }
      int child = -1;
      for (int edge = 0; edge < node_count - 1; ++edge) {
        if (edge_parents[edge] == row
            && edge_tokens[edge] == selected_token) {
          child = edge + 1;
          break;
        }
      }
      if (child >= 0 && depth < max_depth) {
        output[accepted_count] = child;
        ++accepted_count;
        current_node = child;
      } else {
        output[max_depth] = accepted_count;
        output[max_depth + 1] = selected_token;
        finished = 1;
      }
    }
    __syncthreads();
  }
}

torch::Tensor fused_tree_sample_scan_cuda(
    torch::Tensor probabilities,
    torch::Tensor edge_parents,
    torch::Tensor edge_tokens,
    torch::Tensor uniforms,
    int64_t max_depth) {
  TORCH_CHECK(probabilities.is_cuda(), "probabilities must be CUDA");
  TORCH_CHECK(probabilities.is_contiguous(), "probabilities must be contiguous");
  TORCH_CHECK(probabilities.dim() == 2, "probabilities must have rank two");
  TORCH_CHECK(edge_parents.is_cuda() && edge_tokens.is_cuda(),
              "tree edges must be CUDA");
  TORCH_CHECK(edge_parents.scalar_type() == torch::kLong
              && edge_tokens.scalar_type() == torch::kLong,
              "tree edges must use torch.long");
  TORCH_CHECK(edge_parents.is_contiguous() && edge_tokens.is_contiguous(),
              "tree edges must be contiguous");
  TORCH_CHECK(uniforms.is_cuda()
              && uniforms.scalar_type() == torch::kFloat64,
              "uniforms must be CUDA float64");
  TORCH_CHECK(max_depth >= 0 && uniforms.numel() >= max_depth + 1,
              "not enough uniforms for the tree depth");
  const int64_t node_count = probabilities.size(0);
  const int64_t vocabulary = probabilities.size(1);
  TORCH_CHECK(edge_parents.numel() == node_count - 1
              && edge_tokens.numel() == node_count - 1,
              "tree edge count mismatch");
  TORCH_CHECK(node_count > 0 && vocabulary > 0,
              "empty probability tree");
  TORCH_CHECK(node_count <= INT_MAX && vocabulary <= INT_MAX
              && max_depth <= INT_MAX, "tree dimensions exceed CUDA limits");

  c10::cuda::CUDAGuard device_guard(probabilities.device());
  auto output = torch::full(
      {max_depth + 2}, -1,
      torch::TensorOptions().dtype(torch::kLong).device(probabilities.device()));
  constexpr int threads = 640;
  auto stream = at::cuda::getCurrentCUDAStream();
  AT_DISPATCH_FLOATING_TYPES(
      probabilities.scalar_type(), "fused_tree_sample_scan_cuda", [&] {
        fused_tree_sample_scan_kernel<scalar_t, threads>
            <<<1, threads, 0, stream>>>(
                probabilities.data_ptr<scalar_t>(),
                edge_parents.data_ptr<int64_t>(),
                edge_tokens.data_ptr<int64_t>(),
                uniforms.data_ptr<double>(),
                output.data_ptr<int64_t>(),
                static_cast<int>(node_count),
                static_cast<int>(vocabulary),
                static_cast<int>(max_depth));
      });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

template <typename scalar_t, int BLOCK_THREADS>
__global__ void fused_tree_sample_logits_scan_kernel(
    const scalar_t* __restrict__ logits,
    const int64_t* __restrict__ edge_parents,
    const int64_t* __restrict__ edge_tokens,
    const double* __restrict__ uniforms,
    int64_t* __restrict__ output,
    double inverse_temperature,
    int node_count,
    int vocabulary,
    int max_depth) {
  using BlockReduce = cub::BlockReduce<double, BLOCK_THREADS>;
  using BlockScan = cub::BlockScan<double, BLOCK_THREADS>;
  union Scratch {
    typename BlockReduce::TempStorage reduce;
    typename BlockScan::TempStorage scan;
  };
  __shared__ Scratch scratch;
  __shared__ double row_max_shared;
  __shared__ double row_total_shared;
  __shared__ int current_node;
  __shared__ int finished;
  __shared__ int accepted_count;
  __shared__ int selected_token;

  const int thread = threadIdx.x;
  const int chunk = (vocabulary + BLOCK_THREADS - 1) / BLOCK_THREADS;
  if (thread == 0) {
    current_node = 0;
    finished = 0;
    accepted_count = 0;
  }
  __syncthreads();

  for (int depth = 0; depth <= max_depth; ++depth) {
    if (finished) {
      break;
    }
    const int row = current_node;
    const int begin = thread * chunk;
    const int end = min(begin + chunk, vocabulary);
    double local_max = -CUDART_INF;
    for (int token = begin; token < end; ++token) {
      local_max = max(
          local_max,
          static_cast<double>(logits[row * vocabulary + token])
              * inverse_temperature);
    }
    const double reduced_max = BlockReduce(scratch.reduce).Reduce(
        local_max, cub::Max());
    if (thread == 0) {
      row_max_shared = reduced_max;
    }
    __syncthreads();

    double local_sum = 0.0;
    for (int token = begin; token < end; ++token) {
      const double scaled =
          static_cast<double>(logits[row * vocabulary + token])
              * inverse_temperature;
      local_sum += exp(scaled - row_max_shared);
    }
    double exclusive_prefix = 0.0;
    double reduced_total = 0.0;
    BlockScan(scratch.scan).ExclusiveSum(
        local_sum, exclusive_prefix, reduced_total);
    if (thread == 0) {
      row_total_shared = reduced_total;
      selected_token = -1;
    }
    __syncthreads();

    const double threshold = uniforms[depth] * row_total_shared;
    if (begin < end && threshold >= exclusive_prefix
        && threshold < exclusive_prefix + local_sum) {
      double prefix = exclusive_prefix;
      for (int token = begin; token < end; ++token) {
        const double scaled =
            static_cast<double>(logits[row * vocabulary + token])
                * inverse_temperature;
        prefix += exp(scaled - row_max_shared);
        if (threshold < prefix) {
          atomicCAS(&selected_token, -1, token);
          break;
        }
      }
    }
    __syncthreads();

    if (thread == 0) {
      if (selected_token < 0) {
        selected_token = vocabulary - 1;
      }
      int child = -1;
      for (int edge = 0; edge < node_count - 1; ++edge) {
        if (edge_parents[edge] == row
            && edge_tokens[edge] == selected_token) {
          child = edge + 1;
          break;
        }
      }
      if (child >= 0 && depth < max_depth) {
        output[accepted_count] = child;
        ++accepted_count;
        current_node = child;
      } else {
        output[max_depth] = accepted_count;
        output[max_depth + 1] = selected_token;
        finished = 1;
      }
    }
    __syncthreads();
  }
}

torch::Tensor fused_tree_sample_logits_scan_cuda(
    torch::Tensor logits,
    torch::Tensor edge_parents,
    torch::Tensor edge_tokens,
    torch::Tensor uniforms,
    double temperature,
    int64_t max_depth) {
  TORCH_CHECK(logits.is_cuda() && logits.is_contiguous(),
              "logits must be contiguous CUDA");
  TORCH_CHECK(logits.dim() == 2, "logits must have rank two");
  TORCH_CHECK(edge_parents.is_cuda() && edge_tokens.is_cuda(),
              "tree edges must be CUDA");
  TORCH_CHECK(edge_parents.scalar_type() == torch::kLong
              && edge_tokens.scalar_type() == torch::kLong,
              "tree edges must use torch.long");
  TORCH_CHECK(edge_parents.is_contiguous() && edge_tokens.is_contiguous(),
              "tree edges must be contiguous");
  TORCH_CHECK(uniforms.is_cuda()
              && uniforms.scalar_type() == torch::kFloat64,
              "uniforms must be CUDA float64");
  TORCH_CHECK(std::isfinite(temperature) && temperature > 0.0,
              "temperature must be finite and positive");
  const int64_t node_count = logits.size(0);
  const int64_t vocabulary = logits.size(1);
  TORCH_CHECK(node_count > 0 && vocabulary > 0,
              "empty logit tree");
  TORCH_CHECK(edge_parents.numel() == node_count - 1
              && edge_tokens.numel() == node_count - 1,
              "tree edge count mismatch");
  TORCH_CHECK(max_depth >= 0 && uniforms.numel() >= max_depth + 1,
              "not enough uniforms for the tree depth");
  TORCH_CHECK(node_count <= INT_MAX && vocabulary <= INT_MAX
              && max_depth <= INT_MAX, "tree dimensions exceed CUDA limits");

  c10::cuda::CUDAGuard device_guard(logits.device());
  auto output = torch::full(
      {max_depth + 2}, -1,
      torch::TensorOptions().dtype(torch::kLong).device(logits.device()));
  constexpr int threads = 640;
  const double inverse_temperature = 1.0 / temperature;
  auto stream = at::cuda::getCurrentCUDAStream();
  AT_DISPATCH_FLOATING_TYPES_AND2(
      torch::kHalf, torch::kBFloat16, logits.scalar_type(),
      "fused_tree_sample_logits_scan_cuda", [&] {
        fused_tree_sample_logits_scan_kernel<scalar_t, threads>
            <<<1, threads, 0, stream>>>(
                logits.data_ptr<scalar_t>(),
                edge_parents.data_ptr<int64_t>(),
                edge_tokens.data_ptr<int64_t>(),
                uniforms.data_ptr<double>(),
                output.data_ptr<int64_t>(),
                inverse_temperature,
                static_cast<int>(node_count),
                static_cast<int>(vocabulary),
                static_cast<int>(max_depth));
      });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

template <typename scalar_t, int BLOCK_THREADS>
__global__ void fused_internal_tree_sample_scan_kernel(
    const scalar_t* __restrict__ probabilities,
    const int64_t* __restrict__ internal_rows,
    const int64_t* __restrict__ edge_parents,
    const int64_t* __restrict__ edge_tokens,
    const double* __restrict__ uniforms,
    int64_t* __restrict__ output,
    int node_count,
    int vocabulary,
    int max_depth) {
  using BlockScan = cub::BlockScan<double, BLOCK_THREADS>;
  __shared__ typename BlockScan::TempStorage scan_storage;
  __shared__ int current_node;
  __shared__ int finished;
  __shared__ int accepted_count;
  __shared__ int selected_token;

  const int thread = threadIdx.x;
  const int chunk = (vocabulary + BLOCK_THREADS - 1) / BLOCK_THREADS;
  if (thread == 0) {
    current_node = 0;
    finished = 0;
    accepted_count = 0;
  }
  __syncthreads();

  for (int depth = 0; depth <= max_depth; ++depth) {
    if (finished) {
      break;
    }
    const int original_row = current_node;
    const int compact_row = static_cast<int>(internal_rows[original_row]);
    const int begin = thread * chunk;
    const int end = min(begin + chunk, vocabulary);
    double local_sum = 0.0;
    for (int token = begin; token < end; ++token) {
      local_sum += static_cast<double>(
          probabilities[compact_row * vocabulary + token]);
    }

    double exclusive_prefix = 0.0;
    double row_total = 0.0;
    BlockScan(scan_storage).ExclusiveSum(
        local_sum, exclusive_prefix, row_total);
    if (thread == 0) {
      selected_token = -1;
    }
    __syncthreads();

    const double threshold = uniforms[depth] * row_total;
    if (begin < end && threshold >= exclusive_prefix
        && threshold < exclusive_prefix + local_sum) {
      double prefix = exclusive_prefix;
      for (int token = begin; token < end; ++token) {
        prefix += static_cast<double>(
            probabilities[compact_row * vocabulary + token]);
        if (threshold < prefix) {
          atomicCAS(&selected_token, -1, token);
          break;
        }
      }
    }
    __syncthreads();

    if (thread == 0) {
      if (selected_token < 0) {
        selected_token = vocabulary - 1;
      }
      int child = -1;
      for (int edge = 0; edge < node_count - 1; ++edge) {
        if (edge_parents[edge] == original_row
            && edge_tokens[edge] == selected_token) {
          child = edge + 1;
          break;
        }
      }
      if (child >= 0) {
        output[accepted_count] = child;
        ++accepted_count;
        if (internal_rows[child] >= 0 && depth < max_depth) {
          current_node = child;
        } else {
          // The selected child is a leaf.  Its probability row was deliberately
          // not materialized; the caller samples that one row on demand.
          output[max_depth + 2] = child;
          finished = 1;
        }
      } else {
        output[max_depth + 1] = selected_token;
        finished = 1;
      }
      output[max_depth] = accepted_count;
    }
    __syncthreads();
  }
}

torch::Tensor fused_internal_tree_sample_scan_cuda(
    torch::Tensor internal_probabilities,
    torch::Tensor internal_rows,
    torch::Tensor edge_parents,
    torch::Tensor edge_tokens,
    torch::Tensor uniforms,
    int64_t max_depth) {
  TORCH_CHECK(internal_probabilities.is_cuda()
              && internal_probabilities.is_contiguous(),
              "internal probabilities must be contiguous CUDA");
  TORCH_CHECK(internal_probabilities.dim() == 2
              && internal_probabilities.size(0) > 0
              && internal_probabilities.size(1) > 0,
              "internal probabilities must be a nonempty matrix");
  TORCH_CHECK(internal_rows.is_cuda()
              && internal_rows.scalar_type() == torch::kLong
              && internal_rows.is_contiguous(),
              "internal row map must be contiguous CUDA long");
  TORCH_CHECK(edge_parents.is_cuda() && edge_tokens.is_cuda()
              && edge_parents.scalar_type() == torch::kLong
              && edge_tokens.scalar_type() == torch::kLong
              && edge_parents.is_contiguous() && edge_tokens.is_contiguous(),
              "tree edges must be contiguous CUDA long");
  TORCH_CHECK(uniforms.is_cuda()
              && uniforms.scalar_type() == torch::kFloat64,
              "uniforms must be CUDA float64");
  const int64_t node_count = internal_rows.numel();
  const int64_t vocabulary = internal_probabilities.size(1);
  TORCH_CHECK(node_count > 1 && edge_parents.numel() == node_count - 1
              && edge_tokens.numel() == node_count - 1,
              "tree dimensions mismatch");
  TORCH_CHECK(max_depth >= 1 && uniforms.numel() >= max_depth + 1,
              "not enough uniforms for the tree depth");
  TORCH_CHECK(node_count <= INT_MAX && vocabulary <= INT_MAX
              && max_depth <= INT_MAX, "tree dimensions exceed CUDA limits");

  c10::cuda::CUDAGuard device_guard(internal_probabilities.device());
  auto output = torch::full(
      {max_depth + 3}, -1,
      torch::TensorOptions().dtype(torch::kLong)
          .device(internal_probabilities.device()));
  constexpr int threads = 640;
  auto stream = at::cuda::getCurrentCUDAStream();
  AT_DISPATCH_FLOATING_TYPES(
      internal_probabilities.scalar_type(),
      "fused_internal_tree_sample_scan_cuda", [&] {
        fused_internal_tree_sample_scan_kernel<scalar_t, threads>
            <<<1, threads, 0, stream>>>(
                internal_probabilities.data_ptr<scalar_t>(),
                internal_rows.data_ptr<int64_t>(),
                edge_parents.data_ptr<int64_t>(),
                edge_tokens.data_ptr<int64_t>(),
                uniforms.data_ptr<double>(),
                output.data_ptr<int64_t>(),
                static_cast<int>(node_count),
                static_cast<int>(vocabulary),
                static_cast<int>(max_depth));
      });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

template <typename scalar_t>
__global__ void fused_tree_sample_cooperative_kernel(
    const scalar_t* __restrict__ probabilities,
    const int64_t* __restrict__ edge_parents,
    const int64_t* __restrict__ edge_tokens,
    const double* __restrict__ uniforms,
    int64_t* __restrict__ state,
    double* __restrict__ workspace,
    int node_count,
    int vocabulary,
    int max_depth) {
  cg::grid_group grid = cg::this_grid();
  extern __shared__ double lane_sums[];
  const int block_chunk = (vocabulary + gridDim.x - 1) / gridDim.x;

  for (int depth = 0; depth <= max_depth; ++depth) {
    if (state[max_depth + 3]) {
      break;
    }
    const int row = static_cast<int>(state[max_depth + 2]);
    const int begin = blockIdx.x * block_chunk;
    const int end = min(begin + block_chunk, vocabulary);
    double local = 0.0;
    for (int token = begin + threadIdx.x; token < end; token += blockDim.x) {
      local += static_cast<double>(probabilities[row * vocabulary + token]);
    }
    lane_sums[threadIdx.x] = local;
    __syncthreads();
    for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
      if (threadIdx.x < offset) {
        lane_sums[threadIdx.x] += lane_sums[threadIdx.x + offset];
      }
      __syncthreads();
    }
    if (threadIdx.x == 0) {
      workspace[blockIdx.x] = lane_sums[0];
    }
    grid.sync();

    // One thread chooses the vocabulary partition.  Its residual threshold is
    // then consumed by that partition's block, so no second kernel launch or
    // host decision is needed between tree depths.
    if (blockIdx.x == 0 && threadIdx.x == 0) {
      double total = 0.0;
      for (int block = 0; block < gridDim.x; ++block) {
        total += workspace[block];
      }
      const double threshold = uniforms[depth] * total;
      double prefix = 0.0;
      int selected_block = gridDim.x - 1;
      for (int block = 0; block < gridDim.x; ++block) {
        const double next = prefix + workspace[block];
        if (threshold < next || block == gridDim.x - 1) {
          selected_block = block;
          workspace[gridDim.x] = threshold - prefix;
          break;
        }
        prefix = next;
      }
      state[max_depth + 4] = selected_block;
    }
    grid.sync();

    const int selected_block = static_cast<int>(state[max_depth + 4]);
    if (blockIdx.x == selected_block) {
      const int selected_begin = selected_block * block_chunk;
      const int selected_end = min(selected_begin + block_chunk, vocabulary);
      const int lane_chunk =
          (selected_end - selected_begin + blockDim.x - 1) / blockDim.x;
      const int lane_begin = selected_begin + threadIdx.x * lane_chunk;
      const int lane_end = min(lane_begin + lane_chunk, selected_end);
      double selected_local = 0.0;
      for (int token = lane_begin; token < lane_end; ++token) {
        selected_local +=
            static_cast<double>(probabilities[row * vocabulary + token]);
      }
      lane_sums[threadIdx.x] = selected_local;
      __syncthreads();

      if (threadIdx.x == 0) {
        const double local_threshold = workspace[gridDim.x];
        double prefix = 0.0;
        int selected_lane = blockDim.x - 1;
        for (int lane = 0; lane < blockDim.x; ++lane) {
          const double next = prefix + lane_sums[lane];
          if (local_threshold < next || lane == blockDim.x - 1) {
            selected_lane = lane;
            break;
          }
          prefix = next;
        }
        int selected_token = min(
            selected_begin + selected_lane * lane_chunk, vocabulary - 1);
        const int token_end = min(selected_token + lane_chunk, selected_end);
        for (int token = selected_token; token < token_end; ++token) {
          prefix += static_cast<double>(probabilities[row * vocabulary + token]);
          selected_token = token;
          if (local_threshold < prefix) {
            break;
          }
        }

        int child = -1;
        for (int edge = 0; edge < node_count - 1; ++edge) {
          if (edge_parents[edge] == row && edge_tokens[edge] == selected_token) {
            child = edge + 1;
            break;
          }
        }
        const int accepted = static_cast<int>(state[max_depth]);
        if (child >= 0 && depth < max_depth) {
          state[accepted] = child;
          state[max_depth] = accepted + 1;
          state[max_depth + 2] = child;
        } else {
          state[max_depth + 1] = selected_token;
          state[max_depth + 3] = 1;
        }
      }
    }
    grid.sync();
  }
}

torch::Tensor fused_tree_sample_parallel_cuda(
    torch::Tensor probabilities,
    torch::Tensor edge_parents,
    torch::Tensor edge_tokens,
    torch::Tensor uniforms,
    int64_t max_depth) {
  TORCH_CHECK(probabilities.is_cuda() && probabilities.is_contiguous(),
              "probabilities must be contiguous CUDA");
  TORCH_CHECK(probabilities.dim() == 2, "probabilities must have rank two");
  TORCH_CHECK(edge_parents.is_cuda() && edge_tokens.is_cuda(), "tree edges must be CUDA");
  TORCH_CHECK(edge_parents.scalar_type() == torch::kLong && edge_tokens.scalar_type() == torch::kLong,
              "tree edges must use torch.long");
  TORCH_CHECK(uniforms.is_cuda() && uniforms.scalar_type() == torch::kFloat64,
              "uniforms must be CUDA float64");
  const int64_t node_count = probabilities.size(0);
  const int64_t vocabulary = probabilities.size(1);
  TORCH_CHECK(node_count > 0 && vocabulary > 0 && edge_parents.numel() == node_count - 1
              && edge_tokens.numel() == node_count - 1, "tree dimensions mismatch");
  TORCH_CHECK(max_depth >= 0 && uniforms.numel() >= max_depth + 1,
              "not enough uniforms for the tree depth");
  TORCH_CHECK(node_count <= INT_MAX && vocabulary <= INT_MAX && max_depth <= INT_MAX,
              "tree dimensions exceed CUDA kernel limits");

  c10::cuda::CUDAGuard device_guard(probabilities.device());
  auto state = torch::zeros(
      {max_depth + 5},
      torch::TensorOptions().dtype(torch::kLong).device(probabilities.device()));
  constexpr int requested_blocks = 64;
  constexpr int threads = 256;
  auto stream = at::cuda::getCurrentCUDAStream();
  AT_DISPATCH_FLOATING_TYPES(probabilities.scalar_type(), "fused_tree_sample_parallel_cuda", [&] {
    int device = 0;
    C10_CUDA_CHECK(cudaGetDevice(&device));
    int cooperative = 0;
    C10_CUDA_CHECK(cudaDeviceGetAttribute(
        &cooperative, cudaDevAttrCooperativeLaunch, device));
    TORCH_CHECK(cooperative, "GPU does not support cooperative launch");
    int blocks_per_sm = 0;
    C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &blocks_per_sm, fused_tree_sample_cooperative_kernel<scalar_t>,
        threads, threads * sizeof(double)));
    cudaDeviceProp properties;
    C10_CUDA_CHECK(cudaGetDeviceProperties(&properties, device));
    const int blocks = std::min(
        requested_blocks, blocks_per_sm * properties.multiProcessorCount);
    TORCH_CHECK(blocks > 0, "No cooperative kernel occupancy available");
    auto workspace = torch::empty(
        {blocks + 1},
        torch::TensorOptions().dtype(torch::kFloat64).device(probabilities.device()));
    const scalar_t* probability_ptr = probabilities.data_ptr<scalar_t>();
    const int64_t* parent_ptr = edge_parents.data_ptr<int64_t>();
    const int64_t* token_ptr = edge_tokens.data_ptr<int64_t>();
    const double* uniform_ptr = uniforms.data_ptr<double>();
    int64_t* state_ptr = state.data_ptr<int64_t>();
    double* workspace_ptr = workspace.data_ptr<double>();
    int nodes = static_cast<int>(node_count);
    int vocab = static_cast<int>(vocabulary);
    int depth = static_cast<int>(max_depth);
    void* arguments[] = {
        &probability_ptr, &parent_ptr, &token_ptr, &uniform_ptr,
        &state_ptr, &workspace_ptr, &nodes, &vocab, &depth};
    C10_CUDA_CHECK(cudaLaunchCooperativeKernel(
        reinterpret_cast<void*>(fused_tree_sample_cooperative_kernel<scalar_t>),
        dim3(blocks), dim3(threads), arguments,
        threads * sizeof(double), stream));
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return state;
}
"""


@lru_cache(maxsize=1)
def load_fused_tree_sampler():
    if not torch.cuda.is_available():
        raise RuntimeError("The fused tree sampler requires CUDA")
    from torch.utils.cpp_extension import load_inline

    return load_inline(
        name="gbv_fused_tree_sampler_v13",
        cpp_sources=[CPP_SOURCE],
        cuda_sources=[CUDA_SOURCE],
        functions=["fused_tree_sample_cuda", "fused_tree_sample_parallel_cuda",
                   "fused_tree_sample_scan_cuda",
                   "fused_tree_sample_logits_scan_cuda",
                   "fused_internal_tree_sample_scan_cuda"],
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3"],
        with_cuda=True,
        verbose=False,
    )


def _topology(parents, tokens):
    parents = list(parents)
    tokens = list(tokens)
    node_count = len(parents)
    if (node_count < 1 or len(tokens) != node_count - 1
            or parents[0] != -1
            or any(parent < 0 or parent >= node
                   for node, parent in enumerate(parents[1:], 1))
            or len(set(zip(parents[1:], tokens))) != len(tokens)):
        raise ValueError("Invalid probability-tree topology")
    depths = [0] * node_count
    for node in range(1, node_count):
        depths[node] = depths[parents[node]] + 1
    return parents, tokens, max(depths)


def tree_verify_ancestral_fused(parents, tokens, all_p, generator=None,
                                validate: bool = True):
    """Sample only the visited DDTree rows inside one persistent CUDA block."""
    parents, tokens, max_depth = _topology(parents, tokens)
    node_count = len(parents)
    if (not all_p.is_cuda or all_p.ndim != 2 or all_p.shape[0] != node_count
            or all_p.shape[1] < 1 or not all_p.is_floating_point()):
        raise ValueError("Fused DDTree probability tensor mismatch")
    if any(token < 0 or token >= all_p.shape[1] for token in tokens):
        raise ValueError("Tree token is outside the Target vocabulary")
    if validate:
        valid = (torch.isfinite(all_p).all() & (all_p >= 0).all()
                 & (all_p.sum(-1) > 0).all())
        if not bool(valid):
            raise FloatingPointError("Invalid Target probabilities for fused DDTree")

    device = all_p.device
    edge_parents = torch.tensor(parents[1:], dtype=torch.long, device=device)
    edge_tokens = torch.tensor(tokens, dtype=torch.long, device=device)
    uniforms = torch.rand(
        max_depth + 1, dtype=torch.float64, device=device, generator=generator
    )
    packed = load_fused_tree_sampler().fused_tree_sample_cuda(
        all_p.contiguous(), edge_parents, edge_tokens, uniforms, max_depth
    ).tolist()
    accepted_count = int(packed[max_depth])
    bonus = int(packed[max_depth + 1])
    if not 0 <= accepted_count <= max_depth or not 0 <= bonus < all_p.shape[1]:
        raise RuntimeError("Fused tree sampler returned invalid control values")
    nodes = [int(node) for node in packed[:accepted_count]]
    output_tokens = [tokens[node - 1] for node in nodes]
    return nodes, output_tokens, bonus


def tree_verify_ancestral_fused_parallel(parents, tokens, all_p, generator=None,
                                         validate: bool = True):
    """Traverse on device with multi-SM row reductions and no host decisions."""
    parents, tokens, max_depth = _topology(parents, tokens)
    node_count = len(parents)
    if (not all_p.is_cuda or all_p.ndim != 2 or all_p.shape[0] != node_count
            or all_p.shape[1] < 1 or not all_p.is_floating_point()):
        raise ValueError("Parallel fused DDTree probability tensor mismatch")
    if any(token < 0 or token >= all_p.shape[1] for token in tokens):
        raise ValueError("Tree token is outside the Target vocabulary")
    if validate:
        valid = (torch.isfinite(all_p).all() & (all_p >= 0).all()
                 & (all_p.sum(-1) > 0).all())
        if not bool(valid):
            raise FloatingPointError("Invalid Target probabilities for parallel fused DDTree")

    device = all_p.device
    edge_parents = torch.tensor(parents[1:], dtype=torch.long, device=device)
    edge_tokens = torch.tensor(tokens, dtype=torch.long, device=device)
    uniforms = torch.rand(
        max_depth + 1, dtype=torch.float64, device=device, generator=generator
    )
    packed = load_fused_tree_sampler().fused_tree_sample_parallel_cuda(
        all_p.contiguous(), edge_parents, edge_tokens, uniforms, max_depth
    ).tolist()
    accepted_count = int(packed[max_depth])
    bonus = int(packed[max_depth + 1])
    if not 0 <= accepted_count <= max_depth or not 0 <= bonus < all_p.shape[1]:
        raise RuntimeError("Parallel fused tree sampler returned invalid control values")
    nodes = [int(node) for node in packed[:accepted_count]]
    return nodes, [tokens[node - 1] for node in nodes], bonus


def tree_verify_ancestral_fused_scan(parents, tokens, all_p, generator=None,
                                      validate: bool = True):
    """Traverse DDTree with one persistent block and a parallel CUB row scan."""
    parents, tokens, max_depth = _topology(parents, tokens)
    node_count = len(parents)
    if (not all_p.is_cuda or all_p.ndim != 2 or all_p.shape[0] != node_count
            or all_p.shape[1] < 1 or not all_p.is_floating_point()):
        raise ValueError("Scan-fused DDTree probability tensor mismatch")
    if any(token < 0 or token >= all_p.shape[1] for token in tokens):
        raise ValueError("Tree token is outside the Target vocabulary")
    if validate:
        valid = (torch.isfinite(all_p).all() & (all_p >= 0).all()
                 & (all_p.sum(-1) > 0).all())
        if not bool(valid):
            raise FloatingPointError("Invalid Target probabilities for scan-fused DDTree")

    device = all_p.device
    edge_parents = torch.tensor(parents[1:], dtype=torch.long, device=device)
    edge_tokens = torch.tensor(tokens, dtype=torch.long, device=device)
    uniforms = torch.rand(
        max_depth + 1, dtype=torch.float64, device=device, generator=generator
    )
    packed = load_fused_tree_sampler().fused_tree_sample_scan_cuda(
        all_p.contiguous(), edge_parents, edge_tokens, uniforms, max_depth
    ).tolist()
    accepted_count = int(packed[max_depth])
    bonus = int(packed[max_depth + 1])
    if not 0 <= accepted_count <= max_depth or not 0 <= bonus < all_p.shape[1]:
        raise RuntimeError("Scan-fused tree sampler returned invalid control values")
    nodes = [int(node) for node in packed[:accepted_count]]
    return nodes, [tokens[node - 1] for node in nodes], bonus


def tree_verify_ancestral_logits_fused_scan(
        parents, tokens, all_logits, temperature: float,
        probability_dtype=torch.float64, generator=None,
        validate: bool = True):
    """Normalize and traverse only visited logit rows in one CUDA launch.

    The Target vocabulary head remains unchanged and produces the same BF16
    logits as official DDTree.  The persistent kernel performs stable FP64
    softmax weights, inverse-CDF sampling, and tree traversal without
    materializing a node-by-vocabulary FP64 probability matrix.
    """
    parents, tokens, max_depth = _topology(parents, tokens)
    node_count = len(parents)
    if (not all_logits.is_cuda or all_logits.ndim != 2
            or all_logits.shape[0] != node_count
            or all_logits.shape[1] < 1
            or not all_logits.is_floating_point()
            or not all_logits.is_contiguous()):
        raise ValueError("Direct-logits Target tensor mismatch")
    if (not isinstance(temperature, (float, int)) or temperature <= 0
            or not math.isfinite(float(temperature))):
        raise ValueError("Direct-logits temperature must be finite and positive")
    if probability_dtype != torch.float64:
        raise ValueError("Direct-logits verifier requires FP64 probability arithmetic")
    if any(token < 0 or token >= all_logits.shape[1] for token in tokens):
        raise ValueError("Tree token is outside the Target vocabulary")
    if validate and not bool(torch.isfinite(all_logits).all()):
        raise FloatingPointError("Invalid Target logits for direct-logits DDTree")

    device = all_logits.device
    edge_parents = torch.tensor(parents[1:], dtype=torch.long, device=device)
    edge_tokens = torch.tensor(tokens, dtype=torch.long, device=device)
    uniforms = torch.rand(
        max_depth + 1, dtype=torch.float64, device=device, generator=generator,
    )
    packed = load_fused_tree_sampler().fused_tree_sample_logits_scan_cuda(
        all_logits, edge_parents, edge_tokens, uniforms,
        float(temperature), max_depth,
    ).tolist()
    accepted_count = int(packed[max_depth])
    bonus = int(packed[max_depth + 1])
    if not 0 <= accepted_count <= max_depth or not 0 <= bonus < all_logits.shape[1]:
        raise RuntimeError("Direct-logits sampler returned invalid control values")
    nodes = [int(node) for node in packed[:accepted_count]]
    return nodes, [tokens[node - 1] for node in nodes], bonus, {
        "visited_probability_rows": accepted_count + 1,
        "total_tree_rows": node_count,
        "lm_head_rows": node_count,
    }


def _sample_row_inverse_cdf(probability_row, uniform):
    """Use the same FP64 inverse-CDF rule as the persistent scan kernel."""
    cumulative = probability_row.cumsum(0)
    threshold = uniform * cumulative[-1]
    token = torch.searchsorted(cumulative, threshold, right=False)
    return int(token.clamp_max(probability_row.numel() - 1).item())


def _compact_tree_metadata(parents, tokens, internal_nodes, device):
    node_count = len(parents)
    internal_map = [-1] * node_count
    for compact_row, node in enumerate(internal_nodes):
        internal_map[node] = compact_row
    if internal_map[0] != 0:
        raise ValueError("A nontrivial probability tree must have an internal root")
    # Pack every integer control array into one small host-to-device transfer.
    # Row 3 stores the compact-to-original map and is padded only so the tensor
    # remains rectangular; the kernel never reads that padding.
    return torch.tensor([
        internal_map,
        parents,
        [-1, *tokens],
        [*internal_nodes, *([-1] * (node_count - len(internal_nodes)))],
    ], dtype=torch.long, device=device)


def _tree_verify_internal_fused_scan(
        parents, tokens, max_depth, internal_nodes, metadata, internal_p, uniforms,
        validate: bool = True):
    """Walk compact internal rows and report a leaf that needs one final row."""
    node_count = len(parents)
    internal_nodes = list(internal_nodes)
    if (not internal_p.is_cuda or internal_p.ndim != 2
            or internal_p.shape[0] != len(internal_nodes)
            or internal_p.shape[1] < 1 or not internal_p.is_floating_point()):
        raise ValueError("Compact internal probability tensor mismatch")
    if internal_nodes != sorted(set(parents[1:])) or not internal_nodes:
        raise ValueError("Internal nodes do not match the probability tree")
    if any(token < 0 or token >= internal_p.shape[1] for token in tokens):
        raise ValueError("Tree token is outside the Target vocabulary")
    if (not uniforms.is_cuda or uniforms.dtype != torch.float64
            or uniforms.numel() < max_depth + 1):
        raise ValueError("Tree uniforms must be CUDA FP64 and cover the depth")
    if (metadata.shape != (4, node_count) or metadata.dtype != torch.long
            or metadata.device != internal_p.device or not metadata.is_contiguous()):
        raise ValueError("Compact tree metadata tensor mismatch")
    if validate:
        valid = (torch.isfinite(internal_p).all() & (internal_p >= 0).all()
                 & (internal_p.sum(-1) > 0).all())
        if not bool(valid):
            raise FloatingPointError("Invalid compact internal probabilities")

    packed = load_fused_tree_sampler().fused_internal_tree_sample_scan_cuda(
        internal_p.contiguous(),
        metadata[0], metadata[1, 1:], metadata[2, 1:],
        uniforms,
        max_depth,
    ).tolist()
    accepted_count = int(packed[max_depth])
    bonus = int(packed[max_depth + 1])
    terminal_leaf = int(packed[max_depth + 2])
    if not 0 <= accepted_count <= max_depth:
        raise RuntimeError("Compact fused tree sampler returned invalid path length")
    if (terminal_leaf < 0) == (bonus < 0):
        raise RuntimeError("Compact fused tree sampler returned an invalid exit")
    if terminal_leaf >= 0 and terminal_leaf in internal_nodes:
        raise RuntimeError("Compact fused tree sampler returned an internal leaf")
    if bonus >= internal_p.shape[1]:
        raise RuntimeError("Compact fused tree sampler returned an invalid bonus")
    nodes = [int(node) for node in packed[:accepted_count]]
    if terminal_leaf >= 0 and (not nodes or nodes[-1] != terminal_leaf):
        raise RuntimeError("Compact fused tree sampler omitted its terminal leaf")
    return nodes, [tokens[node - 1] for node in nodes], bonus, terminal_leaf


def tree_verify_ancestral_lazy_softmax_fused_scan(
        parents, tokens, all_logits, temperature: float,
        probability_dtype=torch.float64, generator=None,
        validate: bool = True):
    """Normalize only internal rows, scan the path, then normalize one leaf.

    The Target transformer and BF16 vocabulary head still run on the exact same
    tree-shaped batch as DDTree.  The optimization is confined to posterior
    construction: leaf softmax rows are deferred until the walk actually reaches
    one, and the internal rows are traversed by the persistent scan kernel.
    """
    from .sampling import probabilities

    parents, tokens, max_depth = _topology(parents, tokens)
    node_count = len(parents)
    if (all_logits.ndim != 2 or all_logits.shape[0] != node_count
            or all_logits.shape[1] < 1 or not all_logits.is_floating_point()):
        raise ValueError("Lazy-softmax Target logits mismatch")
    internal_nodes = sorted(set(parents[1:]))
    if not internal_nodes:
        p = probabilities(all_logits[0], temperature, probability_dtype)
        uniform = torch.rand(
            1, dtype=torch.float64, device=all_logits.device,
            generator=generator,
        )[0]
        bonus = _sample_row_inverse_cdf(p, uniform)
        return [], [], bonus, {
            "internal_projected_rows": 0,
            "leaf_projected_rows": 1,
            "projected_rows": 1,
            "total_tree_rows": 1,
            "lm_head_rows": 1,
            "probability_rows": 1,
        }

    device = all_logits.device
    metadata = _compact_tree_metadata(
        parents, tokens, internal_nodes, device,
    )
    uniforms = torch.rand(
        max_depth + 1, dtype=torch.float64, device=device,
        generator=generator,
    )
    internal_index = metadata[3, :len(internal_nodes)]
    internal_p = probabilities(
        all_logits.index_select(0, internal_index),
        temperature, probability_dtype,
    )
    nodes, output_tokens, bonus, terminal_leaf = (
        _tree_verify_internal_fused_scan(
            parents, tokens, max_depth, internal_nodes, metadata,
            internal_p, uniforms, validate,
        )
    )
    leaf_projected = 0
    if terminal_leaf >= 0:
        leaf_p = probabilities(
            all_logits[terminal_leaf], temperature, probability_dtype,
        )
        if validate:
            valid = (torch.isfinite(leaf_p).all() & (leaf_p >= 0).all()
                     & (leaf_p.sum() > 0))
            if not bool(valid):
                raise FloatingPointError("Invalid lazy-softmax leaf probabilities")
        bonus = _sample_row_inverse_cdf(leaf_p, uniforms[len(nodes)])
        leaf_projected = 1
    return nodes, output_tokens, bonus, {
        "internal_projected_rows": len(internal_nodes),
        "leaf_projected_rows": leaf_projected,
        "projected_rows": len(internal_nodes) + leaf_projected,
        "total_tree_rows": node_count,
        "lm_head_rows": node_count,
        "probability_rows": len(internal_nodes) + leaf_projected,
    }


def tree_verify_ancestral_lazy_projection_fused_scan(
        parents, tokens, final_hidden, lm_head, temperature: float,
        probability_dtype=torch.float64, generator=None,
        validate: bool = True):
    """Project compact internal rows and traverse them with the fused scan.

    This is the more aggressive sibling of lazy-softmax scan: it defers both
    the BF16 vocabulary head and FP64 normalization for all leaves, then applies
    both operations to at most the one reached leaf.
    """
    from .sampling import probabilities

    parents, tokens, max_depth = _topology(parents, tokens)
    node_count = len(parents)
    if (final_hidden.ndim != 2 or final_hidden.shape[0] != node_count
            or not final_hidden.is_floating_point()):
        raise ValueError("Lazy-projection Target hidden states mismatch")
    internal_nodes = sorted(set(parents[1:]))
    if not internal_nodes:
        p = probabilities(
            lm_head(final_hidden[:1])[0], temperature, probability_dtype,
        )
        uniform = torch.rand(
            1, dtype=torch.float64, device=final_hidden.device,
            generator=generator,
        )[0]
        bonus = _sample_row_inverse_cdf(p, uniform)
        return [], [], bonus, {
            "internal_projected_rows": 0,
            "leaf_projected_rows": 1,
            "projected_rows": 1,
            "total_tree_rows": 1,
            "lm_head_rows": 1,
            "probability_rows": 1,
        }

    device = final_hidden.device
    metadata = _compact_tree_metadata(
        parents, tokens, internal_nodes, device,
    )
    uniforms = torch.rand(
        max_depth + 1, dtype=torch.float64, device=device,
        generator=generator,
    )
    internal_logits = lm_head(final_hidden.index_select(
        0, metadata[3, :len(internal_nodes)],
    ))
    internal_p = probabilities(
        internal_logits, temperature, probability_dtype,
    )
    nodes, output_tokens, bonus, terminal_leaf = (
        _tree_verify_internal_fused_scan(
            parents, tokens, max_depth, internal_nodes, metadata,
            internal_p, uniforms, validate,
        )
    )
    leaf_projected = 0
    if terminal_leaf >= 0:
        leaf_p = probabilities(
            lm_head(final_hidden[terminal_leaf:terminal_leaf + 1])[0],
            temperature, probability_dtype,
        )
        if validate:
            valid = (torch.isfinite(leaf_p).all() & (leaf_p >= 0).all()
                     & (leaf_p.sum() > 0))
            if not bool(valid):
                raise FloatingPointError(
                    "Invalid lazy-projection fused leaf probabilities"
                )
        bonus = _sample_row_inverse_cdf(leaf_p, uniforms[len(nodes)])
        leaf_projected = 1
    projected_rows = len(internal_nodes) + leaf_projected
    return nodes, output_tokens, bonus, {
        "internal_projected_rows": len(internal_nodes),
        "leaf_projected_rows": leaf_projected,
        "projected_rows": projected_rows,
        "total_tree_rows": node_count,
        "lm_head_rows": projected_rows,
        "probability_rows": projected_rows,
    }
