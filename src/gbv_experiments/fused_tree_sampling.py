"""Lazy CUDA extension for one-launch ancestral probability-tree sampling."""
from __future__ import annotations

from functools import lru_cache

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
"""


CUDA_SOURCE = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <cub/block/block_scan.cuh>
#include <algorithm>
#include <climits>

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
        name="gbv_fused_tree_sampler_v11",
        cpp_sources=[CPP_SOURCE],
        cuda_sources=[CUDA_SOURCE],
        functions=["fused_tree_sample_cuda", "fused_tree_sample_parallel_cuda",
                   "fused_tree_sample_scan_cuda"],
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
