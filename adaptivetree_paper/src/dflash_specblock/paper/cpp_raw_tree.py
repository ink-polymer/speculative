"""Optional compiled max-prefix enumerator for the guarded AdaptiveTree path."""
from __future__ import annotations

from functools import lru_cache


@lru_cache(maxsize=1)
def load_cpp_raw_tree_module():
    """Compile once per process; callers retain a semantics-identical fallback."""
    try:
        from torch.utils.cpp_extension import load_inline
    except Exception:
        return None

    source = r"""
#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <queue>
#include <vector>

struct Candidate {
    double negative_score;
    std::array<int16_t, 16> ranks{};
    int64_t rank_length;
    int64_t parent;
    int64_t depth;
    int64_t rank;
    double score;
};

struct CandidateGreater {
    bool operator()(const Candidate& left, const Candidate& right) const {
        if (left.negative_score != right.negative_score) {
            return left.negative_score > right.negative_score;
        }
        const int64_t shared = std::min(left.rank_length, right.rank_length);
        for (int64_t index = 0; index < shared; ++index) {
            if (left.ranks[index] != right.ranks[index]) {
                return left.ranks[index] > right.ranks[index];
            }
        }
        if (left.rank_length != right.rank_length) {
            return left.rank_length > right.rank_length;
        }
        if (left.parent != right.parent) return left.parent > right.parent;
        if (left.depth != right.depth) return left.depth > right.depth;
        if (left.rank != right.rank) return left.rank > right.rank;
        return left.score > right.score;
    }
};

std::vector<torch::Tensor> build_raw_prefix_tree(
    torch::Tensor top_log_probs,
    torch::Tensor top_token_ids,
    int64_t budget,
    torch::Tensor nodes,
    torch::Tensor depths,
    torch::Tensor scores,
    torch::Tensor parents,
    torch::Tensor visibility) {
    TORCH_CHECK(!top_log_probs.is_cuda() && !top_token_ids.is_cuda(),
                "top-k tensors must be on CPU");
    TORCH_CHECK(top_log_probs.dim() == 2 && top_token_ids.dim() == 2,
                "top-k tensors must have rank two");
    TORCH_CHECK(top_log_probs.scalar_type() == torch::kFloat,
                "log probabilities must be float32");
    TORCH_CHECK(top_token_ids.scalar_type() == torch::kLong,
                "token ids must be int64");
    TORCH_CHECK(top_log_probs.sizes() == top_token_ids.sizes(),
                "top-k tensor shapes must match");
    TORCH_CHECK(budget > 0 && top_log_probs.size(1) > 0,
                "budget and top-k width must be positive");
    TORCH_CHECK(!nodes.is_cuda() && !depths.is_cuda() && !scores.is_cuda()
                && !parents.is_cuda() && !visibility.is_cuda(),
                "output tensors must be on CPU");
    TORCH_CHECK(nodes.scalar_type() == torch::kLong
                && depths.scalar_type() == torch::kLong
                && scores.scalar_type() == torch::kDouble
                && parents.scalar_type() == torch::kLong
                && visibility.scalar_type() == torch::kBool,
                "output tensor dtypes are invalid");
    TORCH_CHECK(nodes.numel() >= budget && depths.numel() >= budget
                && scores.numel() >= budget && parents.numel() >= budget + 1
                && visibility.dim() == 2
                && visibility.size(0) >= budget + 1
                && visibility.size(1) >= budget + 1,
                "output tensors are smaller than the requested budget");

    auto probabilities = top_log_probs.contiguous();
    auto token_ids = top_token_ids.contiguous();
    auto probability = probabilities.accessor<float, 2>();
    auto token = token_ids.accessor<int64_t, 2>();
    const int64_t depth_limit = probabilities.size(0);
    const int64_t topk = probabilities.size(1);
    TORCH_CHECK(depth_limit <= 16, "compiled raw tree supports at most 16 depths");

    visibility.zero_();
    auto node = nodes.accessor<int64_t, 1>();
    auto depth_output = depths.accessor<int64_t, 1>();
    auto score_output = scores.accessor<double, 1>();
    auto parent_output = parents.accessor<int64_t, 1>();
    auto* visible = visibility.data_ptr<bool>();
    const int64_t visibility_stride = visibility.size(1);
    parent_output[0] = -1;
    visible[0] = true;

    const double first_score = static_cast<double>(probability[0][0]);
    std::priority_queue<Candidate, std::vector<Candidate>, CandidateGreater> heap;
    Candidate first;
    first.negative_score = -first_score;
    first.rank_length = 1;
    first.parent = 0;
    first.depth = 1;
    first.rank = 0;
    first.score = first_score;
    heap.push(first);
    int64_t count = 0;
    while (!heap.empty() && count < budget) {
        Candidate current = heap.top();
        heap.pop();
        const int64_t index = count + 1;
        node[count] = token[current.depth - 1][current.rank];
        depth_output[count] = current.depth;
        score_output[count] = current.score;
        parent_output[index] = current.parent;
        std::memcpy(visible + index * visibility_stride,
                    visible + current.parent * visibility_stride,
                    index * sizeof(bool));
        visible[index * visibility_stride + index] = true;
        ++count;

        if (current.rank + 1 < topk) {
            const double sibling_score = (
                current.score
                - static_cast<double>(probability[current.depth - 1][current.rank])
                + static_cast<double>(probability[current.depth - 1][current.rank + 1]));
            Candidate sibling = current;
            sibling.negative_score = -sibling_score;
            sibling.ranks[sibling.rank_length - 1] += 1;
            sibling.rank += 1;
            sibling.score = sibling_score;
            heap.push(sibling);
        }
        if (current.depth < depth_limit) {
            const double child_score = (
                current.score + static_cast<double>(probability[current.depth][0]));
            Candidate child = current;
            child.negative_score = -child_score;
            child.ranks[child.rank_length] = 0;
            child.rank_length += 1;
            child.parent = index;
            child.depth += 1;
            child.rank = 0;
            child.score = child_score;
            heap.push(child);
        }
    }
    return {nodes, depths, scores, parents, visibility};
}

std::vector<int64_t> follow_raw_tree(
    torch::Tensor node_token_ids,
    torch::Tensor parents,
    torch::Tensor posterior_tokens,
    int64_t node_count) {
    TORCH_CHECK(!node_token_ids.is_cuda() && !parents.is_cuda()
                && !posterior_tokens.is_cuda(),
                "tree and posterior metadata must be on CPU");
    TORCH_CHECK(node_token_ids.scalar_type() == torch::kLong
                && parents.scalar_type() == torch::kLong
                && posterior_tokens.scalar_type() == torch::kLong,
                "tree and posterior metadata must be int64");
    TORCH_CHECK(node_count >= 0 && node_count <= node_token_ids.numel()
                && node_count + 1 <= parents.numel()
                && node_count + 1 <= posterior_tokens.numel(),
                "node count is outside the metadata bounds");
    auto nodes = node_token_ids.contiguous();
    auto parent_tensor = parents.contiguous();
    auto posterior = posterior_tokens.contiguous();
    const auto* token = nodes.data_ptr<int64_t>();
    const auto* parent = parent_tensor.data_ptr<int64_t>();
    const auto* target = posterior.data_ptr<int64_t>();
    std::vector<int64_t> accepted{0};
    int64_t current = 0;
    int64_t next_token = target[0];
    while (true) {
        int64_t child = -1;
        for (int64_t index = 1; index <= node_count; ++index) {
            if (parent[index] == current && token[index - 1] == next_token) {
                child = index;
                break;
            }
        }
        if (child < 0) break;
        current = child;
        accepted.push_back(current);
        next_token = target[current];
    }
    accepted.insert(accepted.begin(), next_token);
    return accepted;
}
"""
    try:
        return load_inline(
            name="adaptive_raw_prefix_ext_v8",
            cpp_sources=[source],
            functions=["build_raw_prefix_tree", "follow_raw_tree"],
            extra_cflags=["-O3"],
            verbose=False,
        )
    except Exception:
        return None
