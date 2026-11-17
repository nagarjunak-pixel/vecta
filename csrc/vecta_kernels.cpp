// Native C++20 (-O3) kernels for Vecta: exact brute-force L2 top-k, gathered L2 distances for graph
// search, product-quantization asymmetric distance (ADC) scans and Hamming-distance top-k over
// packed binary codes. All distances accumulate in double over float32 inputs, and every top-k
// orders by (distance, id) so results are deterministic.

#include <bit>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <queue>
#include <utility>
#include <vector>

namespace {

using Entry = std::pair<double, int64_t>;  // (distance, id); max-heap keeps the worst on top.

inline double l2(const float* __restrict__ a, const float* __restrict__ b, int32_t dim) {
    double acc = 0.0;
    for (int32_t d = 0; d < dim; ++d) {
        const double diff = static_cast<double>(a[d]) - static_cast<double>(b[d]);
        acc += diff * diff;
    }
    return acc;
}

inline void push_topk(std::priority_queue<Entry>& heap, int32_t k, double dist, int64_t id) {
    if (static_cast<int32_t>(heap.size()) < k) {
        heap.emplace(dist, id);
    } else if (Entry(dist, id) < heap.top()) {
        heap.pop();
        heap.emplace(dist, id);
    }
}

inline void drain(std::priority_queue<Entry>& heap, int32_t k, int64_t* ids, double* dists) {
    // Pop worst-first into the tail so the output is ascending; pad unused slots with -1 / +inf.
    const int32_t found = static_cast<int32_t>(heap.size());
    for (int32_t j = found; j < k; ++j) {
        ids[j] = -1;
        dists[j] = std::numeric_limits<double>::infinity();
    }
    for (int32_t j = found - 1; j >= 0; --j) {
        ids[j] = heap.top().second;
        dists[j] = heap.top().first;
        heap.pop();
    }
}

}  // namespace

extern "C" {

// Exact k-nearest neighbours under squared L2 for nq queries against nb base vectors.
void vecta_l2_topk(
    const float* __restrict__ queries,
    int64_t nq,
    const float* __restrict__ base,
    int64_t nb,
    int32_t dim,
    int32_t k,
    const uint8_t* __restrict__ alive,  // may be null: all alive
    int64_t* __restrict__ out_ids,
    double* __restrict__ out_dists
) {
    for (int64_t q = 0; q < nq; ++q) {
        std::priority_queue<Entry> heap;
        const float* qv = queries + q * dim;
        for (int64_t i = 0; i < nb; ++i) {
            if (alive != nullptr && alive[i] == 0) {
                continue;
            }
            push_topk(heap, k, l2(qv, base + i * dim, dim), i);
        }
        drain(heap, k, out_ids + q * k, out_dists + q * k);
    }
}

// Squared L2 from one query to a gathered list of base rows (HNSW neighbour expansion).
void vecta_l2_gather(
    const float* __restrict__ query,
    const float* __restrict__ base,
    int32_t dim,
    const int64_t* __restrict__ ids,
    int64_t n,
    double* __restrict__ out
) {
    for (int64_t i = 0; i < n; ++i) {
        out[i] = l2(query, base + ids[i] * dim, dim);
    }
}

// PQ asymmetric distance scan: dist(i) = sum_m lut[m, codes[i, m]]; returns top-k of the n rows.
// row_ids maps local rows to global ids (so IVF lists can be scanned in place).
void vecta_pq_adc_topk(
    const uint8_t* __restrict__ codes,
    int64_t n,
    int32_t m,
    const float* __restrict__ lut,  // [m, 256]
    const int64_t* __restrict__ row_ids,
    int32_t k,
    int64_t* __restrict__ out_ids,
    double* __restrict__ out_dists
) {
    std::priority_queue<Entry> heap;
    for (int64_t i = 0; i < n; ++i) {
        const uint8_t* c = codes + i * m;
        double acc = 0.0;
        for (int32_t j = 0; j < m; ++j) {
            acc += static_cast<double>(lut[j * 256 + c[j]]);
        }
        push_topk(heap, k, acc, row_ids[i]);
    }
    drain(heap, k, out_ids, out_dists);
}

// Hamming top-k over packed binary codes (words uint64 per vector) using hardware popcount.
void vecta_hamming_topk(
    const uint64_t* __restrict__ codes,
    int64_t n,
    int32_t words,
    const uint64_t* __restrict__ query,
    const uint8_t* __restrict__ alive,
    int32_t k,
    int64_t* __restrict__ out_ids,
    double* __restrict__ out_dists
) {
    std::priority_queue<Entry> heap;
    for (int64_t i = 0; i < n; ++i) {
        if (alive != nullptr && alive[i] == 0) {
            continue;
        }
        const uint64_t* c = codes + i * words;
        int64_t dist = 0;
        for (int32_t w = 0; w < words; ++w) {
            dist += std::popcount(c[w] ^ query[w]);
        }
        push_topk(heap, k, static_cast<double>(dist), i);
    }
    drain(heap, k, out_ids, out_dists);
}

}  // extern "C"
