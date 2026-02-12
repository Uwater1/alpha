#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>
#include <numeric>

extern "C" {

// Helper to compute average ranks for a vector with NaNs
void compute_ranks(const float* data, double* ranks, int n) {
    std::vector<int> indices;
    indices.reserve(n);
    for (int i = 0; i < n; ++i) {
        if (!std::isnan(data[i])) {
            indices.push_back(i);
        } else {
            ranks[i] = std::nan("");
        }
    }

    int m = indices.size();
    if (m == 0) return;

    std::sort(indices.begin(), indices.end(), [&](int i, int j) {
        return data[i] < data[j];
    });

    for (int i = 0; i < m; ) {
        int j = i + 1;
        while (j < m && data[indices[j]] == data[indices[i]]) {
            j++;
        }
        // Average rank: (i+1 + j) / 2.0
        double avg_rank = (i + j + 1) / 2.0;
        for (int k = i; k < j; k++) {
            ranks[indices[k]] = avg_rank;
        }
        i = j;
    }
}

// Compute Pearson correlation of two vectors, ignoring NaNs pairwise
double pearson_corr(const double* x, const double* y, int n) {
    double sum_x = 0, sum_y = 0, sum_xx = 0, sum_yy = 0, sum_xy = 0;
    int count = 0;
    for (int i = 0; i < n; ++i) {
        if (!std::isnan(x[i]) && !std::isnan(y[i])) {
            sum_x += x[i];
            sum_y += y[i];
            sum_xx += x[i] * x[i];
            sum_yy += y[i] * y[i];
            sum_xy += x[i] * y[i];
            count++;
        }
    }

    if (count < 2) return std::nan("");

    double mean_x = sum_x / count;
    double mean_y = sum_y / count;

    double numerator = sum_xy - count * mean_x * mean_y;
    double denominator = std::sqrt((sum_xx - count * mean_x * mean_x) * (sum_yy - count * mean_y * mean_y));

    if (denominator == 0) return std::nan("");
    return numerator / denominator;
}

// Matrix IC: f_matrix and r_matrix are Date x Asset (row-major)
// results is a Date-length array
void compute_matrix_ic(const float* f_matrix, const float* r_matrix, double* results, int n_dates, int n_assets) {
    #pragma omp parallel for
    for (int i = 0; i < n_dates; ++i) {
        std::vector<double> f_ranks(n_assets);
        std::vector<double> r_ranks(n_assets);
        
        compute_ranks(f_matrix + i * n_assets, f_ranks.data(), n_assets);
        compute_ranks(r_matrix + i * n_assets, r_ranks.data(), n_assets);
        
        results[i] = pearson_corr(f_ranks.data(), r_ranks.data(), n_assets);
    }
}

// Compute mean returns per quantile
// q_matrix: Date x Asset (int8, 1-indexed)
// r_matrix: Date x Asset
// results: quantiles x horizons matrix
void compute_quantile_means(const int8_t* q_matrix, const float* r_matrix, double* results, int n_dates, int n_assets, int n_quantiles) {
    std::vector<double> q_sums(n_quantiles, 0.0);
    std::vector<int> q_counts(n_quantiles, 0);

    for (int i = 0; i < n_dates * n_assets; ++i) {
        int8_t q = q_matrix[i];
        if (q >= 1 && q <= n_quantiles && !std::isnan(r_matrix[i])) {
            q_sums[q-1] += r_matrix[i];
            q_counts[q-1]++;
        }
    }

    for (int q = 0; q < n_quantiles; ++q) {
        if (q_counts[q] > 0) {
            results[q] = q_sums[q] / q_counts[q];
        } else {
            results[q] = std::nan("");
        }
    }
}

// Compute daily turnover per quantile
void compute_quantile_turnover(const int8_t* q_matrix, double* results, int n_dates, int n_assets, int n_quantiles, int period) {
    for (int q = 1; q <= n_quantiles; ++q) {
        double total_turnover = 0;
        int valid_days = 0;

        for (int i = 0; i < n_dates; ++i) {
            int prev_idx = i - period;
            if (prev_idx < 0) continue;
            
            int count_current = 0;
            int overlap = 0;
            for (int j = 0; j < n_assets; ++j) {
                int8_t val_current = q_matrix[i * n_assets + j];
                int8_t val_prev = q_matrix[prev_idx * n_assets + j];
                
                if (val_current == q) {
                    count_current++;
                    if (val_prev == q) overlap++;
                }
            }
            if (count_current > 0) {
                // Turnover = 1.0 - (overlap / count_current)
                total_turnover += (1.0 - (double)overlap / (double)count_current);
                valid_days++;
            }
        }
        results[q-1] = (valid_days > 0) ? (total_turnover / (double)valid_days) : std::nan("");
    }
}

// Fast cross-sectional rank for a single day
// x: input array of size n
// results: output array of size n (0-1 normalized)
void compute_rank(const float* x, float* results, int n) {
    std::vector<double> d_ranks(n);
    compute_ranks(x, d_ranks.data(), n);
    
    int valid_count = 0;
    for (int i = 0; i < n; ++i) {
        if (!std::isnan(d_ranks[i])) valid_count++;
    }

    if (valid_count <= 1) {
        for (int i = 0; i < n; ++i) {
            if (!std::isnan(d_ranks[i])) results[i] = 0.5f;
            else results[i] = std::nanf("");
        }
        return;
    }

    for (int i = 0; i < n; ++i) {
        if (!std::isnan(d_ranks[i])) {
            results[i] = (float)((d_ranks[i] - 1.0) / (valid_count - 1.0));
        } else {
            results[i] = std::nanf("");
        }
    }
}

} // extern "C"
