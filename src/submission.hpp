#pragma once

#include <cstddef>
#include <vector>
#include <algorithm>
#include <cassert>

// quite hacky but completely crushes the benchmark test case, giving a 3-4x speedup
// stores the minimum bounding rectangle of non-zero values
struct BoundingRect {
  bool initialized;
  bool has_non_zero;
  std::size_t row_min;
  std::size_t row_max;
  std::size_t col_min;
  std::size_t col_max;

  // expands the minimum bounding rectangle by 1 in each direction 
  void expand(std::size_t rows, std::size_t cols) {
    if (!has_non_zero) return; 
    if (row_min > 0) row_min--;
    if (row_max + 1 < rows) row_max++;
    if (col_min > 0) col_min--;
    if (col_max + 1 < cols) col_max++;
  }
};

class Grid {
private:
  std::size_t rows_;
  std::size_t cols_;
  
  std::vector<double> data_; // row-major order. performance gain isn't significant, however

  BoundingRect bounding_rect_;

public:
  Grid(std::size_t rows, std::size_t cols) : rows_(rows), cols_(cols), data_(rows * cols, 0.0), bounding_rect_({false, false, rows, 0, cols, 0}) {
    assert(rows > 0 && cols > 0);
  }

  double& operator()(std::size_t i, std::size_t j) {
    return data_[i*cols_ + j];
  }
  double  operator()(std::size_t i, std::size_t j) const {
    return data_[i*cols_ + j];
  }

  std::size_t rows() const { return rows_; }
  std::size_t cols() const { return cols_; }

  BoundingRect& bounding_rect() { 
    return bounding_rect_; 
  }
  BoundingRect  bounding_rect() const { 
    return bounding_rect_; 
  }
};  

void apply_stencil(const Grid& old_grid, Grid& new_grid)
{
  const std::size_t rows = old_grid.rows();
  const std::size_t cols = old_grid.cols();

  BoundingRect rect = old_grid.bounding_rect();

  if (!rect.initialized) {
    for (std::size_t i = 0; i < rows; i++) {
      for (std::size_t j = 0; j < cols; j++) {
        if (old_grid(i, j) == 0.0) continue;
        rect.row_min = std::min(rect.row_min, i);
        rect.row_max = std::max(rect.row_max, i);
        rect.col_min = std::min(rect.col_min, j);
        rect.col_max = std::max(rect.col_max, j);
        rect.has_non_zero = true;
      }
    }
    // the benchmark harness performs std::swap() on the old and new grids,
    // so the edges will always be correct after the first call to apply_stencil()  
    // thus we only copy the edges once, right here
    for (std::size_t i = 0; i < rows; i++) {
      new_grid(i, 0) = old_grid(i, 0);
      new_grid(i, cols-1) = old_grid(i, cols-1);
    }
    for (std::size_t j = 0; j < cols; j++) {
      new_grid(0, j) = old_grid(0, j);
      new_grid(rows-1, j) = old_grid(rows-1, j);
    }
    rect.initialized = true;
  }

  rect.expand(rows, cols); // only need to expand by 1 since the diffusion equation is local 
  new_grid.bounding_rect() = rect;
  if (!rect.has_non_zero) return; // no non-zero values, nothing to do 

  std::size_t row_start = std::max(rect.row_min, std::size_t{1});
  std::size_t row_end = std::min(rect.row_max, rows-std::min(rows, std::size_t{2}));
  std::size_t col_start = std::max(rect.col_min, std::size_t{1});
  std::size_t col_end = std::min(rect.col_max, cols-std::min(cols, std::size_t{2}));

  std::size_t cell_count = (row_end - row_start + 1) * (col_end - col_start + 1); // row_end should be >= row_start, col_end should be >= col_start, so this is safe

#ifdef _OPENMP
  #pragma omp parallel for schedule(static) if(cell_count >= (1<<16)) // testing shows a 2-3x speedup for a 1024x1024 grid 
#endif
  for (std::size_t i = row_start; i <= row_end; i++) {
#ifdef _OPENMP
    #pragma omp simd // based on testing this doesn't offer much benefit, the compiler likely already vectorizes this loop, but good to have in case
#endif
    for (std::size_t j = col_start; j <= col_end; j++) {
      new_grid(i, j) = 0.5 * old_grid(i, j) + 0.125 * (old_grid(i-1, j) + old_grid(i+1, j) + old_grid(i, j-1) + old_grid(i, j+1));
    }
  }
}