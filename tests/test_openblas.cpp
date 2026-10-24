#include <cblas.h>

#include <array>
#include <cstddef>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
constexpr std::size_t guard_size = 16;
constexpr double guard_value = -987654321.0;

struct Case {
  blasint m;
  blasint n;
  blasint k;
  CBLAS_TRANSPOSE transa;
  CBLAS_TRANSPOSE transb;
  double alpha;
  double beta;
};

struct Matrix {
  blasint rows;
  blasint columns;
  blasint leading;
  std::vector<double> storage;

  Matrix(blasint row_count, blasint column_count, blasint padding)
      : rows(row_count), columns(column_count), leading(row_count + padding),
        storage(guard_size * 2 + static_cast<std::size_t>(leading) * columns, guard_value) {}

  double* data() { return storage.data() + guard_size; }
  double& at(blasint row, blasint column) {
    return storage[guard_size + static_cast<std::size_t>(column) * leading + row];
  }
};

double a_value(blasint row, blasint inner) {
  return static_cast<double>((row % 7 + 1) * (inner % 3 + 1));
}

double b_value(blasint inner, blasint column) {
  return static_cast<double>((column % 11 + 1) * (inner % 5 + 1));
}

const char* transpose_name(CBLAS_TRANSPOSE value) {
  return value == CblasNoTrans ? "N" : "T";
}

void describe(const Case& test) {
  std::cerr << "M=" << test.m << "，N=" << test.n << "，K=" << test.k
            << "，转置=" << transpose_name(test.transa) << transpose_name(test.transb)
            << "，alpha=" << test.alpha << "，beta=" << test.beta;
}

std::size_t verify_case(const Case& test) {
  const bool transpose_a = test.transa == CblasTrans;
  const bool transpose_b = test.transb == CblasTrans;
  Matrix a(transpose_a ? test.k : test.m, transpose_a ? test.m : test.k, 3);
  Matrix b(transpose_b ? test.n : test.k, transpose_b ? test.k : test.n, 5);
  Matrix c(test.m, test.n, 7);
  for (blasint inner = 0; inner < test.k; ++inner) {
    for (blasint row = 0; row < test.m; ++row) {
      (transpose_a ? a.at(inner, row) : a.at(row, inner)) = a_value(row, inner);
    }
    for (blasint column = 0; column < test.n; ++column) {
      (transpose_b ? b.at(column, inner) : b.at(inner, column)) = b_value(inner, column);
    }
  }
  for (blasint column = 0; column < test.n; ++column) {
    for (blasint row = 0; row < test.m; ++row) {
      c.at(row, column) = static_cast<double>((row + column) % 5 - 2);
    }
  }
  const auto original_a = a.storage;
  const auto original_b = b.storage;
  Matrix expected = c;
  // 小整数乘加全程可精确表示；独立三重循环无需容差，也不复用 BLAS。
  for (blasint column = 0; column < test.n; ++column) {
    for (blasint row = 0; row < test.m; ++row) {
      double product = 0.0;
      for (blasint inner = 0; inner < test.k; ++inner) {
        product += a_value(row, inner) * b_value(inner, column);
      }
      expected.at(row, column) = test.alpha * product + test.beta * c.at(row, column);
    }
  }
  cblas_dgemm(CblasColMajor, test.transa, test.transb, test.m, test.n, test.k,
              test.alpha, a.data(), a.leading, b.data(), b.leading,
              test.beta, c.data(), c.leading);

  std::size_t checks = 0;
  const auto unchanged = [&](const Matrix& matrix, const std::vector<double>& original, const char* name) {
    for (std::size_t slot = 0; slot < original.size(); ++slot) {
      ++checks;
      if (matrix.storage[slot] != original[slot]) {
        describe(test);
        std::cerr << "；输入矩阵 " << name << " 的存储位置 " << slot
                  << " 被修改，实际=" << matrix.storage[slot] << "，预期=" << original[slot] << '\n';
        throw std::runtime_error("BLAS 修改了只读输入或其填充区");
      }
    }
  };
  unchanged(a, original_a, "A");
  unchanged(b, original_b, "B");
  for (std::size_t slot = 0; slot < c.storage.size(); ++slot) {
    ++checks;
    if (c.storage[slot] != expected.storage[slot]) {
      describe(test);
      std::cerr << "；C 存储位置=" << slot;
      const auto matrix_size = static_cast<std::size_t>(c.leading) * c.columns;
      if (slot >= guard_size && slot - guard_size < matrix_size) {
        const auto relative = slot - guard_size;
        std::cerr << "，行=" << relative % c.leading << "，列=" << relative / c.leading;
      }
      std::cerr << "，实际=" << c.storage[slot] << "，预期=" << expected.storage[slot] << '\n';
      throw std::runtime_error("BLAS 乘积错误或输出填充区被修改");
    }
  }
  return checks;
}

struct TrmmCase {
  CBLAS_SIDE side;
  CBLAS_UPLO triangle;
  CBLAS_TRANSPOSE transpose;
  CBLAS_DIAG diagonal;
  blasint m;
  blasint n;
  double alpha;
};

std::size_t verify_trmm(const TrmmCase& test) {
  const bool left = test.side == CblasLeft;
  const blasint dimension = left ? test.m : test.n;
  Matrix a(dimension, dimension, 3);
  Matrix b(test.m, test.n, 5);
  const auto in_triangle = [&](blasint row, blasint column) {
    return test.triangle == CblasUpper ? row <= column : row >= column;
  };
  for (blasint column = 0; column < dimension; ++column) {
    for (blasint row = 0; row < dimension; ++row) {
      if (in_triangle(row, column)) {
        // 单位对角故意存为 7，未使用的半三角仍保留哨兵，检验是否被错误读取。
        a.at(row, column) = row == column && test.diagonal == CblasUnit
                               ? 7.0 : static_cast<double>((row + column) % 3 + 1);
      }
    }
  }
  const auto b_value_at = [](blasint row, blasint column) {
    return static_cast<double>((row % 5 + 1) * (column % 3 + 1));
  };
  for (blasint column = 0; column < test.n; ++column) {
    for (blasint row = 0; row < test.m; ++row) b.at(row, column) = b_value_at(row, column);
  }
  const auto original_a = a.storage;
  Matrix expected = b;
  const auto op_a = [&](blasint row, blasint column) {
    const blasint source_row = test.transpose == CblasTrans ? column : row;
    const blasint source_column = test.transpose == CblasTrans ? row : column;
    if (!in_triangle(source_row, source_column)) return 0.0;
    if (source_row == source_column && test.diagonal == CblasUnit) return 1.0;
    return static_cast<double>((source_row + source_column) % 3 + 1);
  };
  for (blasint column = 0; column < test.n; ++column) {
    for (blasint row = 0; row < test.m; ++row) {
      double product = 0.0;
      for (blasint inner = 0; inner < dimension; ++inner) {
        product += left ? op_a(row, inner) * b_value_at(inner, column)
                        : b_value_at(row, inner) * op_a(inner, column);
      }
      expected.at(row, column) = test.alpha * product;
    }
  }
  cblas_dtrmm(CblasColMajor, test.side, test.triangle, test.transpose, test.diagonal,
              test.m, test.n, test.alpha, a.data(), a.leading, b.data(), b.leading);
  const auto describe_trmm = [&] {
    std::cerr << "三角乘法，M=" << test.m << "，N=" << test.n
              << "，侧=" << (left ? "左" : "右")
              << "，三角=" << (test.triangle == CblasUpper ? "上" : "下")
              << "，转置=" << transpose_name(test.transpose)
              << "，对角=" << (test.diagonal == CblasUnit ? "单位" : "非单位");
  };
  std::size_t checks = 0;
  for (std::size_t slot = 0; slot < a.storage.size(); ++slot) {
    ++checks;
    if (a.storage[slot] != original_a[slot]) {
      describe_trmm();
      std::cerr << "；A 存储位置=" << slot << "，实际=" << a.storage[slot]
                << "，预期=" << original_a[slot] << '\n';
      throw std::runtime_error("BLAS 三角乘法修改了只读输入或其填充区");
    }
  }
  for (std::size_t slot = 0; slot < b.storage.size(); ++slot) {
    ++checks;
    if (b.storage[slot] != expected.storage[slot]) {
      describe_trmm();
      std::cerr << "；B 存储位置=" << slot;
      const auto matrix_size = static_cast<std::size_t>(b.leading) * b.columns;
      if (slot >= guard_size && slot - guard_size < matrix_size) {
        const auto relative = slot - guard_size;
        std::cerr << "，行=" << relative % b.leading << "，列=" << relative / b.leading;
      }
      std::cerr << "，实际=" << b.storage[slot] << "，预期=" << expected.storage[slot] << '\n';
      throw std::runtime_error("BLAS 三角乘积错误或输出填充区被修改");
    }
  }
  return checks;
}
}  // 匿名命名空间

int main() {
  std::cout << std::setprecision(17);
  std::cerr << std::setprecision(17);
  std::cout << "BLAS 配置：" << openblas_get_config()
            << "；CPU 内核：" << openblas_get_corename() << std::endl;
  try {
    // 小尺寸覆盖余数与四种转置；较大的长条矩阵同时覆盖通用分块路径。
    // 不强制选择 CPU 内核，其他架构也必须满足相同的公共 CBLAS 合同。
    const std::array cases{
        Case{15, 5, 341, CblasNoTrans, CblasNoTrans, 1.0, 0.0},
        Case{16, 6, 342, CblasNoTrans, CblasNoTrans, 2.0, 1.0},
        Case{17, 7, 384, CblasNoTrans, CblasTrans, -2.0, 1.0},
        Case{31, 12, 385, CblasTrans, CblasNoTrans, 1.0, -1.0},
        Case{32, 6, 768, CblasTrans, CblasTrans, 2.0, 0.0},
        Case{33, 7, 341, CblasTrans, CblasNoTrans, -2.0, -1.0},
        Case{513, 6, 341, CblasNoTrans, CblasNoTrans, 1.0, 0.0},
        Case{513, 6, 342, CblasNoTrans, CblasNoTrans, 1.0, 0.0},
        Case{513, 6, 384, CblasNoTrans, CblasNoTrans, 1.0, 1.0},
        Case{513, 6, 385, CblasNoTrans, CblasTrans, 2.0, -1.0},
        Case{513, 6, 768, CblasTrans, CblasNoTrans, -2.0, 0.0},
        Case{257, 12, 384, CblasTrans, CblasTrans, 1.0, 1.0},
        Case{529, 5, 384, CblasNoTrans, CblasNoTrans, 2.0, 1.0},
        Case{449, 7, 384, CblasNoTrans, CblasNoTrans, 1.0, -1.0},
        Case{257, 12, 768, CblasNoTrans, CblasNoTrans, -2.0, 1.0},
    };
    std::size_t checks = 0;
    for (const auto& test : cases) checks += verify_case(test);
    const std::array trmm_cases{
        TrmmCase{CblasLeft, CblasUpper, CblasNoTrans, CblasNonUnit, 384, 6, 1.0},
        TrmmCase{CblasLeft, CblasLower, CblasTrans, CblasUnit, 384, 16, -2.0},
        TrmmCase{CblasRight, CblasLower, CblasNoTrans, CblasNonUnit, 6, 384, 1.0},
        TrmmCase{CblasRight, CblasUpper, CblasTrans, CblasUnit, 16, 384, 2.0},
    };
    for (const auto& test : trmm_cases) checks += verify_trmm(test);
    std::cout << "公共 BLAS 数值与存储边界检查通过：" << cases.size() + trmm_cases.size()
              << " 组，" << checks << " 项比较\n";
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
  return 0;
}
