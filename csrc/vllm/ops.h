#pragma once

#include <optional>
#include <torch/library.h>

#include "core/scalar_type.hpp"

#include <vector>

void rms_norm(torch::Tensor& out, torch::Tensor& input, torch::Tensor& weight,
              double epsilon);
void fused_add_rms_norm(torch::Tensor& input, torch::Tensor& residual,
                        torch::Tensor& weight, double epsilon);