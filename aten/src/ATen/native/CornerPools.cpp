#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <ATen/NativeFunctions.h>
#include <ATen/core/op_registration/op_registration.h>

namespace at {
namespace native {

at::Tensor top_forward(at::Tensor input) {
  // Initialize output
  at::Tensor output = at::zeros_like(input);

  // Get height
  int64_t height = input.size(2);

  output.copy_(input);

  for (int64_t ind = 1; ind < height; ind <<= 1) {
    at::Tensor max_temp = at::slice(output, 2, 0, height-ind);
    at::Tensor cur_temp = at::slice(output, 2, 0, height-ind);
    at::Tensor next_temp = at::slice(output, 2, ind, height);
    at::max_out(max_temp, cur_temp, next_temp);
  }

  return output;
}

at::Tensor bottom_forward(at::Tensor input) {
  // Initialize output
  at::Tensor output = at::zeros_like(input);

  // Get height
  int64_t height = input.size(2);

  output.copy_(input);

  for (int64_t ind = 1; ind < height; ind <<= 1) {
    at::Tensor max_temp = at::slice(output, 2, ind, height);
    at::Tensor cur_temp = at::slice(output, 2, ind, height);
    at::Tensor next_temp = at::slice(output, 2, 0, height-ind);
    at::max_out(max_temp, cur_temp, next_temp);
  }

  return output;
}

at::Tensor left_forward(at::Tensor input) {
  // Initialize output
  at::Tensor output = at::zeros_like(input);

  // Get width
  int64_t width = input.size(3);

  output.copy_(input);

  for (int64_t ind = 1; ind < width; ind <<= 1) {
    at::Tensor max_temp = at::slice(output, 3, 0, width-ind);
    at::Tensor cur_temp = at::slice(output, 3, 0, width-ind);
    at::Tensor next_temp = at::slice(output, 3, ind, width);
    at::max_out(max_temp, cur_temp, next_temp);
  }

  return output;
}

at::Tensor right_forward(at::Tensor input) {
  // Initialize output
  at::Tensor output = at::zeros_like(input);

  // Get width
  int64_t width = input.size(3);

  output.copy_(input);

  for (int64_t ind = 1; ind < width; ind <<= 1) {
    at::Tensor max_temp = at::slice(output, 3, ind, width);
    at::Tensor cur_temp = at::slice(output, 3, ind, width);
    at::Tensor next_temp = at::slice(output, 3, 0, width-ind);
    at::max_out(max_temp, cur_temp, next_temp);
  }

  return output;
}

static auto registry = torch::RegisterOperators()
  .op(torch::RegisterOperators::options()
    .schema("corner_pools::top(Tensor a) -> Tensor")
    .catchAllKernel<decltype(top_forward), &top_forward>())
  .op(torch::RegisterOperators::options()
    .schema("corner_pools::bottom(Tensor a) -> Tensor")
    .catchAllKernel<decltype(bottom_forward), &bottom_forward>())
  .op(torch::RegisterOperators::options()
    .schema("corner_pools::left(Tensor a) -> Tensor")
    .catchAllKernel<decltype(left_forward), &left_forward>())
  .op(torch::RegisterOperators::options()
    .schema("corner_pools::right(Tensor a) -> Tensor")
    .catchAllKernel<decltype(right_forward), &right_forward>());

// static auto registry = torch::RegisterOperators()
//   .op(torch::RegisterOperators::options()
//     .schema("aten::detach(Tensor self) -> Tensor")
//     .catchAllKernel<decltype(detach), &detach>());

} // namespace native
} // namespace at

