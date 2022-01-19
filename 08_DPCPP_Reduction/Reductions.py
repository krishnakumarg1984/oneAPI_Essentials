# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.13.6
#   kernelspec:
#     display_name: "Python 3.7 (Intel\xAE oneAPI)"
#     language: python
#     name: c009-intel_distribution_of_python_3_oneapi-beta05-python
# ---

# # Reductions in DPC++

# ##### Sections
# - [What are Reductions?](#What-are-Reductions?)
# - _Code:_ [Reduction with single_task](#Reduction-with-single_task)
# - _Code:_ [Reduction with parallel_for](#Reduction-with-parallel_for)
# - [Group Reduction](#Group-Reduction)
# - _Code:_ [Reduction using work_group reduce](#Reduction-using-work_group-reduce)
# - [Reduction simplification in parallel_for](#Reduction-simplification-in-parallel_for)
# - _Code:_ [Reduction in parallel_for USM](#Reduction-in-parallel_for-USM)
# - _Code:_ [Reduction in parallel_for Buffers](#Reduction-in-parallel_for-Buffers)
# - _Code:_ [Multiple Reductions in one kernel](#Multiple-Reductions-in-one-kernel)
# - _Code:_ [Reduction with Custom Operator](#Reduction-with-Custom-Operator)

# ## Learning Objectives

# - Understand how reductions can be performed with parallel kernels
# - Take advantages __reduce function__ to do reduction at sub_group and work_group level
# - Use DPC++ __reduction extension__ to simplify reduction with parallel kernels
# - Use __multiple__ reductions in a single kernel.

# ## What are Reductions?

# A __reduction produces a single value by combining multiple values__ in an unspecified order, using an operator that is both associative and commutative (e.g. addition). Only the final value resulting from a reduction is of interest to the programmer.
#
# A very common example is calculating __sum__ by adding a bunch of values.
#
# Parallelizing reductions can be tricky because of the nature of computation and accelerator hardware. Lets look at code examples showing how reduction can be performed on GPU using kernel invocation using __single_task__ and __parallel_for__:

# ### Reduction with single_task

# The simplest way to write a kernel function to compute sum for GPU is using a kernel invocation using __single_task__ and using a simple __for-loop__ to compute the sum of all values in the array. This way of reduction works but there is no parallelism in computation.
#
# ```cpp
#   q.single_task([=](){
#     int sum = 0;
#     for(int i=0;i<N;i++){
#         sum += data[i];
#     }
#     data[0] = sum;
#   });
# ```
#

# The DPC++ code below demonstrates computing sum of array of values using `single_task` for kernel invocation.
# 1. Inspect the code cell below and click run ▶ to save the code to file
# 2. Next run ▶ the cell in the __Build and Run__ section below the code to compile and execute the code.

# +
# %%writefile lab/sum_single_task.cpp
//==============================================================
// Copyright © 2020 Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================
#include <CL/sycl.hpp>
using namespace sycl;

static constexpr size_t N = 1024; // global size

int main() {
  //# setup sycl::queue with default device selector
  queue q;
  std::cout << "Device : " << q.get_device().get_info<info::device::name>() << "\n";

  //# initialize data array using usm
  auto data = malloc_shared<int>(N, q);
  for (int i = 0; i < N; i++) data[i] = i;

  //# user single_task to add all numbers
  q.single_task([=](){
    int sum = 0;
    for(int i=0;i<N;i++){
        sum += data[i];
    }
    data[0] = sum;
  }).wait();

  std::cout << "Sum = " << data[0] << "\n";
  
  free(data, q);
  return 0;
}

# -

# #### Build and Run
# Select the cell below and click run ▶ to compile and execute the code:

# ! chmod 755 q; chmod 755 run_sum_single_task.sh; if [ -x "$(command -v qsub)" ]; then ./q run_sum_single_task.sh; else ./run_sum_single_task.sh; fi

# _If the Jupyter cells are not responsive or if they error out when you compile the code samples, please restart the Jupyter Kernel: 
# "Kernel->Restart Kernel and Clear All Outputs" and compile the code samples again_.

# ## Reduction with parallel_for

# ND-Range kernel allows grouping executions that map to __compute units__ on hardware which allows for parallel execution of work-groups. As shows in the picture below, the entire range is divided into `work_group` which execute on a compute unit on the GPU hardware. Depending on number of compute units in the hardware, multiple work_groups can be executed to get parallelism. This allows to compute sum of each `work_group` and then it is further reduced to add all the work_group sums using a `single_task` kernel invocation. This gives better performance than the previous example which only uses `single_task` to do reduction.
#
# <img src="assets/hwmapping.png" alt="hwmapping.png" width="600"/>
#
# The code below uses `nd_range parallel_for` kernel to compute sum of values for every work-group and eventually another `single_task` kernel is used to compute sum of all work_group sums to get final result:
#
# ```cpp
#   q.parallel_for(nd_range<1>(N, B), [=](nd_item<1> item){
#     size_t index = item.get_global_id(0);
#     if(item.get_local_id(0) == 0 ){
#       int sum_wg = 0;
#       for(int i=index; i<index+B; i++){
#         sum_wg += data[i];
#       }
#       data[index] = sum_wg;
#     }
#   });
#
#   q.single_task([=](){
#     int sum = 0;
#     for(int i=0;i<N;i+=B){
#         sum += data[i];
#     }
#     data[0] = sum;
#   });
# ```
#
#

# The DPC++ code below demonstrates using nd-range kernel to calculate sum at each work_group and then adds all work_group sums using a `single_task` kernel invocation:
#
# 1. Inspect the code cell below and click run ▶ to save the code to file.
#
# 2. Next run ▶ the cell in the __Build and Run__ section below the code to compile and execute the code.

# +
# %%writefile lab/sum_work_group.cpp
//==============================================================
// Copyright © 2020 Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================
#include <CL/sycl.hpp>
using namespace sycl;

static constexpr size_t N = 1024; // global size
static constexpr size_t B = 128; // work-group size

int main() {
  //# setup queue with in_order property
  queue q(property::queue::in_order{});
  std::cout << "Device : " << q.get_device().get_info<info::device::name>() << "\n";

  //# initialize data array using usm
  auto data = malloc_shared<int>(N, q);
  for (int i = 0; i < N; i++) data[i] = i;

  //# use parallel_for to calculate sum for each work_group
  q.parallel_for(nd_range<1>(N, B), [=](nd_item<1> item){
    size_t index = item.get_global_id(0);
    if(item.get_local_id(0) == 0 ){
      int sum_wg = 0;
      for(int i=index; i<index+B; i++){
        sum_wg += data[i];
      }
      data[index] = sum_wg;
    }
  });

  q.single_task([=](){
    int sum = 0;
    for(int i=0;i<N;i+=B){
        sum += data[i];
    }
    data[0] = sum;
  }).wait();

  std::cout << "Sum = " << data[0] << "\n";
  
  free(data, q);
  return 0;
}

# -

# #### Build and Run
# Select the cell below and click run ▶ to compile and execute the code:

# ! chmod 755 q; chmod 755 run_sum_work_group.sh; if [ -x "$(command -v qsub)" ]; then ./q run_sum_work_group.sh; else ./run_sum_work_group.sh; fi

# _If the Jupyter cells are not responsive or if they error out when you compile the code samples, please restart the Jupyter Kernel: 
# "Kernel->Restart Kernel and Clear All Outputs" and compile the code samples again_.

# ## Group Reduction

# The `sycl::reduce_over_group()` function is a new extention in DPC++ which can also be used to perform certain common reduction operations in the kernel function for each `sub_group` or `work_group`. The reduce function can be used to **simplify reduction computation** with one line of code as shown below, instead of manually coding reduction with for-loop:
#
#
# ```cpp
#       sum = sycl::reduce_over_group(group, data[i], sycl::plus<>());
# ```
#
# The `sycl::reduce_over_group()` function takes three parameters: work-group/sub-group, work-item and operation to be performed on the group. There are various common parallel operations available like `sycl::plus<>()`, `sycl::maximum<>()` or `sycl::minimum<>()`
#
# Using this reduce function on a `sub_group` will optimize computation by leveraging sub_group shuffle operation to load values from register instead of making repeated access to global memory. The reduce function can also be used on a `work_group` which is also optimized implicitly by making use of sub_group functionality. 
#
# The next section show how reduce function can be used on `work_group` to do reduction computation:
#
#

# ### Reduction using work_group reduce

# The code below uses work_group reduce function to add all items in a work_group and then the final computation is accomplished using single_task kernel invocation to add all work_group sums.
#
# The DPC++ code below demonstrates work-group reduce: Inspect code, there are no modifications necessary:
#
# 1. Inspect the code cell below and click run ▶ to save the code to file.
#
# 2. Next run ▶ the cell in the __Build and Run__ section below the code to compile and execute the code.

# +
# %%writefile lab/sum_workgroup_reduce.cpp
//==============================================================
// Copyright © 2020 Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================
#include <CL/sycl.hpp>

using namespace sycl;

static constexpr size_t N = 1024; // global size
static constexpr size_t B = 128; // work-group size

int main() {
  //# setup queue with in_order property
  queue q(property::queue::in_order{});
  std::cout << "Device : " << q.get_device().get_info<info::device::name>() << "\n";

  //# initialize data array using usm
  auto data = malloc_shared<int>(N, q);
  for (int i = 0; i < N; i++) data[i] = i;

  //# use parallel_for to calculate sum for work_group using reduce
  q.parallel_for(nd_range<1>(N, B), [=](nd_item<1> item){
    auto wg = item.get_group();
    auto i = item.get_global_id(0);

    //# Adds all elements in work_group using work_group reduce
    int sum_wg = reduce_over_group(wg, data[i], plus<>());

    //# write work_group sum to first location for each work_group
    if (item.get_local_id(0) == 0) data[i] = sum_wg;

  });

  q.single_task([=](){
    int sum = 0;
    for(int i=0;i<N;i+=B){
        sum += data[i];
    }
    data[0] = sum;
  }).wait();

  std::cout << "Sum = " << data[0] << "\n";

  free(data, q);
  return 0;
}

# -

# #### Build and Run
# Select the cell below and click run ▶ to compile and execute the code:

# ! chmod 755 q; chmod 755 run_sum_workgroup_reduce.sh; if [ -x "$(command -v qsub)" ]; then ./q run_sum_workgroup_reduce.sh; else ./run_sum_workgroup_reduce.sh; fi

# _If the Jupyter cells are not responsive or if they error out when you compile the code samples, please restart the Jupyter Kernel: 
# "Kernel->Restart Kernel and Clear All Outputs" and compile the code samples again_.

# ## Reduction simplification in parallel_for

# In the previous examples of reduction, the computation requires a two step approach to first perform reduction at group level and then perform reduction of output from each group. This section introduces to a new extension that will greatly simplify reduction computation.
#
# __DPC++ introduces reduction to the ND-range version of parallel_for__, using syntax that is roughly aligned with OpenMP and C++ for_loop.
#
# It is common for parallel kernels to produce a single output resulting from some combination of all inputs (e.g. the sum). Writing efficient reductions is a complex task, depending on both device and runtime characteristics. Providing an abstraction for reductions in SYCL would greatly improve programmer productivity.
#
# `sycl::reduction` object in parallel_for encapsulates the reduction variable, an optional operator identity and the reduction operator as shown below:
#
# ```cpp
#      q.parallel_for(nd_range<1>{N, B}, sycl::reduction(sum, sycl::plus<>()), [=](nd_item<1> it, auto& temp) {
#        int i = it.get_global_id(0);
#        temp.combine(data[i]);
#      });
# ```

# ### Reduction in parallel_for USM

# The code below uses __sycl::reduction__ object in _parallel_for_ to compute the reduction with just one kernel using Unified Shared Memory(USM) for memory management.
#
# The DPC++ code below demonstrates reduction in parallel_for with USM: Inspect code, there are no modifications necessary:
#
# 1. Inspect the code cell below and click run ▶ to save the code to file.
#
# 2. Next run ▶ the cell in the __Build and Run__ section below the code to compile and execute the code.

# +
# %%writefile lab/sum_reduction_usm.cpp
//==============================================================
// Copyright © 2020 Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================
#include <CL/sycl.hpp>

using namespace sycl;

static constexpr size_t N = 1024; // global size
static constexpr size_t B = 128; // work-group size

int main() {
  //# setup queue with default selector
  queue q;
  std::cout << "Device : " << q.get_device().get_info<info::device::name>() << "\n";

  //# initialize data array using usm
  auto data = malloc_shared<int>(N, q);
  for (int i = 0; i < N; i++) data[i] = i;

  //# implicit USM for writing sum value
  int* sum = malloc_shared<int>(1, q);
  *sum = 0;

  //# nd-range kernel parallel_for with reduction parameter
  q.parallel_for(nd_range<1>{N, B}, reduction(sum, plus<>()), [=](nd_item<1> it, auto& temp) {
    auto i = it.get_global_id(0);
    temp.combine(data[i]);
  }).wait();

  std::cout << "Sum = " << *sum << "\n";

  free(data, q);
  free(sum, q);
  return 0;
}

# -

# #### Build and Run
# Select the cell below and click run ▶ to compile and execute the code:

# ! chmod 755 q; chmod 755 run_sum_reduction_usm.sh; if [ -x "$(command -v qsub)" ]; then ./q run_sum_reduction_usm.sh; else ./run_sum_reduction_usm.sh; fi

# _If the Jupyter cells are not responsive or if they error out when you compile the code samples, please restart the Jupyter Kernel: 
# "Kernel->Restart Kernel and Clear All Outputs" and compile the code samples again_.

# ### Reduction in parallel_for Buffers

# The code below uses __sycl::reduction__ object in _parallel_for_ to compute the reduction with just one kernel using SYCL buffers and accessors for memory management.
#
# The DPC++ code below demonstrates reduction in parallel_for with Buffers: Inspect code, there are no modifications necessary:
#
# 1. Inspect the code cell below and click run ▶ to save the code to file.
#
# 2. Next run ▶ the cell in the __Build and Run__ section below the code to compile and execute the code.

# +
# %%writefile lab/sum_reduction_buffers.cpp
//==============================================================
// Copyright © 2020 Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================
#include <CL/sycl.hpp>

using namespace sycl;

static constexpr size_t N = 1024; // global size
static constexpr size_t B = 128; // work-group size

int main() {
  queue q;
  std::cout << "Device : " << q.get_device().get_info<info::device::name>() << "\n";

  std::vector<int> data(N);
  for (int i = 0; i < N; i++) data[i] = i;
  int sum = 0;
  {
    //# create buffers for data and sum
    buffer buf_data(data);
    buffer buf_sum(&sum, range(1));

    q.submit([&](handler& h) {
      //# create accessors for buffer
      accessor acc_data(buf_data, h, read_only);

      //# nd-range kernel parallel_for with reduction parameter
      h.parallel_for(nd_range<1>{N, B}, reduction(buf_sum, h, plus<>()), [=](nd_item<1> it, auto& temp) {
        auto i = it.get_global_id(0);
        temp.combine(acc_data[i]);
      });
    });
  }
  std::cout << "Sum = " << sum << "\n";

  return 0;
}

# -

# #### Build and Run
# Select the cell below and click run ▶ to compile and execute the code:

# ! chmod 755 q; chmod 755 run_sum_reduction_buffers.sh; if [ -x "$(command -v qsub)" ]; then ./q run_sum_reduction_buffers.sh; else ./run_sum_reduction_buffers.sh; fi

# _If the Jupyter cells are not responsive or if they error out when you compile the code samples, please restart the Jupyter Kernel: 
# "Kernel->Restart Kernel and Clear All Outputs" and compile the code samples again_.

# ### Multiple Reductions in one kernel

# The code below uses multiple __sycl::reduction__ objects in _parallel_for_ to compute the reductions with just one kernel using SYCL buffers and accessors for memory management.
#
# Multiple reductions are also supported with just one kernel, the code snippet below shows how to definne a kernel using parallel_for with multiple reduction objects:
#
# ```cpp
# h.parallel_for(nd_range<1>{N, B}, reduction1, reduction2, ..., [=](nd_item<1> it, auto& temp1, auto& temp2, ...) {
#   // kernel code
# });
# ```
#
# The DPC++ code below demonstrates multiple reduction in parallel_for with Buffers: Inspect code, there are no modifications necessary:
#
# 1. Inspect the code cell below and click run ▶ to save the code to file.
#
# 2. Next run ▶ the cell in the __Build and Run__ section below the code to compile and execute the code.

# +
# %%writefile lab/multiple_reductions_buffers.cpp
//==============================================================
// Copyright © 2020 Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================
#include <CL/sycl.hpp>

using namespace sycl;

static constexpr size_t N = 1024; // global size
static constexpr size_t B = 128; // work-group size

int main() {
  queue q;
  std::cout << "Device : " << q.get_device().get_info<info::device::name>() << "\n";

  //# initialize inputs and outputs
  std::vector<int> data(N);
  for (int i = 0; i < N; i++) data[i] = i;
  int sum = 0, min = 0, max = 0;
  {
    //# create buffers
    buffer buf_data(data);
    buffer buf_sum(&sum, range(1));
    buffer buf_min(&min, range(1));
    buffer buf_max(&max, range(1));

    q.submit([&](handler& h) {
      //# create accessors for data and results
      accessor acc_data(buf_data, h, read_only);
        
      //# define reduction objects for sum, min, max reduction
      auto reduction_sum = reduction(buf_sum, h, plus<>());
      auto reduction_min = reduction(buf_min, h, minimum<>());
      auto reduction_max = reduction(buf_max, h, maximum<>());
      
      //# parallel_for with multiple reduction objects
      h.parallel_for(nd_range<1>{N, B}, reduction_sum, reduction_min, reduction_max, [=](nd_item<1> it, auto& temp_sum, auto& temp_min, auto& temp_max) {
        auto i = it.get_global_id();
        temp_sum.combine(acc_data[i]);
        temp_min.combine(acc_data[i]);
        temp_max.combine(acc_data[i]);
      });
    });
  }
  //# compute mid-range
  auto mid_range = (min+max)/2.f;
 
  //# print results
  std::cout << "Sum       = " << sum << "\n";
  std::cout << "Min       = " << min << "\n"; 
  std::cout << "Max       = " << max << "\n";
  std::cout << "Mid-Range = " << mid_range << "\n";

  return 0;
}

# -

# #### Build and Run
# Select the cell below and click run ▶ to compile and execute the code:

# ! chmod 755 q; chmod 755 run_multiple_reductions_buffers.sh; if [ -x "$(command -v qsub)" ]; then ./q run_multiple_reductions_buffers.sh; else ./run_multiple_reductions_buffers.sh; fi

# _If the Jupyter cells are not responsive or if they error out when you compile the code samples, please restart the Jupyter Kernel: 
# "Kernel->Restart Kernel and Clear All Outputs" and compile the code samples again_.

# ### Reduction with Custom Operator

# The code below uses __sycl::reduction__ object in _parallel_for_ to compute the reduction object that uses a custom operator to find minumum value and index.
#
# The DPC++ code below demonstrates reduction in parallel_for with custom user defined operator to perform reduction: Inspect code, there are no modifications necessary:
#
# 1. Inspect the code cell below and click run ▶ to save the code to file.
#
# 2. Next run ▶ the cell in the __Build and Run__ section below the code to compile and execute the code.

# +
# %%writefile lab/reduction_custom_operator.cpp
//==============================================================
// Copyright © 2020 Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================
#include <CL/sycl.hpp>
#include <time.h>

using namespace sycl;

static constexpr size_t N = 256; // global size
static constexpr size_t B = 64; // work-group size

template <typename T, typename I>
struct pair {
  bool operator<(const pair& o) const {
    return val <= o.val || (val == o.val && idx <= o.idx);
  }
  T val;
  I idx;
};

int main() {
  //# setup queue with default selector
  queue q;
 
  //# initialize input data and result using usm
  auto result = malloc_shared<pair<int, int>>(1, q);
  auto data = malloc_shared<int>(N, q);

  //# initialize input data with random numbers
  srand(time(0));
  for (int i = 0; i < N; ++i) data[i] = rand() % 256;
  std::cout << "Input Data:\n";
  for (int i = 0; i < N; i++) std::cout << data[i] << " "; std::cout << "\n\n";

  //# custom operator for reduction to find minumum and index
  pair<int, int> operator_identity = {std::numeric_limits<int>::max(), std::numeric_limits<int>::min()};
  *result = operator_identity;
  auto reduction_object = reduction(result, operator_identity, minimum<pair<int, int>>());

  //# parallel_for with user defined reduction object
  q.parallel_for(nd_range<1>{N, B}, reduction_object, [=](nd_item<1> item, auto& temp) {
       int i = item.get_global_id(0);
       temp.combine({data[i], i});
  }).wait();

  std::cout << "Minimum value and index = " << result->val << " at " << result->idx << "\n";

  free(result, q);
  free(data, q);
  return 0;
}
# -

# #### Build and Run
# Select the cell below and click run ▶ to compile and execute the code:

# ! chmod 755 q; chmod 755 run_reduction_custom_operator.sh; if [ -x "$(command -v qsub)" ]; then ./q run_reduction_custom_operator.sh; else ./run_reduction_custom_operator.sh; fi

# _If the Jupyter cells are not responsive or if they error out when you compile the code samples, please restart the Jupyter Kernel: 
# "Kernel->Restart Kernel and Clear All Outputs" and compile the code samples again_.

# ## Summary

# `sycl::reduce_over_group` function for sub_group/work_group and `sycl::reduction` in parallel_for helps to optimize and simplify reduction computation in DPC++
