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

# + [markdown] tags=[]
# # Subgroups
# -

# ##### Sections
# - [What are Subgroups?](#What-are-Subgroups?)
# - [How a Subgroup Maps to Graphics Hardware](#How-a-Subgroup-Maps-to-Graphics-Hardware)
# - _Code:_ [Subgroup info](#Subgroup-info)
# - _Code:_ [Subgroup Size](#Subgroup-Size)
# - [Subgroup Functions and Algorithms](#Subgroup-Functions-and-Algorithms)
# - _Code:_ [Subgroup Shuffle](#Subgroup-Shuffle)
# - _Code:_ [Subgroup - Reduce](#Lab-Exercise:-Subgroup---Reduce)
# - _Code:_ [Subgroup - Broadcast](#Lab-Exercise:-Subgroup---Broadcast)
# - _Code:_ [Subgroup - Votes](#Lab-Exercise:-Subgroup---Votes)
#

# ## Learning Objectives

# - Understand advantages of using Subgroups in Data Parallel C++ (DPC++)
# - Take advantage of Subgroup collectives in ND-Range kernel implementation
# - Use Subgroup Shuffle operations to avoid explicit memory operations

# ## What are Subgroups?

# On many modern hardware platforms, __a subset of the work-items in a work-group__ are executed simultaneously or with additional scheduling guarantees. These subset of work-items are called subgroups. Leveraging subgroups will help to __map execution to low-level hardware__ and may help in achieving higher performance.

# ## Subgroups in ND-Range Kernel Execution

# Parallel execution with the ND_RANGE Kernel helps to group work items that map to hardware resources. This helps to __tune applications for performance__.
#
# The execution range of an ND-range kernel is divided into __work-groups__, __subgroups__ and __work-items__ as shown in picture below.

# ![ND-range kernel execution](assets/ndrange.png)

# ## How a Subgroup Maps to Graphics Hardware

# | | |
# |:---:|:---|
# | __Work-item__ | Represents the individual instances of a kernel function. | 
# | __Work-group__ | The entire iteration space is divided into smaller groups called work-groups, work-items within a work-group are scheduled on a single compute unit on hardware. | 
# | __Subgroup__ | A subset of work-items within a work-group that are executed simultaneously, may be mapped to vector hardware. (DPC++) | 
#

# The picture below shows how work-groups and subgroups map to __Intel® Gen11 Graphics Hardware__.

# ![ND-Range Hardware Mapping](assets/hwmapping.png)

# ## Why use Subgroups?

# - Work-items in a sub-group can __communicate directly using shuffle operations__, without explicit memory operations.
# - Work-items in a sub-group can synchronize using sub-group barriers and __guarantee memory consistency__ using sub-group memory fences.
# - Work-items in a sub-group have access to __sub-group functions and algorithms__, providing fast implementations of common parallel patterns.

# ## sub_group class

# The subgroup handle can be obtained from the nd_item using the __get_sub_group()__

# ```cpp
#         sycl::sub_group sg = nd_item.get_sub_group();
#
#                  OR
#
#         auto sg = nd_item.get_sub_group();
# ```

# Once you have the subgroup handle, you can query for more information about the subgroup, do shuffle operations or use collective functions.

# ## Subgroup info

# The subgroup handle can be queried to get other information like number of work-items in subgroup, or number of subgroups in a work-group which will be needed for developers to implement kernel code using subgroups:
# - __get_local_id()__ returns the index of the work-item within its subgroup
# - __get_local_range()__ returns the size of sub_group 
# - __get_group_id()__ returns the index of the subgroup
# - __get_group_range()__ returns the number of subgroups within the parent work-group
#
#
# ```cpp
#     h.parallel_for(nd_range<1>(64,64), [=](nd_item<1> item){
#       /* get sub_group handle */
#       auto sg = item.get_sub_group();
#       /* query sub_group and print sub_group info once per sub_group */
#       if(sg.get_local_id()[0] == 0){
#         out << "sub_group id: " << sg.get_group_id()[0]
#             << " of " << sg.get_group_range()[0]
#             << ", size=" << sg.get_local_range()[0] 
#             << "\n";
#       }
#     });
# ```

# ### Lab Exercise: Subgroup Info

# The DPC++ code below demonstrates subgroup query methods to print sub-group info: Inspect code, there are no modifications necessary:
# 1. Inspect the code cell below and click run ▶ to save the code to file
# 2. Next run ▶ the cell in the __Build and Run__ section below the code to compile and execute the code.

# +
# %%writefile lab/sub_group_info.cpp
//==============================================================
// Copyright © 2020 Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================
#include <CL/sycl.hpp>
using namespace sycl;

static constexpr size_t N = 64; // global size
static constexpr size_t B = 64; // work-group size

int main() {
  queue q;
  std::cout << "Device : " << q.get_device().get_info<info::device::name>() << "\n";

  q.submit([&](handler &h) {
    //# setup sycl stream class to print standard output from device code
    auto out = stream(1024, 768, h);

    //# nd-range kernel
    h.parallel_for(nd_range<1>(N, B), [=](nd_item<1> item) {
      //# get sub_group handle
      auto sg = item.get_sub_group();

      //# query sub_group and print sub_group info once per sub_group
      if (sg.get_local_id()[0] == 0) {
        out << "sub_group id: " << sg.get_group_id()[0] << " of "
            << sg.get_group_range()[0] << ", size=" << sg.get_local_range()[0]
            << "\n";
      }
    });
  }).wait();
}
# -

# #### Build and Run
# Select the cell below and click run ▶ to compile and execute the code:

# ! chmod 755 q; chmod 755 run_sub_group_info.sh; if [ -x "$(command -v qsub)" ]; then ./q run_sub_group_info.sh; else ./run_sub_group_info.sh; fi

# _If the Jupyter cells are not responsive or if they error out when you compile the code samples, please restart the Jupyter Kernel: 
# "Kernel->Restart Kernel and Clear All Outputs" and compile the code samples again_.

# ## Subgroup Size

# For tuning applications for performance, sub-group size may have to be set a specific value. For example Intel(R) GPU supports sub-groups sizes of 8, 16 and 32; by default the compiler implimentation will pick optimal sub-group size, but it can also be forced to use a specific value.
#
# The supported sub-group sizes for a GPU can be queried from device information as shown below:
#
# ```cpp
# auto sg_sizes = q.get_device().get_info<info::device::sub_group_sizes>();
#                                                       ^
# ```
#
# `reqd_sub_group_size(S)` allows setting a specific sub-group size to use for kernel execution, the specified value should be one of the supported sizes and must be a compile time constant value.
#
# ```cpp
#     q.parallel_for(nd_range<1>(N, B), [=](nd_item<1> item)[[intel::reqd_sub_group_size(16)]] {
#                                                           ^
#         // Kernel Code
#         
#     }).wait();
#
# ```

# ### Lab Exercise: Subgroup Size

# The code below shows how to query for supported sub-group sizes, and also how to set kernel to use a specific supported sub-group size.
#
# The DPC++ code below demonstrates how to use reqd_sub_group_size() to let the kernel use a specified sub-group size, change the __`S = 32`__ to __16__ or __8__ to change sub_group sizes and check the output:
#
# 1. Inspect the code cell below and click run ▶ to save the code to file.
#
# 2. Next run ▶ the cell in the __Build and Run__ section below the code to compile and execute the code.

# +
# %%writefile lab/sub_group_reqd_size.cpp
//==============================================================
// Copyright © 2020 Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================
#include <CL/sycl.hpp>
using namespace sycl;

static constexpr size_t N = 64; // global size
static constexpr size_t B = 64; // work-group size

int main() {
  queue q;
  std::cout << "Device : " << q.get_device().get_info<info::device::name>() << "\n";

  //# get all supported sub_group sizes and print
  auto sg_sizes = q.get_device().get_info<info::device::sub_group_sizes>();
  std::cout << "Supported Sub-Group Sizes : ";
  for (int i=0; i<sg_sizes.size(); i++) std::cout << sg_sizes[i] << " "; std::cout << "\n";
    
  //# find out maximum supported sub_group size
  auto max_sg_size = std::max_element(sg_sizes.begin(), sg_sizes.end());
  std::cout << "Max Sub-Group Size        : " << max_sg_size[0] << "\n";
    
  q.submit([&](handler &h) {
    //# setup sycl stream class to print standard output from device code
    auto out = stream(1024, 768, h);

    //# nd-range kernel with user specified sub_group size
    h.parallel_for(nd_range<1>(N, B), [=](nd_item<1> item)[[intel::reqd_sub_group_size(32)]] {
      //# get sub_group handle
      auto sg = item.get_sub_group();

      //# query sub_group and print sub_group info once per sub_group
      if (sg.get_local_id()[0] == 0) {
        out << "sub_group id: " << sg.get_group_id()[0] << " of "
            << sg.get_group_range()[0] << ", size=" << sg.get_local_range()[0]
            << "\n";
      }
    });
  }).wait();
}
# -

# #### Build and Run
# Select the cell below and click run ▶ to compile and execute the code:

# ! chmod 755 q; chmod 755 run_sub_group_reqd_size.sh; if [ -x "$(command -v qsub)" ]; then ./q run_sub_group_reqd_size.sh; else ./run_sub_group_reqd_size.sh; fi

# _If the Jupyter cells are not responsive or if they error out when you compile the code samples, please restart the Jupyter Kernel: 
# "Kernel->Restart Kernel and Clear All Outputs" and compile the code samples again_.

# ## Subgroup Functions and Algorithms

# The sub-group functions and algorithms expose functionality tied to work-items within a sub-group.  
#
# Providing these implementations as library functions instead __increases developer productivity__ and gives implementations the ability to __generate highly optimized 
# code__ for individual target devices.
#
# Below are some of the functions and algorithms available for sub-groups, they include useful fuctionalities to perform shuffles, reductions, scans and votes:
#
# - select_by_group
# - shift_group_left
# - shift_group_right
# - permute_group_by_xor
# - group_broadcast
# - reduce_over_group
# - exclusive_scan_over_group
# - inclusive_scan_over_group
# - any_of_group
# - all_of_group
# - none_of_group

# ## Subgroup Shuffle

# One of the most useful features of subgroups is the ability to __communicate directly between individual work-items__ without explicit memory operations.
#
# Shuffle operations enable us to remove work-group local memory usage from our kernels and/or to __avoid unnecessary repeated accesses to global memory__.
#
# Below are the different types of shuffle operations available for sub-groups:
# - `select_by_group(sg, x, id)`
# - `shift_group_left(sg, x, delta)`
# - `shift_group_right(sg, x, delta)`
# - `permute_group_by_xor(sg, x, mask)`
#
# The code below uses `permute_group_by_xor` to swap the values of two work-items:
#
# ```cpp
#     h.parallel_for(nd_range<1>(N,B), [=](nd_item<1> item){
#       auto sg = item.get_sub_group();
#       auto i = item.get_global_id(0);
#       /* Shuffles */
#       //data[i] = select_by_group(sg, data[i], 2);
#       //data[i] = shift_group_left(sg, data[i], 1);
#       //data[i] = shift_group_right(sg, data[i], 1);
#       data[i] = permute_group_by_xor(sg, data[i], 1);
#     });
#
# ```
#
# <img src="assets/shuffle_xor.png" alt="shuffle_xor" width="300"/>

# ### Lab Exercise: Subgroup Shuffle

# The code below uses subgroup shuffle to swap items in a subgroup. You can try other shuffle operations or change the fixed constant in the shuffle function to express some common commuinication patterns using `permute_group_by_xor`.
#
# The DPC++ code below demonstrates sub-group shuffle operations, the code shows how `permute_group_by_xor` can be used to swap adjacent elements in sub-group, and also you can change the code to reverse the order of element in sub-group using a different mask.
#
# 1. Inspect the code cell below and click run ▶ to save the code to file.
#
# 2. Next run ▶ the cell in the __Build and Run__ section below the code to compile and execute the code.

# +
# %%writefile lab/sub_group_shuffle.cpp
//==============================================================
// Copyright © 2020 Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================
#include <CL/sycl.hpp>
using namespace sycl;

static constexpr size_t N = 256; // global size
static constexpr size_t B = 64;  // work-group size

int main() {
  queue q;
  std::cout << "Device : " << q.get_device().get_info<info::device::name>() << "\n";

  //# initialize data array using usm
  int *data = malloc_shared<int>(N, q);
  for (int i = 0; i < N; i++) data[i] = i;
  for (int i = 0; i < N; i++) std::cout << data[i] << " ";
  std::cout << "\n\n";

  q.parallel_for(nd_range<1>(N, B), [=](nd_item<1> item) {
    auto sg = item.get_sub_group();
    auto i = item.get_global_id(0);

    //# swap adjacent items in array using sub_group permute_group_by_xor
    data[i] = permute_group_by_xor(sg, data[i], 1);
      
    //# reverse the order of items in sub_group using permute_group_by_xor
    //data[i] = permute_group_by_xor(sg, data[i], sg.get_max_local_range() - 1);
      
  }).wait();

  for (int i = 0; i < N; i++) std::cout << data[i] << " ";
  std::cout << "\n";

  free(data, q);
  return 0;
}
# -

# #### Build and Run
# Select the cell below and click run ▶ to compile and execute the code:

# ! chmod 755 q; chmod 755 run_sub_group_shuffle.sh; if [ -x "$(command -v qsub)" ]; then ./q run_sub_group_shuffle.sh; else ./run_sub_group_shuffle.sh; fi

# _If the Jupyter cells are not responsive or if they error out when you compile the code samples, please restart the Jupyter Kernel: 
# "Kernel->Restart Kernel and Clear All Outputs" and compile the code samples again_.

# ### Lab Exercise: Subgroup - Reduce

# The code below uses subgroup `reduce_over_group` function to perform reduction for all items in a subgroup. 
#
# ```cpp
#     h.parallel_for(nd_range<1>(N,B), [=](nd_item<1> item){
#       auto sg = item.get_sub_group();
#       auto i = item.get_global_id(0);
#       /* Reduction Collective on Sub-group */
#       int result = reduce_over_group(sg, data[i], plus<>());
#       //int result = reduce_over_group(sg, data[i], maximum<>());
#       //int result = reduce_over_group(sg, data[i], minimum<>());
#     });
#
# ```
#
# The DPC++ code below demonstrates sub-group collectives: Inspect code, you can change the operator "_plus_" to "_maximum_" or "_minimum_" and check output:
#
# 1. Inspect the code cell below and click run ▶ to save the code to file.
#
# 2. Next run ▶ the cell in the __Build and Run__ section below the code to compile and execute the code.

# +
# %%writefile lab/sub_group_reduce.cpp
//==============================================================
// Copyright © 2020 Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================
#include <CL/sycl.hpp>
using namespace sycl;

static constexpr size_t N = 256; // global size
static constexpr size_t B = 64;  // work-group size

int main() {
  queue q;
  std::cout << "Device : " << q.get_device().get_info<info::device::name>() << "\n";

  //# initialize data array using usm
  int *data = malloc_shared<int>(N, q);
  for (int i = 0; i < N; i++) data[i] = i;
  for (int i = 0; i < N; i++) std::cout << data[i] << " ";
  std::cout << "\n\n";

  q.parallel_for(nd_range<1>(N, B), [=](nd_item<1> item) {
    auto sg = item.get_sub_group();
    auto i = item.get_global_id(0);

    //# Add all elements in sub_group using sub_group collectives
    int result = reduce_over_group(sg, data[i], plus<>());

    //# write sub_group sum in first location for each sub_group
    if (sg.get_local_id()[0] == 0) {
      data[i] = result;
    } else {
      data[i] = 0;
    }
  }).wait();

  for (int i = 0; i < N; i++) std::cout << data[i] << " ";
  std::cout << "\n";

  free(data, q);
  return 0;
}
# -

# #### Build and Run
# Select the cell below and click run ▶ to compile and execute the code:

# ! chmod 755 q; chmod 755 run_sub_group_reduce.sh; if [ -x "$(command -v qsub)" ]; then ./q run_sub_group_reduce.sh; else ./run_sub_group_reduce.sh; fi

# _If the Jupyter cells are not responsive or if they error out when you compile the code samples, please restart the Jupyter Kernel: 
# "Kernel->Restart Kernel and Clear All Outputs" and compile the code samples again_.

# ### Lab Exercise: Subgroup - Broadcast

# The code below uses subgroup collectives `group_broadcast` function, this enables one work-item in a group to share the value of a variable with all other work-items in the group.
#
# The DPC++ code below demonstrates sub-group broadcast function: Inspect code, there are no modifications necessary:
#
# 1. Inspect the code cell below and click run ▶ to save the code to file.
#
# 2. Next run ▶ the cell in the __Build and Run__ section below the code to compile and execute the code.

# +
# %%writefile lab/sub_group_broadcast.cpp
//==============================================================
// Copyright © 2020 Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================
#include <CL/sycl.hpp>
using namespace sycl;

static constexpr size_t N = 256; // global size
static constexpr size_t B = 64; // work-group size

int main() {
  queue q;
  std::cout << "Device : " << q.get_device().get_info<info::device::name>() << "\n";

  //# initialize data array using usm
  int *data = malloc_shared<int>(N, q);
  for(int i=0; i<N; i++) data[i] = i;
  for(int i=0; i<N; i++) std::cout << data[i] << " "; 
  std::cout << "\n\n";  

  //# use parallel_for and sub_groups
  q.parallel_for(nd_range<1>(N, B), [=](nd_item<1> item) {
    auto sg = item.get_sub_group();
    auto i = item.get_global_id(0);

    //# write sub_group item values to broadcast value at index 3
    data[i] = group_broadcast(sg, data[i], 3);

  }).wait();

  for(int i=0; i<N; i++) std::cout << data[i] << " "; 
  std::cout << "\n";
  
  free(data, q);
  return 0;
}
# -

# #### Build and Run
# Select the cell below and click run ▶ to compile and execute the code:

# ! chmod 755 q; chmod 755 run_sub_group_broadcast.sh; if [ -x "$(command -v qsub)" ]; then ./q run_sub_group_broadcast.sh; else ./run_sub_group_broadcast.sh; fi

# _If the Jupyter cells are not responsive or if they error out when you compile the code samples, please restart the Jupyter Kernel: 
# "Kernel->Restart Kernel and Clear All Outputs" and compile the code samples again_.

# ### Lab Exercise: Subgroup - Votes

# The `any_of_group`, `all_of_group` and `none_of_group` functions (henceforth referred to collectively as
# “vote” functions) enable work-items to compare the result of a Boolean
# condition across their group.
#
# The DPC++ code below demonstrates sub-group collectives `any_of_group`, `all_of_group` and `none_of_group` functions: Inspect code, there are no modifications necessary:
#
# 1. Inspect the code cell below and click run ▶ to save the code to file.
#
# 2. Next run ▶ the cell in the __Build and Run__ section below the code to compile and execute the code.

# +
# %%writefile lab/sub_group_votes.cpp
//==============================================================
// Copyright © 2020 Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================
#include <CL/sycl.hpp>
using namespace sycl;

static constexpr size_t N = 32; // global size
static constexpr size_t B = 16; // work-group size

int main() {
  queue q;
  std::cout << "Device : " << q.get_device().get_info<info::device::name>() << "\n";

  //# initialize input and output array using usm
  auto input = malloc_shared<int>(N, q);
  auto all = malloc_shared<int>(N, q);
  auto any = malloc_shared<int>(N, q);
  auto none = malloc_shared<int>(N, q);
    
  //# initialize values for input array  
  for(int i=0; i<N; i++) { if (i< 10) input[i] = 0; else input[i] = i; }
  std::cout << "input:\n";
  for(int i=0; i<N; i++) std::cout << input[i] << " "; std::cout << "\n";  

  //# use parallel_for and sub_groups
  q.parallel_for(nd_range<1>(N, B), [=](nd_item<1> item)[[intel::reqd_sub_group_size(8)]] {
    auto sg = item.get_sub_group();
    auto i = item.get_global_id(0);

    //# write items with vote functions
    all[i] = all_of_group(sg, input[i]);
    any[i] = any_of_group(sg, input[i]);
    none[i] = none_of_group(sg, input[i]);

  }).wait();

  std::cout << "all_of:\n";
  for(int i=0; i<N; i++) std::cout << all[i] << " "; std::cout << "\n";
  std::cout << "any_of:\n";
  for(int i=0; i<N; i++) std::cout << any[i] << " "; std::cout << "\n";
  std::cout << "none_of:\n";
  for(int i=0; i<N; i++) std::cout << none[i] << " "; std::cout << "\n";
  
  free(input, q);
  free(all, q);
  free(any, q);
  free(none, q);
  return 0;
}
# -

# #### Build and Run
# Select the cell below and click run ▶ to compile and execute the code:

# ! chmod 755 q; chmod 755 run_sub_group_votes.sh; if [ -x "$(command -v qsub)" ]; then ./q run_sub_group_votes.sh; else ./run_sub_group_votes.sh; fi

# _If the Jupyter cells are not responsive or if they error out when you compile the code samples, please restart the Jupyter Kernel: 
# "Kernel->Restart Kernel and Clear All Outputs" and compile the code samples again_.

# ## Summary

# Subgroups allow kernel programming that maps executions at low-level hardware and may help in achieving higher levels of performance.

# <html><body><span style="color:green"><h1>Survey</h1></span></body></html>
#
# [We would appreciate any feedback you’d care to give, so that we can improve the overall training quality and experience. Thanks! ](https://intel.az1.qualtrics.com/jfe/form/SV_574qnSw6eggbn1z)

# <html><body><span style="color:Red"><h1>Reset Notebook</h1></span></body></html>
#
# ##### Should you be experiencing any issues with your notebook or just want to start fresh run the below cell.
#
#

# + jupyter={"source_hidden": true}
from IPython.display import display, Markdown, clear_output
import ipywidgets as widgets
button = widgets.Button(
    description='Reset Notebook',
    disabled=False,
    button_style='', # 'success', 'info', 'warning', 'danger' or ''
    tooltip='This will update this notebook, overwriting any changes.',
    icon='check' # (FontAwesome names without the `fa-` prefix)
)
out = widgets.Output()
def on_button_clicked(_):
      # "linking function with output"
      with out:
          # what happens when we press the button
          clear_output()
          # !rsync -a --size-only /data/oneapi_workshop/oneAPI_Essentials/04_DPCPP_Sub_Groups/ ~/oneAPI_Essentials/04_DPCPP_Sub_Groups
          print('Notebook reset -- now click reload on browser.')
# linking button and function together using a button's method
button.on_click(on_button_clicked)
# displaying button and its output together
widgets.VBox([button,out])
