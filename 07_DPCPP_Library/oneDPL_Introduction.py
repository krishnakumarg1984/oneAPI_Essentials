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

# # Intel® oneAPI DPC++ Library

# #### Sections
# - [What is Intel® oneAPI DPC++ Library?](#What-is-Intel®-oneAPI-DPC++-Library?)
# - [Why use oneDPL for DPC++ Heterogeneous Computing?](#Why-use-oneDPL-for-DPC++-Heterogeneous-Computing?)
# - _Code:_ [Simple oneDPL example](#Simple-oneDPL-example)
# - [oneDPL Algorithms](#oneDPL-Algorithms)
# - [DPC++ Execution Policy Usage](#DPC++-Execution-Policy-Usage)
# - _Code:_ [oneDPL with Buffer Iterators](#oneDPL-with-Buffer-Iterators)
# - _Code:_ [oneDPL with USM Pointers](#oneDPL-with-USM-Pointers)
# - _Code:_ [oneDPL with USM Allocators](#oneDPL-with-USM-Allocators)

# ## Learning Objectives
#
# - Simplify DPC++ programming by using Intel® oneAPI DPC++ Library (oneDPL)
# - Use DPC++ Library algorithms for Heterogeneous Computing
# - Implement oneDPL algorithms using Buffers and Unified Shared Memory

# ## What is Intel® oneAPI DPC++ Library?
#
# The Intel® oneAPI DPC++ Library ___(oneDPL)___ is a companion to the Intel® oneAPI DPC++ Compiler and provides an alternative for C++ developers who create heterogeneous applications and solutions. Its APIs are based on familiar standards—C++ STL, Parallel STL (PSTL), and SYCL* — to maximize productivity and performance across CPUs, GPUs, and FPGAs.
#
# oneDPL consists of the following components:
#
# * __Standard C++ APIs__
# * __Parallel STL__ algorithms
# * __Extensions APIs__ - additional set of library classes and functions

# ## Why use oneDPL for DPC++ Heterogeneous Computing?
# The Intel oneAPI DPC++ Library helps to __maximize productivity__ and __performance__ across CPUs, GPUs, and FPGAs.
#
# __Maximize performance__ by offloading computation to devices like GPU, for example the code snippet below shows how an existing functionality that executes on CPU can be offloaded to devices like GPU or FPGA using oneDPL.
#
# _Compute on CPU:_
# ```cpp  
#   std::sort(v.begin(), v.end());  
# ```
#
# _Compute on GPU with oneDPL:_
# ```cpp
#   sycl::queue q(sycl::gpu_selector{});
#   std::sort(oneapi::dpl::execution::make_device_policy(q), v.begin(), v.end());
#                                     ^                  ^  
# ```
#
# __Maximize productivity__ by making use of oneDPL algorithms instead of writing DPC++ kernel code for the algorithms that already exist in oneDPL, for example the entire DPC++ kernel code in the below DPC++ example can be accomplished with one line of code when using DPC++ Library algorithm.
#
# ```cpp
# #include<CL/sycl.hpp>
# using namespace sycl;
# constexpr int N = 4;
#
# int main() {
#   queue q;
#   std::vector<int> v(N);
#     
# //==================================================================↓
#   {
#     buffer<int> buf(v.data(),v.size());
#     q.submit([&](handler &h){
#        auto V = buf.get_access<access::mode::read_write>(h);
#        h.parallel_for(range<1>(N),[=] (id<1> i){ V[i] = 20; }); 
#     });
#   }
# //==================================================================↑
#     
#   for(int i = 0; i < v.size(); i++) std::cout << v[i] << "\n";
#   return 0;
# }
# ```
# The above code block can be accomplished with one line of code using oneDPL:
#
# ```cpp
#   std::fill(oneapi::dpl::execution::make_device_policy(q), v.begin(), v.end(), 20);
# ```
# The above code will create a temporary SYCL buffer, computes the algorith on device and copies back the buffer.

# ### Simple oneDPL example

# The example below shows how a single line of code with Parallel STL alogorithm can replace the DPC++ kernel code to get same results as previous example
# 1. Inspect the code cell below and click run ▶ to save the code to file
# 2. Next run ▶ the cell in the __Build and Run__ section below the code to compile and execute the code.

# +
# %%writefile lab/dpl_simple.cpp
//==============================================================
// Copyright © 2020 Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================

#include <oneapi/dpl/algorithm>
#include <oneapi/dpl/execution>
#include<CL/sycl.hpp>
using namespace sycl;
constexpr int N = 4;

int main() {
  queue q;
  std::cout << "Device : " << q.get_device().get_info<info::device::name>() << "\n";
  std::vector<int> v(N);
    
  //# Parallel STL fill function with device policy
  std::fill(oneapi::dpl::execution::make_device_policy(q), v.begin(), v.end(), 20);
    
  for(int i = 0; i < v.size(); i++) std::cout << v[i] << "\n";
  return 0;
}
# -

# #### Build and Run
# Select the cell below and click run ▶ to compile and execute the code above:

# ! chmod 755 q; chmod 755 run_dpl_simple.sh;if [ -x "$(command -v qsub)" ]; then ./q run_dpl_simple.sh; else ./run_dpl_simple.sh; fi

# _If the Jupyter cells are not responsive or if they error out when you compile the code samples, please restart the Jupyter Kernel: 
# "Kernel->Restart Kernel and Clear All Outputs" and compile the code samples again_

# ## oneDPL Algorithms
# 1. __C++ standard APIs__ have been tested and function well within DPC++ kernels. To use them, include the corresponding C++ standard header files and use the std namespace. List of tested C++ standard APIs available for DPC++ can be found [here](https://software.intel.com/content/www/us/en/develop/documentation/oneapi-dpcpp-library-guide/top/tested-standard-c-apis.html) for reference.
#
# 2. __Parallel STL__ which offers efficient support for both parallel and vectorized execution of algorithms for Intel® processors is extended with support for DPC++ compliant devices by introducing special DPC++ execution policies and functions. List of different Parallel STL algorithms available for DPC++ can be found [here](https://software.intel.com/content/www/us/en/develop/articles/get-started-with-parallel-stl.html) for reference.
#
# 3. __Extension APIs__ are non-standard algorithms, utility classes and iterators. List of different extension APIs available for DPC++ can be found [here](https://software.intel.com/content/www/us/en/develop/documentation/oneapi-dpcpp-library-guide/top/extension-api.html) for reference.
#
# All oneDPL header files are in the dpstd directory. Depending on the algorithm you use, include appropriate header files:
# ```cpp
# Then add a subset of the following set of lines, depending on the algorithms you intend to use:
# #include <oneapi/dpl/algorithm>
# #include <oneapi/dpl/numeric>
# #include <oneapi/dpl/memory>
# ```
# oneDPL has its own namespace `oneapi::dpl` for all its extensions, including DPC++ execution policies, non-standard algorithms, special iterators, etc.

# ## DPC++ Execution Policy Usage
#
# The DPC++ execution policy specifies where and how a Parallel STL algorithm runs. It inherits a standard C++ execution policy, encapsulates a SYCL* device or queue, and enables you to set an optional kernel name. DPC++ execution policies can be used with all standard C++ algorithms that support execution policies.
#
# 1. Add `#include <oneapi/dpl/execution>` to your code.
# 2. Create a policy object by providing a standard policy type, a optional class type for a unique kernel name as a template argument and one of the following constructor arguments:
#   * A SYCL queue
#   * A SYCL device
#   * A SYCL device selector
#   * An existing policy object with a different kernel name
# 3. The `oneapi::dpl::execution::dpcpp_default` object is a predefined object of the device_policy class, created with a default kernel name and a default queue. Use it to create customized policy objects, or to pass directly when invoking an algorithm.
#
# Below is example showing usage of execution policy to use with Parallel STL:
# ```cpp
# queue q;
# auto policy = oneapi::dpl::execution::make_device_policy(q);
# std::fill(policy, v.begin(), v.end(), 20);
# ```
#
# - Parallel STL algorithms can be called with ordinary iterators. 
# - A temporary SYCL buffer is created and the data is copied to this buffer. 
# - After processing of the temporary buffer on a device is complete, the data is copied back to the host. 

# ### Using multiple oneDPL algorithms
#
# The code example below uses two algorithms, the input vector is doubled using `std::for_each` algorithm and then it is sorted using `std::sort` algorithm. Execute the code below to find out if this is the right way or not?
#
# 1. Inspect the code cell below and click run ▶ to save the code to file
# 2. Next run ▶ the cell in the __Build and Run__ section below the code to compile and execute the code.

# +
# %%writefile lab/dpl_sortdouble.cpp
//==============================================================
// Copyright © 2020 Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================

#include <oneapi/dpl/algorithm>
#include <oneapi/dpl/execution>
#include<CL/sycl.hpp>
using namespace sycl;
using namespace oneapi::dpl::execution;

int main() {
  queue q;
  std::cout << "Device : " << q.get_device().get_info<info::device::name>() << "\n";
  std::vector<int> v{2,3,1,4};
    
  std::for_each(make_device_policy(q), v.begin(), v.end(), [](int &a){ a *= 2; });
  std::sort(make_device_policy(q), v.begin(), v.end());
    
  for(int i = 0; i < v.size(); i++) std::cout << v[i] << "\n";
  return 0;
}
# -

# #### Build and Run
# Select the cell below and click run ▶ to compile and execute the code above:

# ! chmod 755 q; chmod 755 run_dpl_sortdouble.sh;if [ -x "$(command -v qsub)" ]; then ./q run_dpl_sortdouble.sh; else ./run_dpl_sortdouble.sh; fi

# _If the Jupyter cells are not responsive or if they error out when you compile the code samples, please restart the Jupyter Kernel: 
# "Kernel->Restart Kernel and Clear All Outputs" and compile the code samples again_

# The above example works __but__ memory is copied to device and back twice when vectors are passed directly to the oneDPL algorithms:
# ```cpp
# //# memory copied host -> device
# std::for_each(make_device_policy(q), v.begin(), v.end(), [](int &a){ a *= 2; });
# //# memory copied device -> host
#
# //# memory copied host -> device
# std::sort(make_device_policy(q), v.begin(), v.end());
# //# memory copied device -> host
# ```
#
# To avoid memory being copied back and forth twice, we have to use create buffer and use __buffer iterators__ which is explained below 

# ## oneDPL with Buffer Iterators
#
# The `oneapi::dpl::begin` and `oneapi::dpl::end` are special helper functions that allow you to pass SYCL buffers to Parallel STL algorithms. These functions accept a SYCL buffer and return an object of an unspecified type. This will require the following header file:
#
#
# ```cpp
# #include <oneapi/dpl/iterator>
# ```
# Using buffer iterators will ensure that memory is not copied back and forth in between each algorithm execution on device. The code example below shows how the same example above is implemented using __buffer iterators__ which make sure the memory stays on device until the buffer is destructed.

# The code below shows simple oneDPL code. Inspect code, there are no modifications necessary. 
# 1. Inspect the code cell below and click run ▶ to save the code to file
# 2. Next run ▶ the cell in the __Build and Run__ section below the code to compile and execute the code.

# +
# %%writefile lab/dpl_buffer.cpp
//==============================================================
// Copyright © 2020 Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================

#include <oneapi/dpl/algorithm>
#include <oneapi/dpl/execution>
#include <oneapi/dpl/iterator>
#include <CL/sycl.hpp>
using namespace sycl;
using namespace oneapi::dpl::execution;


int main(){
  queue q;
  std::cout << "Device : " << q.get_device().get_info<info::device::name>() << "\n";
  std::vector<int> v{2,3,1,4};
    
  //# Create a buffer and use buffer iterators in Parallel STL algorithms
  {
    buffer buf(v);
    auto buf_begin = oneapi::dpl::begin(buf);
    auto buf_end   = oneapi::dpl::end(buf);

    std::for_each(make_device_policy(q), buf_begin, buf_end, [](int &a){ a *= 3; });
    std::sort(make_device_policy(q), buf_begin, buf_end);
  }
    
  for(int i = 0; i < v.size(); i++) std::cout << v[i] << "\n";
  return 0;
}
# -

# #### Build and Run
# Select the cell below and click run ▶ to compile and execute the code above:

# ! chmod 755 q; chmod 755 run_dpl_buffer.sh;if [ -x "$(command -v qsub)" ]; then ./q run_dpl_buffer.sh; else ./run_dpl_buffer.sh; fi

# _If the Jupyter cells are not responsive or if they error out when you compile the code samples, please restart the Jupyter Kernel: 
# "Kernel->Restart Kernel and Clear All Outputs" and compile the code samples again_

# ## oneDPL with Unified Shared Memory
# The following examples demonstrate two ways to use the oneDPL algorithms with Unified Shared Memory (USM), with either using pointers directly to iterate or use vectors to iterate:
# - USM pointers
# - USM allocators
#
# If the same buffer is processed by several algorithms, explicitly wait for completion of each algorithm before passing the buffer to the next one. Also wait for completion before accessing the data at the host.

# ### oneDPL with USM Pointers
# `malloc_shared` will allocate memory which can be accessed on both host and device, this USM pointer can be used to iterate when using oneDPL algorithm by passing pointer to the start and end of allocation.
#
# The code below shows how oneDPL can be used with __USM__ pointer. Inspect code, there are no modifications necessary. 
# 1. Inspect the code cell below and click run ▶ to save the code to file
# 2. Next run ▶ the cell in the __Build and Run__ section below the code to compile and execute the code.

# +
# %%writefile lab/dpl_usm_pointer.cpp
//==============================================================
// Copyright © 2020 Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================
#include <oneapi/dpl/algorithm>
#include <oneapi/dpl/execution>
using namespace sycl;
using namespace oneapi::dpl::execution;
const int N = 4;

int main() {
  queue q;
  std::cout << "Device : " << q.get_device().get_info<info::device::name>() << "\n";
    
  //# USM allocation on device
  int* data = malloc_shared<int>(N, q);
    
  //# Parallel STL algorithm using USM pointer
  std::fill(make_device_policy(q), data, data + N, 20);
  q.wait();
    
  for (int i = 0; i < N; i++) std::cout << data[i] << "\n";
  free(data, q);
  return 0;
}
# -

# #### Build and Run
# Select the cell below and click run ▶ to compile and execute the code above:

# ! chmod 755 q; chmod 755 run_dpl_usm_pointer.sh;if [ -x "$(command -v qsub)" ]; then ./q run_dpl_usm_pointer.sh; else ./run_dpl_usm_pointer.sh; fi

# _If the Jupyter cells are not responsive or if they error out when you compile the code samples, please restart the Jupyter Kernel: 
# "Kernel->Restart Kernel and Clear All Outputs" and compile the code samples again_

# ### oneDPL with USM Allocators
# `usm_allocator` is a C++ allocator class for USM, it takes the data type and kind of allocation as template parameter. This allocator is passed to `std::vector` constructor and the oneDPL algorithm can now use vector iterators.
#
# The code below shows oneDPL with __USM Allocators__ with vector declaration. Inspect code, there are no modifications necessary. 
# 1. Inspect the code cell below and click run ▶ to save the code to file
# 2. Next run ▶ the cell in the __Build and Run__ section below the code to compile and execute the code.

# +
# %%writefile lab/dpl_usm_alloc.cpp
//==============================================================
// Copyright © 2020 Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================
#include <oneapi/dpl/algorithm>
#include <oneapi/dpl/execution>

using namespace sycl;
using namespace oneapi::dpl::execution;

const int N = 4;

int main() {
  queue q;
  std::cout << "Device : " << q.get_device().get_info<info::device::name>() << "\n";
    
  //# USM allocator 
  usm_allocator<int, usm::alloc::shared> alloc(q);
  std::vector<int, decltype(alloc)> v(N, alloc);
    
  //# Parallel STL algorithm with USM allocator
  std::fill(make_device_policy(q), v.begin(), v.end(), 20);
  q.wait();
    
  for (int i = 0; i < v.size(); i++) std::cout << v[i] << "\n";
  return 0;
}
# -

# #### Build and Run
# Select the cell below and click run ▶ to compile and execute the code above:

# ! chmod 755 q; chmod 755 run_dpl_usm_alloc.sh;if [ -x "$(command -v qsub)" ]; then ./q run_dpl_usm_alloc.sh; else ./run_dpl_usm_alloc.sh; fi

# _If the Jupyter cells are not responsive or if they error out when you compile the code samples, please restart the Jupyter Kernel: 
# "Kernel->Restart Kernel and Clear All Outputs" and compile the code samples again_

# # Summary
# In this module you will have learned the following:
# - What is Intel® oneAPI DPC++ Library and Why use it?
# - Usage of oneDPL for Heterogeneous Computing
# - Using oneDPL algorithm with Buffers and Unified Shared Memory
#
#

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
          # !rsync -a --size-only /data/oneapi_workshop/oneAPI_Essentials/07_DPCPP_Library/ ~/oneAPI_Essentials/07_DPCPP_Library
          print('Notebook reset -- now click reload on browser.')
# linking button and function together using a button's method
button.on_click(on_button_clicked)
# displaying button and its output together
widgets.VBox([button,out])
