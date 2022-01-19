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

# # Unified Shared Memory (USM)

# ##### Sections
# - [What is USM?](#What-is-Unified-Shared-Memory?)
# - [Types of USM](#Types-of-USM)
# - _Code:_ [Implicit USM](#USM-Implicit-Data-Movement)
# - _Code:_ [Explicit USM](#USM-Explicit-Data-Movement)
# - [Data Dependency in USM](#Data-dependency-in-USM)
# - _Code:_ [Data Dependency in-order queues](#Lab-Exercise:-USM-and-Data-dependency-1)
# - _Code:_ [Data Dependency out-of-order queues](#Lab-Exercise:-USM-and-Data-dependency-2)

# ## Learning Objectives

# - Use new Data Parallel C++ (DPC++) features such as Unified Shared Memory to simplify programming.
# - Understand implicit and explicit way of moving memory using USM.
# - Solve data dependency between kernel tasks in optimal way.

# ## What is Unified Shared Memory?

# Unified Shared Memory (USM) is a DPC++ tool for data management. USM is a
# __pointer-based approach__ that should be familiar to C and C++ programmers who use malloc
# or new to allocate data. USM __simplifies development__ for the programmer when __porting existing
# C/C++ code__ to DPC++.

# ## Developer view of USM

# The picture below shows __developer view of memory__ without USM and with USM. 
#
# With USM, the developer can reference that same memory object in host and device code.  

# ![Developer View of USM](assets/usm_dev_view.png)

# ## Types of USM

# Unified shared memory provides both __explicit__ and __implicit__ models for managing memory.

# | Type | function call | Description | Accessible on Host | Accessible on Device |
# |:---|:---|:---|:---:|:---:|
# | Device | malloc_device | Allocation on device (explicit) | NO | YES |
# | Host | malloc_host |Allocation on host (implicit) | YES | YES |
# | Shared | malloc_shared | Allocation can migrate between host and device (implicit) | YES | YES |
#

# ## USM Syntax

# __USM Initialization__:
# The initialization below shows example of shared allocation using `malloc_shared`, the "q" queue parameter provides information about the device that memory is accessable.
# ```cpp
# int *data = malloc_shared<int>(N, q);
#                   ^               ^
# ```
#
# OR you can use familiar C++/C style malloc:
# ```cpp
# int *data = static_cast<int *>(malloc_shared(N * sizeof(int), q));
#                                      ^                        ^
# ```
#
# __Freeing USM:__
# ```cpp
# free(data, q);
#            ^
# ```

# ### USM Implicit Data Movement

# The DPC++ code below shows an implementation of USM using <code>malloc_shared</code>, in which data movement happens implicitly between host and device. Useful to __get functional quickly with minimum amount of code__ and developers will not having worry about moving memory between host and device.
#
# The DPC++ code below demonstrates USM Implicit Data Movement: Inspect code, there are no modifications necessary:
#
# 1. Inspect the code cell below and click run ▶ to save the code to file.
#
# 2. Next run ▶ the cell in the __Build and Run__ section below the code to compile and execute the code.

# +
# %%writefile lab/usm.cpp
//==============================================================
// Copyright © 2020 Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================
#include <CL/sycl.hpp>
using namespace sycl;

static const int N = 16;

int main() {
  queue q;
  std::cout << "Device : " << q.get_device().get_info<info::device::name>() << "\n";

  //# USM allocation using malloc_shared
  int *data = malloc_shared<int>(N, q);

  //# Initialize data array
  for (int i = 0; i < N; i++) data[i] = i;

  //# Modify data array on device
  q.parallel_for(range<1>(N), [=](id<1> i) { data[i] *= 2; }).wait();

  //# print output
  for (int i = 0; i < N; i++) std::cout << data[i] << "\n";
  free(data, q);
  return 0;
}
# -

# #### Build and Run
# Select the cell below and click run ▶ to compile and execute the code:

# ! chmod 755 q; chmod 755 run_usm.sh; if [ -x "$(command -v qsub)" ]; then ./q run_usm.sh; else ./run_usm.sh; fi

# _If the Jupyter cells are not responsive, or if they error out when you compile the code samples, please restart the Jupyter Kernel: 
# "Kernel->Restart Kernel and Clear All Outputs" and compile the code samples again_.

# ### USM Explicit Data Movement

# The DPC++ code below shows an implementation of USM using <code>malloc_device</code>, in which data movement between host and device should be done explicitly by developer using <code>memcpy</code>. This allows developers to have more __controlled movement of data__ between host and device.
#
# The DPC++ code below demonstrates USM Explicit Data Movement: Inspect code, there are no modifications necessary:
#
# 1. Inspect the code cell below and click run ▶ to save the code to file.
#
# 2. Next run ▶ the cell in the __Build and Run__ section below the code to compile and execute the code.

# +
# %%writefile lab/usm_explicit.cpp
//==============================================================
// Copyright © 2020 Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================
#include <CL/sycl.hpp>
using namespace sycl;

static const int N = 16;

int main() {
  queue q;
  std::cout << "Device : " << q.get_device().get_info<info::device::name>() << "\n";

  //# initialize data on host
  int *data = static_cast<int *>(malloc(N * sizeof(int)));
  for (int i = 0; i < N; i++) data[i] = i;

  //# Explicit USM allocation using malloc_device
  int *data_device = malloc_device<int>(N, q);

  //# copy mem from host to device
  q.memcpy(data_device, data, sizeof(int) * N).wait();

  //# update device memory
  q.parallel_for(range<1>(N), [=](id<1> i) { data_device[i] *= 2; }).wait();

  //# copy mem from device to host
  q.memcpy(data, data_device, sizeof(int) * N).wait();

  //# print output
  for (int i = 0; i < N; i++) std::cout << data[i] << "\n";
  free(data_device, q);
  free(data);
  return 0;
}
# -

# #### Build and Run
# Select the cell below and click run ▶ to compile and execute the code:

# ! chmod 755 q; chmod 755 run_usm_explicit.sh; if [ -x "$(command -v qsub)" ]; then ./q run_usm_explicit.sh; else ./run_usm_explicit.sh; fi

# _If the Jupyter cells are not responsive, or if they error out when you compile the code samples, please restart the Jupyter Kernel: 
# "Kernel->Restart Kernel and Clear All Outputs" and compile the code samples again_.

# ## When to use USM?

# __SYCL*__ Buffers are __powerful and elegant__. Use them  if the abstraction applies cleanly in your application, and/or if buffers aren’t disruptive to your development. However, replacing all pointers and arrays with buffers in a C++ program can be a burden to programmers so in this case consider using USM.
#
# __USM__ provides a familiar pointer-based C++ interface:
# * Useful when __porting C++ code__ to DPC++ by minimizing changes
# * Use shared allocations when porting code to __get functional quickly__. Note that shared allocation is not intended to provide peak performance out of box.
# * Use explicit USM allocations when __controlled data movement__ is needed.

# ## Data dependency in USM

# When using unified shared memory, dependences between tasks must be specified using events since tasks execute asynchronously and mulitple tasks can execute simultaneously. 
#
# Programmers may either explicitly <code>wait</code> on event objects or use the <code>depends_on</code> method inside a command group to specify a list of events that must complete before a task may begin.
#
# In the example below, the two kernel tasks are updating the same `data` array, these two kernels can execute simultanously and may cause undesired result. The first task must be complete before the second can begin, the next section will show different ways the data dependency can be resolved.
# ```cpp
#     q.parallel_for(range<1>(N), [=](id<1> i) { data[i] += 2; });
#
#     q.parallel_for(range<1>(N), [=](id<1> i) { data[i] += 3; });
#
# ```
#
# ### Different options to manage **data dependency** when using USM:
# - __wait()__ on kernel task
# - use __in_order__ queue property
# - use __depends_on__ method
#
# #### wait()
# - Use __q.wait()__ on kernel task to wait before the next dependent task can begin, however it will block execution on host.
#
# ```cpp
#     q.parallel_for(range<1>(N), [=](id<1> i) { data[i] += 2; });
#     q.wait();  // <--- wait() will make sure that task is complete before continuing
#
#     q.parallel_for(range<1>(N), [=](id<1> i) { data[i] += 3; });
#
# ```
#
# #### in_order queue property
# - Use __in_order__ queue property for the queue, this will serialize all the kerenel tasks. Note that execution will not overlap even if the queues have no data dependency.
#
# ```cpp
#     queue q{property::queue::in_order()}; // <--- this will serialize all kernel tasks
#
#     q.parallel_for(range<1>(N), [=](id<1> i) { data[i] += 2; }); 
#
#     q.parallel_for(range<1>(N), [=](id<1> i) { data[i] += 3; });
# ```
#
# #### depends_on
# - Use __h.depends_on(e)__ method in command group to specify events that must complete before a task may begin.
#
# ```cpp
#     auto e = q.submit([&](handler &h) {  // <--- e is event for kernel task
#       h.parallel_for(range<1>(N), [=](id<1> i) { data[i] += 2; });
#     });
#
#     q.submit([&](handler &h) {
#       h.depends_on(e);  // <--- waits until event e is complete
#       h.parallel_for(range<1>(N), [=](id<1> i) { data[i] += 3; });
#     });
# ```
# - You can also use a simplified way of specifying dependencies by passing an extra parameter in `parallel_for`
#
# ```cpp
#     auto e = q.parallel_for(range<1>(N), [=](id<1> i) { data[i] += 2; }); 
#
#     q.parallel_for(range<1>(N), e, [=](id<1> i) { data[i] += 3; });
#                                 ^
# ```
#

# ## Lab Exercise: USM and Data dependency 1

# The code below uses USM and has three kernels that are submitted to the device. Each kernel modifies the same data array. There is no data dependency between the three queue submissions, so the code can be fixed to get desired output of 20.
#
# There are three solutions, use **in_order** queue property or use **wait()** event or use **depends_on()** method.
#
# **HINT:**
# - Add **wait()** for each queue submit
# - Implement **depends_on()** method in second and third kernel task
# - Use **in_order** queue property instead of regular queue: `queue q{property::queue::in_order()};`
#
#
# 1. Edit the code cell below and click run ▶ to save the code to a file.
# 2. Next run ▶ the cell in the __Build and Run__ section below the code to compile and execute the code.

# +
# %%writefile lab/usm_data.cpp
//==============================================================
// Copyright © 2020 Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================
#include <CL/sycl.hpp>
using namespace sycl;

static const int N = 256;

int main() {
  queue q;
  std::cout << "Device : " << q.get_device().get_info<info::device::name>()
            << "\n";

  //# USM allocation and initialization
  int *data = malloc_shared<int>(N, q);
  for (int i = 0; i < N; i++) data[i] = 10;

  //# There are three kerenel tasks submitted without data dependency, analyze the code and fix data dependency

  q.parallel_for(range<1>(N), [=](id<1> i) { data[i] += 2; });

  q.parallel_for(range<1>(N), [=](id<1> i) { data[i] += 3; });

  q.parallel_for(range<1>(N), [=](id<1> i) { data[i] += 5; }).wait();

  for (int i = 0; i < N; i++) std::cout << data[i] << " ";
  std::cout << "\n";
  free(data, q);
  return 0;
}
# -

# #### Build and Run
# Select the cell below and click run ▶ to compile and execute the code:

# ! chmod 755 run_usm_data.sh; if [ -x "$(command -v qsub)" ]; then ./q run_usm_data.sh; else ./run_usm_data.sh; fi

# _If the Jupyter cells are not responsive or if they error out when you compile the code samples, please restart the Jupyter Kernel: 
# "Kernel->Restart Kernel and Clear All Outputs" and compile the code samples again_.

# #### Solution
# - [usm_data.cpp](src/usm_data.cpp)

# ## Lab Exercise: USM and Data dependency 2

# The code below uses USM and has three kernels that are submitted to device. The first two kernels modify two different memeory objects and the third one has a dependency on the first two. There is no data dependency between the three queue submissions, so the code can be fixed to get desired output of 25.
#
# - Implementing **depends_on()** method gets the best performance
# - Using **in_order** queue property or **wait()** will get results but not the most efficient
#
# **HINT:**
# ```cpp
# auto e1 = ... 
# auto e2 = ...
#
# q.parallel_for(range<1>(N), {e1, e2}, [=](id<1> i) {
# ```
#
#
# 1. Edit the code cell below and click run ▶ to save the code to a file.
# 2. Next run ▶ the cell in the __Build and Run__ section below the code to compile and execute the code.

# +
# %%writefile lab/usm_data2.cpp
//==============================================================
// Copyright © 2020 Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================
#include <CL/sycl.hpp>
using namespace sycl;

static const int N = 1024;

int main() {
  queue q;
  std::cout << "Device : " << q.get_device().get_info<info::device::name>()
            << "\n";

  //# Two USM allocation and initialization
  int *data1 = malloc_shared<int>(N, q);
  int *data2 = malloc_shared<int>(N, q);
  for (int i = 0; i < N; i++) { 
    data1[i] = 10; 
    data2[i] = 10; 
  }

  //# There are 3 kerenel tasks submitted without data dependency, analyze the code and fix data dependency

  q.parallel_for(range<1>(N), [=](id<1> i) { data1[i] += 2; });

  q.parallel_for(range<1>(N), [=](id<1> i) { data2[i] += 3; });

  q.parallel_for(range<1>(N), [=](id<1> i) { data1[i] += data2[i]; }).wait();

  for (int i = 0; i < N; i++) std::cout << data1[i] << " ";
  std::cout << "\n";
  free(data1, q);
  free(data2, q);
  return 0;
}
# -

# #### Build and Run
# Select the cell below and click run ▶ to compile and execute the code:

# ! chmod 755 run_usm_data2.sh; if [ -x "$(command -v qsub)" ]; then ./q run_usm_data2.sh; else ./run_usm_data2.sh; fi

# _If the Jupyter cells are not responsive or if they error out when you compile the code samples, please restart the Jupyter Kernel: 
# "Kernel->Restart Kernel and Clear All Outputs" and compile the code samples again_.

# #### Solution
# - [usm_data2.cpp](src/usm_data2.cpp)

# ## Summary

# USM makes it easy to port C/C++ code to DPC++. USM allows a simple implicit data movement approach to get functional quicky as well as allows controlled data movement with explicit approach.

# <html><body><span style="color:green"><h1>Survey</h1></span></body></html>
#
# [We would appreciate any feedback you’d care to give, so that we can improve the overall training quality and experience. Thanks! ](https://intel.az1.qualtrics.com/jfe/form/SV_71IHlodSGkLU5vv)

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
          # !rsync -a --size-only /data/oneapi_workshop/oneAPI_Essentials/03_DPCPP_Unified_Shared_Memory/ ~/oneAPI_Essentials/03_DPCPP_Unified_Shared_Memory
          print('Notebook reset -- now click reload on browser.')
# linking button and function together using a button's method
button.on_click(on_button_clicked)
# displaying button and its output together
widgets.VBox([button,out])
