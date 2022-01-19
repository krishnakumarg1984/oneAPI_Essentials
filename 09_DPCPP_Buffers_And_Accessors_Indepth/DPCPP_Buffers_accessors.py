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

# # Explore Buffers and Accessors in depth

# ##### Sections
# - [Buffers and Accessors](#Buffers-and-Accessors)
# - [Buffer creation](#Buffer-creation)
#     - _Code:_ [Buffer creation examples](#Buffer-creation-examples)
# - [Buffer Properties](#Buffer-Properties)
#     - _Code:_ [use_host_ptr](#use_host_ptr)
#     - _Code:_ [set_final_data](#set_final_data)
#     - _Code:_ [set_write_back](#set_write_back)
# - [Accessors](#Accessors)
#     - [Access modes](#Access-modes)    
#     - _Code:_ [noinit](#noinit)   
#     - _Code:_ [Host Accessors](#Host-Accessors)
#     - _Code:_ [Initialize buffer data using Host accessors](#Initialize-buffer-data-using-Host-accessors)    
# - [Additional topics on Buffers](#Additional-topics-on-Buffers)
#     - [Sub Buffers](#Sub-Buffers)
#     - [Uncommon ways to create Buffers](#Uncommon-ways-to-create-Buffers)
#     
#

# ## Learning Objectives
# * Explain Buffers and Accessors in depth.
# * Understand the Sub buffers and how to create and use Sub buffers
# * Explain buffers properties and when to use_host_ptr, set_final_data and set_write_data 
# * Explain Accessors and the modes of accessor creation
# * Explain host accessors and the different use cases of host accessors
#

# ## Buffers and Accessors
# __Buffers__ are high level abstarction for data and these are accessbile either on the host machine or on the devices. Buffers encapsulate data in a SYCL application across both devices and host. __Accessors__ is the mechanism to access buffer data. Buffers are 1, 2 or 3 dimensional data.  
#
# One of the most important aspects of data parallel computations is how they access data to accelerate a computation.
# Accelerator devices often have their own attached memories that cannot be directly accessed from the host. Device and host can either share physical memory or have distinct memories. When the memories are distinct, offloading computation requires copying data between host and device. DPC++ does not require you to manage the data copies.
#
# By __creating Buffers and Accessors, DPC++ ensures that the data is available to host and device without any effort on your part__. DPC++ also allows you explicit control over data movement to achieve best peformance. Buffers are accessible on the host and may be accessible on multiple devices.
#
# USM forces programmers to think about where memory lives and what should be accessible where. The buffer abstraction is a higher-level model that hides this from the programmer. Buffers simply represent data, and it becomes the job of the runtime to 
# manage how the data is stored and moved in memory.
#
# While buffers abstract how we represent and store data in a program,we do not directly access the data using the buffer. Instead, we use accessor objects that inform the runtime how we intend	to use
# the data we are accessing.
#
# Buffers are accessible on the host and may be accessible on multiple devices.
#

# ## Buffer creation
#
# The	buffer class is a template class with __three template arguments__. The first argument is the __type of the object__ that the buffer will contain. This type must be safe to copy byte by byte without using any special copy constructors.
#
# The second template argument is __an integer describing the dimensionality of the buffer__.
#
# The final template optional argument, is usually the __default value and this argument specifies a C++ style allocator class that is used to perform any memory allocations on the host__.
#
# __The choice of buffer creation depends on how the buffer needs to be used as well as programmer's coding preferences__. Below are some examples of most common wat how to create a buffer.
#
# * We can initialize __one-dimensional buffers using containers in two different ways__.
#
# * If the container object that provides the initial values for a buffer is also contiguous, then we can use an even simpler form to create the buffer. Buffer b1 creates a buffer from a __vector simply by passing the vector to the constructor__.
#
# The size of the buffer is the size of the container used to initialize it, and the type for the buffer data is the type of the container data.
# ```
#   // Create a buffer of ints from an input iterator
#   std::vector<int> myVec;
#   buffer b1{myVec};
# ```
#
# The below way uses __input iterators where we pass the beginning of the data and the other is the iterator to the end of the obejct__.  
#
#     
# * Buffer b2 is initailized using the start and end iterators as below. 
#
# ```
#   // Create a buffer of ints from an input iterator
#   std::vector<int> myVec;
#   buffer b2{myVec.begin(), myVec.end()};
# ```
#
# * When creating the below  buffers, we let buffer allocate the memory it needs. We can also initialize the buffers with any values at the time of their creation instead of buffers allocating the memory. We can use buffers to effectively wrap existing C++ allocations by passing a source of initial values to the buffer constructor as below.
#
#     Buffer b3 creates a __one-dimensional buffer of 4 doubles__. We pass the host pointer to the C++ array myDoubles to the buffer constructor in addition to the range that specifies the size of the buffer. By passing a pointer to host memory we also need to make sure should not try to access the host memory during the lifetime of the buffer to avoid any data race conditions.
#
# ```
# // Create a buffer of 4 doubles and initialize it from a host pointer
#   double myDoubles[4] = {1.1, 2.2, 3.3, 4.4};
#   buffer b3{myDoubles, range{4}};
# ```

# ### Buffer creation examples
# The DPC++ code below demonstrates different ways to create buffers: Inspect code, there are no modifications necessary:
#
# 1. Inspect the code cell below and click run ▶ to save the code to file.
#
# 2. Next run ▶ the cell in the __Build and Run__ section below the code to compile and execute the code.

# +
# %%writefile lab/buffer_creation.cpp
//==============================================================
// Copyright © 2020 Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================

#include <CL/sycl.hpp>
using namespace sycl;

int main() {
    
   // Create a buffer of ints from an input iterator
  std::vector<int> myVec;
  buffer b1{myVec};
  buffer b2{myVec.begin(), myVec.end()};
  
  // Create a buffer of ints from std::array
  std::array<int,42> my_data;  
  buffer b3{my_data};
  
  
  // Create a buffer of 4 doubles and initialize it from a host pointer
  double myDoubles[4] = {1.1, 2.2, 3.3, 4.4};
  buffer b4{myDoubles, range{4}}; 

  return 0;
}
# -

# #### Build and Run
# Select the cell below and click run ▶ to compile and execute the code:

# ! chmod 755 q; chmod 755 run_buffer_creation.sh; if [ -x "$(command -v qsub)" ]; then ./q run_buffer_creation.sh; else ./run_buffer_creation.sh; fi

# ## Buffer Properties
# Buffers simply represent data, and it becomes the job of the runtime to manage how the data is stored and moved in memory.
#
# While buffers abstract how we represent and store data in a program,we do not directly access the data using the buffer. Instead, we use accessor objects that inform the runtime how we intend	to use the data we are accessing.
#
# Buffers are accessible on the host and may be accessible on multiple devices. Below are some of the properties of the buffer that the programmers can take advantage of

# ### use_host_ptr
# The first property that may be optionally specified during buffer creation is use_host_ptr. __When present, this property requires the buffer to not allocate any memory on the host__, any allocator passed or specified is effectively ignored. Instead, the
# buffer should use the memory pointed to by a host pointer that is passed to the constructor.
#
# Also note that this property __may only be used when a host pointer is passed to the constructor__. 
#
# This option can be useful when the program wants full control over all host memory allocations. 
# ```cpp
# int main() {
# queue q;
# int myInts[42];
# // create a buffer of 42 ints, initialize with a host pointer,
# // and add the use_host_pointer property
# buffer b1(myInts, range(42), property::use_host_ptr{});
# ```

# The DPC++ code below demonstrates usage of use_host_ptr{} property: Inspect code, there are no modifications necessary:
#
# 1. Inspect the code cell below and click run ▶ to save the code to file.
#
# 2. Next run ▶ the cell in the __Build and Run__ section below the code to compile and execute the code.

# +
# %%writefile lab/buffer_host_ptr.cpp
//==============================================================
// Copyright © 2020 Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================
#include <mutex>
#include <CL/sycl.hpp>
using namespace sycl;
static const int N = 20;

int main() {
int myInts[N];
queue q;
//Initialize vector a,b and c
std::vector<float> a(N, 10.0f);
std::vector<float> b(N, 20.0f);

auto R = range<1>(N);
{
    //Create host_ptr buffers for a and b
    buffer buf_a(a,{property::buffer::use_host_ptr()});
    buffer buf_b(b,{property::buffer::use_host_ptr()});    
    
    q.submit([&](handler& h) {
        //create Accessors for a and b
        accessor A(buf_a,h);
        accessor B(buf_b,h,read_only);        
        h.parallel_for(R, [=](auto i) { A[i] += B[1] ; });
      });
}
    
for (int i = 0; i < N; i++) std::cout << a[i] << " ";
return 0;
}
# -

# #### Build and Run
# Select the cell below and click run ▶ to compile and execute the code:

# ! chmod 755 q; chmod 755 run_hst_ptr.sh; if [ -x "$(command -v qsub)" ]; then ./q run_hst_ptr.sh; else ./run_hst_ptr.sh; fi

# _If the Jupyter cells are not responsive or if they error out when you compile the code samples, please restart the Jupyter Kernel: 
# "Kernel->Restart Kernel and Clear All Outputs" and compile the code samples again_

# _If the Jupyter cells are not responsive or if they error out when you compile the code samples, please restart the Jupyter Kernel: 
# "Kernel->Restart Kernel and Clear All Outputs" and compile the code samples again_

# ### set_final_data
#
# Data in the buffer objects cannot be accessed directly and we must create accessors to access this data. We can query the buffer object to retrieve multiple characteristics of this object. We can query the range of a buffer, the total number of data elements it represents, query which allocator object is being used and whether the buffer is sub-buffer or not. 
#
# Buffers can be initialized using a pointer to host memory and once the buffer is destructed the data is written back to the host memory. Updating the host memory is an important task when using the buffer. 
#
# If a buffer is created and initialized from a host pointer to non-const data the same pointer is updated with the updated data when the buffer is destroyed. __The `set_final_data` method of a buffer is the way to update host memory however the buffer was created. When the buffer is destroyed, data will be written to the host using the supplied location.__
#
# The `set_final_data` method of a buffer is a template method that can accept either a raw pointer, a C++ OutputIterator, or a std::weak_ptr. 
#
# Technically, a raw pointer is a	special	case of	an OutputIterator. If the parameter passed to set_final_data is	a std::weak_ptr, the data is not written to the host if the pointer has	expired, or already been deleted.

# The DPC++ code below demonstrates usage of set_final_data() : Inspect code, there are no modifications necessary:
#
# 1. Inspect the code cell below and click run ▶ to save the code to file.
# 2. Next run ▶ the cell in the __Build and Run__ section below the code to compile and execute the code.

# +
# %%writefile lab/buffer_set_final_data.cpp
//==============================================================
// Copyright © 2020 Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================
#include <CL/sycl.hpp>
#include <array>
using namespace sycl;
constexpr int N = 42;

int main() {
  std::array<int,N> my_data;        
  for (int i = 0; i < N; i++)
        my_data[i] = i;     
 
  auto buff = std::make_shared<std::array<int, N>>(); 
  
  {
    queue q;
    buffer my_buffer(my_data);
      
    //Call the set_final_data to the created shared ptr where the values will be written back when the buffer gets destructed.
    //my_buffer.set_final_data(nullptr);    
    my_buffer.set_final_data(buff);   

    q.submit([&](handler &h) {
        // create an accessor to update
        // the buffer on the device
        accessor my_accessor(my_buffer, h);

        h.parallel_for(N, [=](id<1> i) {
            my_accessor[i]*=2;
          });
      });    
  }

  // myData is updated when myBuffer is
  // destroyed upon exiting scope
 
  for (int i = 0; i < N; i++) {
    std::cout << my_data[i] << " ";
  }
  std::cout << "\n"; 
  for (int i = 0; i < N; i++) {
    std::cout <<(*buff)[i] << " ";
  }
  std::cout << "\n"; 
  
}
# -

# #### Build and Run
# Select the cell below and click run ▶ to compile and execute the code:

# ! chmod 755 q; chmod 755 run_set_final_data.sh; if [ -x "$(command -v qsub)" ]; then ./q run_set_final_data.sh; else ./run_set_final_data.sh; fi

# ### set_write_back
#
#  __We can control whether or not writeback occurs from the device to the host by calling the set_write_back method__. This takes a boolen value and we can set it it to false if you do not want to have the results copied back to the host

# The DPC++ code below demonstrates usage of set_write_back() : Inspect code, there are no modifications necessary:
#
# 1. Inspect the code cell below and click run ▶ to save the code to file.
# 2. Next run ▶ the cell in the __Build and Run__ section below the code to compile and execute the code.

# +
# %%writefile lab/buffer_set_write_back.cpp
//==============================================================
// Copyright © 2020 Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================
#include <CL/sycl.hpp>
#include <array>
using namespace sycl;
constexpr int N = 42;

int main() {
  std::array<int,N> my_data;  
        
  for (int i = 0; i < N; i++)
        my_data[i] = i;
    
  {
    queue q(gpu_selector{});
    buffer my_buffer(my_data);
      
    //Call the set_write_back method to control the data to be written back to the host from the device. e
    //Setting it to false will not update the host with the updated values
         
    my_buffer.set_write_back(false);    

    q.submit([&](handler &h) {
        // create an accessor to update
        // the buffer on the device
        accessor my_accessor(my_buffer, h);

        h.parallel_for(N, [=](id<1> i) {
            my_accessor[i]*=2;
          });
      });    
  }

  // myData is updated when myBuffer is
  // destroyed upon exiting scope
 
  for (int i = 0; i < N; i++) {
    std::cout << my_data[i] << " ";
  }
  
}
# -

# #### Build and Run
# Select the cell below and click run ▶ to compile and execute the code:

# ! chmod 755 q; chmod 755 run_set_write_back.sh; if [ -x "$(command -v qsub)" ]; then ./q run_set_write_back.sh; else ./run_set_write_back.sh; fi

# # Accessors

# Data represented by a buffer cannot be directly accessed through the buffer object. Instead, we must create accessor objects that allow us to safely access a buffer’s data. Accessors inform the runtime where and how we want to access data, allowing the runtime to ensure that the right data is in the right place at the right time and the kernels dont run until the data is available.
#
# ### Access modes
#
# When creating an accessor, we must inform the runtime how we are going to use it by	specifying an access mode as described in the below table.
# Access modes are how the runtime is able to perform implicit data movement.
# When the accessor is created with __access::mode::read_write__ we intend to both read and write to the buffer through the accessor. 
#
# __read_only__ tells the runtime that the data needs to be available on the device before this kernel can begin executing. Similarly, __write_only__ lets the runtime know that we will modify the contents of a buffer and may need to copy the results back after computation has ended.
#
# The runtime uses accessors to order the use of data, but it can also use this data to optimize scheduling of kernels and data movement.
#
# | Access Mode | Description |
# |:---|:---|
# | __read_only__ | Read only Access|
# | __write_only__ | Write-only access. Previous contents not discarded |
# | __read_write__ | Read and Write access |
#

# ### noinit 
#
# The second new parameter that we pass to accessor is an optional accessor property. The property we pass,`noinit`,
# lets the runtime know that the previous contents of the buffer can be discarded. 
#
# __The `noinit' property is useful because it can let the runtime eliminate unnecessary data movement__. In this example, since the first task is writing the initial values for our buffer, it’s unnecessary for the runtime to copy the uninitialized host memory to the device before the kernel executes.
#
# The `noinit` property is useful for this example, but it should not be used for read-modify-write cases or kernels where only some values in a buffer may be updated.

# The DPC++ code below demonstrates creating accessors: Inspect code, there are no modifications necessary:
#
# 1. Inspect the code cell below and click run ▶ to save the code to file
# 2. Next run ▶ the cell in the __Build and Run__ section below the code to compile and execute the code.

# +
# %%writefile lab/accessors_sample.cpp
//==============================================================
// Copyright © 2020 Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================


#include <CL/sycl.hpp>
#include <cassert>
using namespace sycl;
constexpr int N = 42;

int main() {
  queue Q;

  // Create 3 buffers of 42 ints
  buffer<int> A{range{N}};
  buffer<int> B{range{N}};
  buffer<int> C{range{N}};  

  Q.submit([&](handler &h) {
      // create device accessors
      accessor aA{A, h, write_only, noinit};
      accessor aB{B, h, write_only, noinit};
      accessor aC{C, h, write_only, noinit};
      h.parallel_for(N, [=](id<1> i) {
          aA[i] = 1;
          aB[i] = 40;
          aC[i] = 0;
        });
    });
  Q.submit([&](handler &h) {
      // create device accessors
      accessor aA{A, h, read_only};
      accessor aB{B, h, read_only};
      accessor aC{C, h, read_write};
      h.parallel_for(N, [=](id<1> i) { aC[i] += aA[i] + aB[i]; });
    }); 

  host_accessor result{C, read_only};
    
  for (int i = 0; i < N; i++) std::cout << result[i] << " ";  
  
  return 0;
}
# -

# ### Build and Run
# Select the cell below and click run ▶ to compile and execute the code:

# ! chmod 755 q; chmod 755 run_accessor.sh;if [ -x "$(command -v qsub)" ]; then ./q run_accessor.sh; else ./run_accessor.sh; fi

# ## Host Accessors
#
# The __Host Accessor__ is an accessor which uses host buffer access target. Host accessors perform two functions. First, they make __data available for access on the host__, as their name implies. Secondly, they __synchronize with the host__ by defining a new dependence between the currently accessing graph and the host. This ensures that the data that gets copied back to the host is the correct value of the computation the graph was performing. 
#
# Buffer takes ownership of the data stored in vector. Creating host accessor is a __blocking call__ and execution on the host may not proceed past the creation of the host accessor until the data is available. Likewise, a buffer cannot be used on a device while a host accessor exists and keeps its data available, consider creating host accessors inside additional C++ scopes in order to free the data once the host accessor is no longer needed. This is an example of the next method for host synchronization.
#
# Certain objects in DPC++ have special behaviors when they are destroyed, and their destructors are invoked. We just learned how host accessors can tie up data on the host until they are destroyed. Buffers and images also have special behavior when they are
# destroyed or leave scope. 
#
# When a buffer is destroyed, it waits for all command groups that use that buffer to finish execution. Once a buffer is no longer being used by any kernel or memory operation, the runtime may have to copy data back to the host. This copy occurs either if the buffer was initialized with a host pointer or if a host pointer was passed to the method set_final_data. The runtime will then copy back the data for that buffer and update the host pointer before the object is destroyed.
#
#
#
# The DPC++ code below demonstrates Synchronization with Host Accessor: Inspect code, there are no modifications necessary:
#
#
# 1. Inspect the code cell below and click run ▶ to save the code to file
#
# 2. Next run ▶ the cell in the __Build and Run__ section below the code to compile and execute the code.
#

# +
# %%writefile lab/host_accessor_sample.cpp
//==============================================================
// Copyright © 2020 Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================

#include <CL/sycl.hpp>
#include <algorithm>
#include <iostream>
using namespace sycl;

int main() {

  static const int N = 1024;

  // Set up queue on any available device
  queue q;

  // Create host containers to initialize on the host
  std::vector<int> in_vec(N), out_vec(N);

  // Initialize input and output vectors
  for (int i=0; i < N; i++) in_vec[i] = i;
  std::fill(out_vec.begin(), out_vec.end(), 0);

  // Create buffers using host allocations (vector in this case)
  buffer in_buf{in_vec}, out_buf{out_vec};

  // Submit the kernel to the queue
  q.submit([&](handler& h) {
    accessor in{in_buf, h};
    accessor out{out_buf, h};

    h.parallel_for(range{N}, [=](id<1> idx) {
      out[idx] = in[idx] * 2;
    });
  });

  // Check that all outputs match expected value
  // Use host accessor!  Buffer is still in scope / alive
  host_accessor A{out_buf};

  //for (int i=0; i<N; i++) std::cout << "A[" << i << "]=" << A[i] << "\n";
    
 int indices[]{0, 1, 2, 3, 4, (N - 1)};
 constexpr size_t indices_size = sizeof(indices) / sizeof(int); 

  for (int i = 0; i < indices_size; i++) {
    int j = indices[i];
    if (i == indices_size - 1) std::cout << "...\n";
    std::cout << "A[" << j << "]=" << A[j] << "\n";
  }

  return 0;
}

# -

# ### Build and Run
# Select the cell below and click run ▶ to compile and execute the code:

# ! chmod 755 q; chmod 755 run_host_accessor.sh;if [ -x "$(command -v qsub)" ]; then ./q run_host_accessor.sh; else ./run_host_accessor.sh; fi

# _If the Jupyter cells are not responsive or if they error out when you compile the code samples,please restart the Jupyter Kernel: 
# "Kernel->Restart Kernel and Clear All Outputs" and compile the code samples again_

# ### Initialize buffer data using Host accessors
# The below example shows how we can initialize the buffer data using host accessors. We created an input buffer (in_buf) and an output buffer (out_buf) and then created two separate host_accessors in_acc, out_acc. Please note that it is very important to create these host_accessors in a separate scope so that the buffer values are initialized properly. Once the host accessor scope ends we submit the job where we are assigning the values of the output buffer to the input buffer and the results are copied back to the host from the device using the other host accessor (A).
#
#
# The DPC++ code below demonstrates Synchronization with Host Accessor: Inspect code, there are no modifications necessary:
#
#
# 1. Inspect the code cell below and click run ▶ to save the code to file
#
# 2. Next run ▶ the cell in the __Build and Run__ section below the code to compile and execute the code.
#

# +
# %%writefile lab/host_accessor_init.cpp
//==============================================================
// Copyright © 2020 Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================

#include <CL/sycl.hpp>
#include <algorithm>
#include <iostream>
using namespace sycl;

int main() {


  constexpr size_t N = 1024;

  // Set up queue on any available device
  queue q;

  // Create buffers of size N
  buffer<int> in_buf{N}, out_buf{N};

  // Use host accessors to initialize the data
  { // CRITICAL: Begin scope for host_accessor lifetime!
    host_accessor in_acc{ in_buf }, out_acc{ out_buf };
    for (int i=0; i < N; i++) {
      in_acc[i] = i;
      out_acc[i] = 0;
    }
  } //Close scope to make host accessors go out of scope!

  // Submit the kernel to the queue
  q.submit([&](handler& h) {
    accessor in{in_buf, h};
    accessor out{out_buf, h};

    h.parallel_for(range{N}, [=](id<1> idx) {
      out[idx] = in[idx];
    });
  });

  // Check that all outputs match expected value
  // Use host accessor!  Buffer is still in scope / alive
  host_accessor A{out_buf};

  //for (int i=0; i<N; i++) std::cout << "A[" << i << "]=" << A[i] << "\n";
  int indices[]{0, 1, 2, 3, 4, (N - 1)};
  constexpr size_t indices_size = sizeof(indices) / sizeof(int); 

  for (int i = 0; i < indices_size; i++) {
    int j = indices[i];
    if (i == indices_size - 1) std::cout << "...\n";
    std::cout << "A[" << j << "]=" << A[j] << "\n";
  }

  return 0;
}

# -

# ### Build and Run
# Select the cell below and click run ▶ to compile and execute the code:

# ! chmod 755 q; chmod 755 run_host_accessor_init.sh;if [ -x "$(command -v qsub)" ]; then ./q run_host_accessor_init.sh; else ./run_host_accessor_init.sh.sh; fi

# _If the Jupyter cells are not responsive or if they error out when you compile the code samples,please restart the Jupyter Kernel: 
# "Kernel->Restart Kernel and Clear All Outputs" and compile the code samples again_

# ## Additional topics on Buffers

# ### Sub Buffers
#
# Creating sub-buffers is another feature of the buffer class. It is possible to create a view of a buffer from another buffer, or a sub-buffer. A sub-buffer requires three things, a reference to a parent buffer, a base index, and the range of the sub-buffer. A sub-buffer cannot be created from a sub-buffer. Multiple sub-buffers can be created from the same buffer, and they are free to overlap. 
#
# Buffer b10 is a two-dimensional buffer of integers with 5 integers per row. Next, we create two sub-buffers from buffer b10, sub-buffers b11 and b12. Buffer b11 starts at index (0,0) and contains every element in the first row. Similarly, buffer b12 starts at index (1,0) and contains every element in the second row, yielding two disjoint sub-buffers.
#
# __The main advantage of using the sub-buffers is since the sub-buffers do not overlap, different kernels can operate on different subbuffers concurrently.__
#
# ```
# // Create a buffer of 2x5 ints and 2 non-overlapping sub-buffers of 5 ints.
#   buffer<int, 2> b10{range{2, 5}};
#   buffer b11{b10, id{0, 0}, range{1, 5}};
#   buffer b12{b10, id{1, 0}, range{1, 5}};
# ```

# The DPC++ code below demonstrates Sub buffers: Inspect code, there are no modifications necessary:
#
# 1. Inspect the code cell below and click run ▶ to save the code to file
#
# 2. Next run ▶ the cell in the __Build and Run__ section below the code to compile and execute the code.

# +
# %%writefile lab/sub_buffers.cpp
//==============================================================
// Copyright © 2020 Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================

#include <CL/sycl.hpp>

using namespace sycl;

int main() {
    const int N = 64;
    const int num1 = 2;
    const int num2 = 3;
    int data[N];
    for (int i = 0; i < N; i++) data[i] = i;
    
    std::cout<<"BUffer Values: ";    
    for (int i = 0; i < N; i++) std::cout << data[i] << " "; 
    std::cout<<"\n";
    buffer B(data, range(N));

    //Create sub buffers with offsets and half of the range. 

    buffer<int> B1(B, 0, range{ N / 2 });
    buffer<int> B2(B, 32, range{ N / 2 });

    //Multiply the  elemets in first sub buffer by 2 
    queue q1;
    q1.submit([&](handler& h) {
        accessor a1(B1, h);
        h.parallel_for(N/2, [=](auto i) { a1[i] *= num1; });
    });

    //Multiply the  elemets in second sub buffer by 3    
    queue q2;
    q2.submit([&](handler& h) {
        accessor a2(B2, h);
        h.parallel_for(N/2, [=](auto i) { a2[i] *= num2; });
    });    
    
    //Host accessors to get the results back to the host from the device
    host_accessor b1(B1, read_only);
    host_accessor b2(B2, read_only);
    
    std::cout<<"Sub Buffer1: ";
    for (int i = 0; i < N/2; i++) std::cout<< b1[i] << " ";
    std::cout<<"\n";
    std::cout<<"Sub Buffer2: ";
    for (int i = 0; i < N/2; i++) std::cout << b2[i] << " ";

    return 0;
}

# -

# ### Build and Run
# Select the cell below and click run ▶ to compile and execute the code:

# ! chmod 755 q; chmod 755 run_sub_buffer.sh;if [ -x "$(command -v qsub)" ]; then ./q run_sub_buffer.sh; else ./run_sub_buffer.sh; fi

# ### Uncommon ways to create Buffers

#
# Below are the other additional ways to create a buffer and the below are __uncommon__. __The choice of buffer creation depends on how the buffer needs to be used as well as programmer's coding preferences__. Below are some examples of how to create a buffer.
#
# As discussed before , the buffer class is a template class with __three template arguments__. The first argument is the __type of the object__ that the buffer will contain, the second template argument is __an integer describing the dimensionality of the buffer__ and the final template optional argument, is usually the __default value__ and this argument specifies a C++ style allocator class that is used to perform any memory allocations on the host.
#
#
# * Buffer b1 is created based on modern C++ and is a __two-dimensional buffer of 10 integers that uses the default allocator__. Here we make use of C++17’s class template argument deduction (CTAD) to automatically infer some of the template arguments. 
#     
#     In this case, we initialize buffer with a two-dimensional range to infer that it is a two-dimensional buffer. Please also note that the allocator template argument has a default value.
#     
# ```
#   // Create a buffer of 2x5 ints using the default allocator and CTAD for range
#   buffer<int, 2> b1{range{2, 5}};
# ```
#
# * Buffer b2 is similar to buffer b1 but here we use C++ CTAD to automatically infer that the buffer is one-dimensional.
#
# ```
#   // Create a buffer of 20 floats of 1 dimension using a default-constructed std::allocator
#   buffer<float> b2{range{20}};
# ```
#
# * If your application uses shared pointers, buffers can also be created using __C++ shared pointer objects__. This method of initialization will properly count the reference and ensure that the memory is not deallocated. Buffer b3 initializes a buffer from a single integer and initializes it using a shared pointer as below.
#
# ```
#   // Create a buffer from a shared pointer to int
#   auto sharedPtr = std::make_shared<int>(42);
#   buffer b3{sharedPtr, range{1}};
# ```
#
#
# * Buffer b4 is a two-dimensional buffer of integers with 5 integers per row. Next, we create two sub-buffers from buffer b4, sub-buffers b5 and b6. Buffer b5 starts at index (0,0) and contains every element in the first row. Similarly, buffer b6 starts at index (1,0) and contains every element in the second row, yielding two disjoint sub-buffers.
#
# ```
# // Create a buffer of 2x5 ints and 2 non-overlapping sub-buffers of 5 ints.
#   buffer<int, 2> b4{range{2, 5}};
#   buffer b5{b4, id{0, 0}, range{1, 5}};
#   buffer b6{b4, id{1, 0}, range{1, 5}};
# ```
#
# * In Buffer b7 we are initializing the buffer with a __pointer to const double__. In this case we can only read values through the host pointer, not write them. However, the type for our buffer while creation is still double, but not const double. This means that the buffer may be written to by a kernel, but we must use a different mechanism to update the host after the buffer gets destructed. 
#
# ```
#   // Create a buffer of 5 doubles and initialize it from a host pointer to
#   // const double
#   const double myConstDbls[5] = {1.0, 2.0, 3.0, 4.0, 5.0};
#   buffer b4{myConstDbls, range{5}};
# ```
#
#
#

# ### Buffer creation examples
# The DPC++ code below demonstrates different ways to create buffers: Inspect code, there are no modifications necessary:
#
# 1. Inspect the code cell below and click run ▶ to save the code to file.
#
# 2. Next run ▶ the cell in the __Build and Run__ section below the code to compile and execute the code.

# +
# %%writefile lab/buffer_creation_uncommon.cpp
//==============================================================
// Copyright © 2020 Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================

#include <CL/sycl.hpp>
using namespace sycl;

int main() {
  
  // Create a buffer of 2x5 ints using the default allocator and CTAD for dimensions
  buffer<int, 2> b1{range{2, 5}};
    
  //Dimensions defaults to 1

  // Create a buffer of 20 floats using a default-constructed std::allocator
  buffer<float> b2{range{20}};
  
  // Create a buffer from a shared pointer to int
  auto sharedPtr = std::make_shared<int>(42);
  buffer b3{sharedPtr, range{1}};
  
  // Create a buffer of 2x5 ints and 2 non-overlapping sub-buffers of 5 ints.
  buffer<int, 2> b4{range{2, 5}};
  buffer b5{b4, id{0, 0}, range{1, 5}};
  buffer b6{b4, id{1, 0}, range{1, 5}};
    
  // Create a buffer of 5 doubles and initialize it from a host pointer to
  // const double
  const double myConstDbls[5] = {1.0, 2.0, 3.0, 4.0, 5.0};
  buffer b7{myConstDbls, range{5}};   

  return 0;
}
# -

# #### Build and Run
# Select the cell below and click run ▶ to compile and execute the code:

# ! chmod 755 q; chmod 755 run_buffer_creation_uncommon.sh; if [ -x "$(command -v qsub)" ]; then ./q run_buffer_creation_uncommon.sh; else ./run_buffer_creation_uncommon.sh; fi

# ***
# # Summary
#
# In this module you learned:
# * Buffers and Accessors in Depth
# * Buffers properties and when to use_host_ptr, set_final_data and set_write_data
# * Sub buffers and how to create and use Sub buffers
# * How to create Accessors, host accessors and initialize buffer data using host accessors
#
#
