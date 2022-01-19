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
# # Overview of DPC++
# -

# ##### Sections
# - [What is Data Parallel C++?](#What-is-Data-Parallel-C++?)
# - _Code:_ [Device Selector](#Device-Selector)
# - [Data Parallel Kernels](#Parallel-Kernels)
# - [DPC++ Code Anatomy](#DPC++-Code-Anatomy)
# - _Code:_ [Implicit dependency with Accessors](#Implicit-dependency-with-Accessors)
# - _Code:_ [Synchronization: Host Accessor](#Synchronization:-Host-Accessor)
# - _Code:_ [Synchronization: Buffer Destruction](#Synchronization:-Buffer-Destruction)
# - _Code:_ [Custom Device Selector](#Custom-Device-Selector)
# - _Code:_ [Complex Number Multiplication](#Lab-Exercise:-Complex-Number-Multiplication)

# ## Learning Objectives
# * Explain the __SYCL__ fundamental classes
# * Use __device selection__ to offload kernel workloads
# * Decide when to use __basic parallel kernels__ and __ND Range Kernels__
# * Create a __host Accessor__
# * Build a sample __DPC++ application__ through hands-on lab exercises

# ## What is Data Parallel C++?
# __oneAPI__ programs are written in __Data Parallel C++ (DPC++)__. It is based on modern C++ productivity benefits and familiar constructs and incorporates the __SYCL__ standard for data parallelism and heterogeneous programming. DPC++ is a __single source__ where __host code__ and __heterogeneous accelerator kernels__ can be mixed in same source files. A DPC++ program is invoked on the host computer and offloads the computation to an accelerator. Programmers use familiar C++ and library constructs with added functionliaties like a __queue__ for work targeting, __buffer__ for data management, and __parallel_for__ for parallelism to direct which parts of the computation and data should be offloaded.

# ## Device
# The __device__ class represents the capabilities of the accelerators in a system utilizing Intel&reg; oneAPI Toolkits. The device class contains member functions for querying information about the device, which is useful for DPC++ programs where multiple devices are created.
# * The function __get_info__ gives information about the device:
#  * Name, vendor, and version of the device
#  * The local and global work item IDs
#  * Width for built in types, clock frequency, cache width and sizes, online or offline
#  
# ```cpp
# queue q;
# device my_device = q.get_device();
# std::cout << "Device: " << my_device.get_info<info::device::name>() << "\n";
# ```
#

# ## Device Selector
# The  __device_selector__ class enables the runtime selection of a particular device to execute kernels based upon user-provided heuristics. The following code sample shows use of the standard device selectors (__default_selector, cpu_selector, gpu_selector…__) and a derived device_selector
#
#  
# ```cpp
# default_selector selector;
# // host_selector selector;
# // cpu_selector selector;
# // gpu_selector selector;
# queue q(selector);
# std::cout << "Device: " << q.get_device().get_info<info::device::name>() << "\n";
# ```
#
# The DPC++ code below shows different device selectors: Inspect code, there are no modifications necessary:
# 1. Inspect the code cell below and click run ▶ to save the code to file
# 2. Next run ▶ the cell in the __Build and Run__ section below the code to compile and execute the code.

# +
# %%writefile lab/gpu_sample.cpp
//==============================================================
// Copyright © 2020 Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================
#include <CL/sycl.hpp>

using namespace cl::sycl;

int main() {
  //# Create a device queue with device selector
  
  gpu_selector selector;
  //cpu_selector selector;
  //default_selector selector;
  //host_selector selector;
  
  queue q(selector);

  //# Print the device name
  std::cout << "Device: " << q.get_device().get_info<info::device::name>() << "\n";

  return 0;
}
# -

# ### Build and Run
# Select the cell below and click run ▶ to compile and execute the code:

# ! chmod 755 q; chmod 755 run_gpu.sh;if [ -x "$(command -v qsub)" ]; then ./q run_gpu.sh; else ./run_gpu.sh; fi

# _If the Jupyter cells are not responsive or if they error out when you compile the code samples, please restart the Jupyter Kernel: 
# "Kernel->Restart Kernel and Clear All Outputs" and compile the code samples again_

# ## Queue
# __Queue__ submits command groups to be executed by the SYCL runtime. Queue is a mechanism where __work is submitted__ to a device.A queue map to one device and multiple queues can be mapped to the same device.
#  
# ```cpp
# q.submit([&](handler& h) {
#     //COMMAND GROUP CODE
# });
# ```
#
#

# ## Kernel
# The __kernel__ class encapsulates methods and data for executing code on the device when a command group is instantiated. Kernel object is not explicitly constructed by the user and is is constructed when a kernel dispatch function, such as __parallel_for__, is called 
#  ```cpp
#  q.submit([&](handler& h) {
#   h.parallel_for(range<1>(N), [=](id<1> i) {
#     A[i] = B[i] + C[i]);
#   });
# });
# ```
#

# ## Choosing where device kernels run
#
# Work is submitted to queues and each queue is associated with exactly one device (e.g. a specific GPU or FPGA). You can decide which device a queue is associated with (if you want) and have as many queues as desired for dispatching work in heterogeneous systems.        
#
# |Target Device |Queue|
# |-----|-------|
# |Create queue targeting any device: | queue() |
# | Create queue targeting a pre-configured classes of devices: | queue(cpu_selector{}); queue(gpu_selector{});  queue(INTEL::fpga_selector{}); queue(accelerator_selector{}); queue(host_selector{});|
# |Create queue targeting specific device (custom criteria): |class custom_selector : public device_selector {int operator()(……  // Any logic you want!  … queue(custom_selector{}); |                    
#                                                           
#                                                                
#
#
#
#
#
#
#
#         
# <img src="Assets/queue.png">

# ## DPC++ Language and Runtime
# DPC++ language and runtime consists of a set of C++ classes, templates, and libraries.
#
#  __Application scope__ and __command group scope__ :
#  * Code that executes on the host
#  * The full capabilities of C++ are available at application and command group scope 
#
# __Kernel__ scope:
#  * Code that executes on the device. 
#  * At __kernel__ scope there are __limitations__ in accepted C++
#

# ## Parallel Kernels
#
# __Parallel Kernel__ allows multiple instances of an operation to execute in parallel. This is useful to __offload__ parallel execution of a basic __for-loop__ in which each iteration is completely independent and in any order. Parallel kernels are expressed using the __parallel_for__ function
# A simple 'for' loop in a C++ application is written as below
#
# ```cpp
# for(int i=0; i < 1024; i++){
#     a[i] = b[i] + c[i];
# });
# ```
#
# Below is how you can offload to accelerator
#
# ```cpp
# h.parallel_for(range<1>(1024), [=](id<1> i){
#     A[i] =  B[i] + C[i];
# });
# ```
#

# ## Basic Parallel Kernels
#
# The functionality of basic parallel kernels is exposed via __range__, __id__, and __item__ classes. __Range__ class is used to describe the __iteration space__ of parallel execution and __id__ class is used to __index__ an individual instance of a kernel in a parallel execution
#
#
# ```cpp
# h.parallel_for(range<1>(1024), [=](id<1> i){
# // CODE THAT RUNS ON DEVICE 
# });
#
# ```
# The above example is good if all you need is the __index (id)__, but if you need the __range__ value in your kernel code, then you can use __item__ class instead of __id__ class , which you can use to query for the __range__ as shown below.  __item__ class represents an __individual instance__ of a kernel function, exposes additional functions to query properties of the execution range
#
#
# ```cpp
# h.parallel_for(range<1>(1024), [=](item<1> item){
#     auto i = item.get_id();
#     auto R = item.get_range();
#     // CODE THAT RUNS ON DEVICE
#     
#     
# });
#
# ```

# ## ND RANGE KERNELS
# Basic Parallel Kernels are easy way to parallelize a for-loop but does not allow performance optimization at hardware level. __ND-Range kernel__ is another way to expresses parallelism which enable low level performance tuning by providing access to __local memory and mapping executions__ to compute units on hardware. The entire iteration space is divided into smaller groups called __work-groups__, __work-items__ within a work-group are scheduled on a single compute unit on hardware.
#
# The grouping of kernel executions into work-groups  will allow control of resource usage and load balance work distribution.The functionality of nd_range kernels is exposed via __nd_range__ and __nd_item__ classes. __nd_range__ class represents a __grouped execution range__ using global execution range and the local execution range of each work-group. __nd_item__ class  represents an __individual instance__ of a kernel function and allows to query for work-group range and index.
#
# ```cpp
# h.parallel_for(nd_range<1>(range<1>(1024),range<1>(64)), [=](nd_item<1> item){
#     auto idx = item.get_global_id();
#     auto local_id = item.get_local_id();
#     // CODE THAT RUNS ON DEVICE
# });
# ```
# <img src="Assets/ndrange.png">

# ## Buffer Model
# __Buffers encapsulate__ data in a SYCL application across both devices and host. __Accessors__ is the mechanism to access buffer data.

# ### DPC++ Code Anatomy
# Programs which utilize oneAPI require the include of __cl/sycl.hpp__. It is recommended to employ the namespace statement to save typing repeated references into the cl::sycl namespace.
#
# ```cpp
# #include <CL/sycl.hpp>
# using namespace cl::sycl;
# ```
#
# __DPC++ programs__ are standard C++. The program is invoked on the __host__ computer, and offloads computation to the __accelerator__. A programmer uses DPC++’s __queue, buffer, device, and kernel abstractions__ to direct which parts of the computation and data should be offloaded.
#
# As a first step in a DPC++ program we create a __queue__. We offload computation to a __device__ by submitting tasks to a queue. The programmer can choose CPU, GPU, FPGA, and other devices through the __selector__. This program uses the default  q here, which means DPC++ runtime selects the most capable device available at runtime by using the default selector. We will talk about the devices, device selectors, and the concepts of buffers, accessors and kernels in the upcoming modules but below is a simple DPC++ program for you to get started with the above concepts.
#
# Device and host can either share physical __memory__ or have distinct memories. When the memories are distinct, offloading computation requires __copying data between host and device__. DPC++ does not require the programmer to manage the data copies. By creating __Buffers and Accessors__, DPC++ ensures that the data is available to host and device without any programmer effort. DPC++ also allows the programmer explicit control over data movement when it is necessary to achieve best peformance.
#
# In a DPC++ program, we define a __kernel__, which is applied to every point in an index space. For simple programs like this one, the index space maps directly to the elements of the array. The kernel is encapsulated in a __C++ lambda function__. The lambda function is passed a point in the index space as an array of coordinates. For this simple program, the index space coordinate is the same as the array index. The __parallel_for__ in the below program applies the lambda to the index space. The index space is defined in the first argument of the parallel_for as a 1 dimensional __range from 0 to N-1__.
#
#
# The code below shows Simple Vector addition using DPC++. Read through the comments addressed in step 1 through step 6.
#
# ```cpp
# void dpcpp_code(int* a, int* b, int* c, int N) {
#   //Step 1: create a device queue
#   //(developer can specify a device type via device selector or use default selector)
#   auto R = range<1>(N);
#   queue q;
#   //Step 2: create buffers (represent both host and device memory)
#   buffer buf_a(a, R);
#   buffer buf_b(b, R);
#   buffer buf_c(c, R);
#   //Step 3: submit a command for (asynchronous) execution
#   q.submit([&](handler &h){
#   //Step 4: create buffer accessors to access buffer data on the device
#   accessor A(buf_a,h,read_only);
#   accessor B(buf_b,h,read_only);
#   accessor C(buf_c,h,write_only);
#   
#   //Step 5: send a kernel (lambda) for execution
#   h.parallel_for(range<1>(N), [=](auto i){
#     //Step 6: write a kernel
#     //Kernel invocations are executed in parallel
#     //Kernel is invoked for each element of the range
#     //Kernel invocation has access to the invocation id
#     C[i] = A[i] + B[i];
#     });
#   });
# }
# ```

#
# ## Implicit dependency with Accessors
# * Accessors create __data dependencies__ in the SYCL graph that order kernel executions
# * If two kernels use the same buffer, the second kernel needs to wait for the completion of the first kernel to avoid race conditions. 
#
#
# <img src="Assets/buffer1.png">
#
#
# The DPC++ code below demonstrates Implicit dependency with Accessors: Inspect code, there are no modifications necessary:
#
# 1. Inspect the code cell below and click run ▶ to save the code to file
#
# 2. Next run ▶ the cell in the __Build and Run__ section below the code to compile and execute the code.

# +
# %%writefile lab/buffer_sample.cpp
//==============================================================
// Copyright © 2020 Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================
#include <CL/sycl.hpp>

constexpr int num=16;
using namespace sycl;

  int main() {
  auto R = range<1>{ num };
  //Create Buffers A and B
  buffer<int> A{ R }, B{ R };
  //Create a device queue
  queue Q;
  //Submit Kernel 1
  Q.submit([&](handler& h) {
    //Accessor for buffer A
    accessor out(A,h,write_only);
    h.parallel_for(R, [=](auto idx) {
      out[idx] = idx[0]; }); });
  //Submit Kernel 2
  Q.submit([&](handler& h) {
    //This task will wait till the first queue is complete
    accessor out(A,h,write_only);
    h.parallel_for(R, [=](auto idx) {
      out[idx] += idx[0]; }); });
  //Submit Kernel 3
  Q.submit([&](handler& h) { 
    //Accessor for Buffer B
    accessor out(B,h,write_only);
    h.parallel_for(R, [=](auto idx) {
      out[idx] = idx[0]; }); });
  //Submit task 4
  Q.submit([&](handler& h) {
   //This task will wait till kernel 2 and 3 are complete
   accessor in (A,h,read_only);
   accessor inout(B,h);
  h.parallel_for(R, [=](auto idx) {
    inout[idx] *= in[idx]; }); }); 
      
 // And the following is back to device code
 host_accessor result(B,read_only);
  for (int i=0; i<num; ++i)
    std::cout << result[i] << "\n";      
  return 0;
}
# -

# ### Build and Run
# Select the cell below and click run ▶ to compile and execute the code:

# ! chmod 755 q; chmod 755 run_buffer.sh;if [ -x "$(command -v qsub)" ]; then ./q run_buffer.sh; else ./run_buffer.sh; fi

# _If the Jupyter cells are not responsive or if they error out when you compile the code samples, please restart the Jupyter Kernel: 
# "Kernel->Restart Kernel and Clear All Outputs" and compile the code samples again_

# ## Host Accessors
# The Host Accessor is an accessor which uses host buffer access target. It is created outside of the scope of the command group and the data that this gives access to will be available on the host. These are used to synchronize the data back to the host by constructing the host accessor objects. Buffer destruction is the other way to synchronize the data back to the host.
#

# ## Synchronization: Host Accessor
#
# Buffer takes ownership of the data stored in vector. Creating host accessor is a __blocking call__ and will only return after all enqueued DPC++ kernels that modify the same buffer in any queue completes execution and the data is available to the host via this host accessor.
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
using namespace sycl;

int main() {
  constexpr int N = 16;
  auto R = range<1>(N);
  std::vector<int> v(N, 10);
  queue q;
  // Buffer takes ownership of the data stored in vector.  
  buffer buf(v);
  q.submit([&](handler& h) {
    accessor a(buf,h);
    h.parallel_for(R, [=](auto i) { a[i] -= 2; });
  });
  // Creating host accessor is a blocking call and will only return after all
  // enqueued DPC++ kernels that modify the same buffer in any queue completes
  // execution and the data is available to the host via this host accessor.
  host_accessor b(buf,read_only);
  for (int i = 0; i < N; i++) std::cout << b[i] << " ";
  return 0;
}
# -

# ### Build and Run
# Select the cell below and click run ▶ to compile and execute the code:

# ! chmod 755 q; chmod 755 run_host_accessor.sh;if [ -x "$(command -v qsub)" ]; then ./q run_host_accessor.sh; else ./run_host_accessor.sh; fi

# _If the Jupyter cells are not responsive or if they error out when you compile the code samples,please restart the Jupyter Kernel: 
# "Kernel->Restart Kernel and Clear All Outputs" and compile the code samples again_

# ## Synchronization: Buffer Destruction
# In the below example Buffer creation happens within a separate function scope. When execution advances beyond this __function scope__, buffer destructor is invoked which relinquishes the ownership of data and copies back the data to the host memory.
#
# The DPC++ code below demonstrates Synchronization with Buffer Destruction: Inspect code, there are no modifications necessary:
#
# 1. Inspect the code cell below and click run ▶ to save the code to a file.
#
# 2. Next run ▶ the cell in the __Build and Run__ section below the code to compile and execute the code.

# +
# %%writefile lab/buffer_destruction2.cpp
//==============================================================
// Copyright © 2020 Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================

#include <CL/sycl.hpp>
constexpr int N = 16;
using namespace sycl;

// Buffer creation happens within a separate function scope.
void dpcpp_code(std::vector<int> &v, queue &q) {
  auto R = range<1>(N);
  buffer buf(v);
  q.submit([&](handler &h) {
    accessor a(buf,h);
    h.parallel_for(R, [=](auto i) { a[i] -= 2; });
  });
}
int main() {
  std::vector<int> v(N, 10);
  queue q;
  dpcpp_code(v, q);
  // When execution advances beyond this function scope, buffer destructor is
  // invoked which relinquishes the ownership of data and copies back the data to
  // the host memory.
  for (int i = 0; i < N; i++) std::cout << v[i] << " ";
  return 0;
}
# -

# ### Build and Run
# Select the cell below and click run ▶ to compile and execute the code:

# ! chmod 755 q; chmod 755 run_buffer_destruction.sh;if [ -x "$(command -v qsub)" ]; then ./q run_buffer_destruction.sh; else ./run_buffer_destruction.sh; fi

# _If the Jupyter cells are not responsive or if they error out when you compile the code samples, please restart the Jupyter Kernel:
# "Kernel->Restart Kernel and Clear All Outputs" and compile the code samples again_

# ## Custom Device Selector
# The following code shows derived device_selector that employs a device selector heuristic. The selected device prioritizes a GPU device because the integer rating returned is higher than for CPU or other accelerator. 
#
# The DPC++ code below demonstrates Custom Device Selector: Inspect code, there are no modifications necessary:
#
# 1. Inspect the code cell below and click run ▶ to save the code to a file.
#
# 2. Next run ▶ the cell in the __Build and Run__ section below the code to compile and execute the code.

# +
# %%writefile lab/custom_device_sample.cpp
//==============================================================
// Copyright © 2020 Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================
#include <CL/sycl.hpp>
#include <iostream>
using namespace sycl;
class my_device_selector : public device_selector {
public:
    my_device_selector(std::string vendorName) : vendorName_(vendorName){};
    int operator()(const device& dev) const override {
    int rating = 0;
    //We are querying for the custom device specific to a Vendor and if it is a GPU device we
    //are giving the highest rating as 3 . The second preference is given to any GPU device and the third preference is given to
    //CPU device.
    if (dev.is_gpu() & (dev.get_info<info::device::name>().find(vendorName_) != std::string::npos))
        rating = 3;
    else if (dev.is_gpu()) rating = 2;
    else if (dev.is_cpu()) rating = 1;
    return rating;
    };
    
private:
    std::string vendorName_;
};
int main() {
    //pass in the name of the vendor for which the device you want to query 
    std::string vendor_name = "Intel";
    //std::string vendor_name = "AMD";
    //std::string vendor_name = "Nvidia";
    my_device_selector selector(vendor_name);
    queue q(selector);
    std::cout << "Device: "
    << q.get_device().get_info<info::device::name>() << "\n";
    return 0;
}

# -

# ### Build and Run
# Select the cell below and click run ▶ to compile and execute the code:

# ! chmod 755 q; chmod 755 run_custom_device.sh;if [ -x "$(command -v qsub)" ]; then ./q run_custom_device.sh; else ./run_custom_device.sh; fi

# _If the Jupyter cells are not responsive or if they error out when you compile the code samples, please restart the Jupyter Kernel: 
# "Kernel->Restart Kernel and Clear All Outputs" and compile the code samples again__

# # Lab Exercise: Complex Number Multiplication
# The following is the definition of a custom class type that represents complex numbers.
# * The file [Complex.hpp](./src/Complex.hpp) defines the Complex2 class.
# * The Complex2 Class got two member variables "real" and "imag" of type int.
# * The Complex2 class got a member function for performing complex number multiplication. The function complex_mul returns the object of type Complex2 performing the multiplication of two complex numbers.
# * We are going to call complex_mul function from our DPC++ code.

# ## LAB EXERCISE
# * In this lab we provide with the source code that computes multiplication of two complex numbers where Complex class is the definition of a custom type that represents complex numbers.
# * In this example the student will learn how to use custom device selector to target GPU or CPU of a specific vendor and then pass in a vector of custom Complex class objects in parallel.The student needs to modify the source code to select Intel® GPU as the first choice and then, setup  a write accessor and call the Complex class member function as kernel to compute the multiplication.
# * Follow the __Step1 and Step 2 and Step 3 in the below code__.
# * The Complex class in the below example is to demonstarte the usage of a custom class and how a custom class can be passed in a DPC++ code, but not to show the functionality of the complex class in the std library. You can use the std::complex library as it is on its own in a DPC++ program
#
# 1. Select the code cell below, __follow the STEPS 1 to 3__ in the code comments, click run ▶ to save the code to file.
# 2. Next run ▶ the cell in the __Build and Run__ section below the code to compile and execute the code.

# +
# %%writefile lab/complex_mult.cpp
//==============================================================
// Copyright © 2020 Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================
#include <CL/sycl.hpp>
#include <iomanip>
#include <vector>
// dpc_common.hpp can be found in the dev-utilities include folder.
// e.g., $ONEAPI_ROOT/dev-utilities/<version>/include/dpc_common.hpp
#include "dpc_common.hpp"
#include "Complex.hpp"

using namespace sycl;
using namespace std;

// Number of complex numbers passing to the DPC++ code
static const int num_elements = 10000;

class CustomDeviceSelector : public device_selector {
 public:
  CustomDeviceSelector(std::string vendorName) : vendorName_(vendorName){};
  int operator()(const device &dev) const override {
    int device_rating = 0;
    //We are querying for the custom device specific to a Vendor and if it is a GPU device we
    //are giving the highest rating as 3 . The second preference is given to any GPU device and the third preference is given to
    //CPU device. 
    //**************Step1: Uncomment the following lines where you are setting the rating for the devices********
    /*if (dev.is_gpu() & (dev.get_info<info::device::name>().find(vendorName_) !=
                        std::string::npos))
      device_rating = 3;
    else if (dev.is_gpu())
      device_rating = 2;
    else if (dev.is_cpu())
      device_rating = 1;*/
    return device_rating;
  };

 private:
  std::string vendorName_;
};

// in_vect1 and in_vect2 are the vectors with num_elements complex nubers and
// are inputs to the parallel function
void DpcppParallel(queue &q, std::vector<Complex2> &in_vect1,
                   std::vector<Complex2> &in_vect2,
                   std::vector<Complex2> &out_vect) {
  auto R = range(in_vect1.size());
  if (in_vect2.size() != in_vect1.size() || out_vect.size() != in_vect1.size()){ 
    std::cout << "ERROR: Vector sizes do not  match"<< "\n";
    return;
  }
  // Setup input buffers
  buffer bufin_vect1(in_vect1);
  buffer bufin_vect2(in_vect2);

  // Setup Output buffers 
  buffer bufout_vect(out_vect);

  std::cout << "Target Device: "
            << q.get_device().get_info<info::device::name>() << "\n";
  // Submit Command group function object to the queue
  q.submit([&](auto &h) {
    // Accessors set as read mode
    accessor V1(bufin_vect1,h,read_only);
    accessor V2(bufin_vect2,h,read_only);
    // Accessor set to Write mode
    //**************STEP 2: Uncomment the below line to set the Write Accessor******************** 
    //accessor V3 (bufout_vect,h,write_only);
    h.parallel_for(R, [=](auto i) {
      //**************STEP 3: Uncomment the below line to call the complex_mul function that computes the multiplication
      //of the  complex numbers********************
      //V3[i] = V1[i].complex_mul(V2[i]);
    });
  });
  q.wait_and_throw();
}
void DpcppScalar(std::vector<Complex2> &in_vect1,
                 std::vector<Complex2> &in_vect2,
                 std::vector<Complex2> &out_vect) {
  if ((in_vect2.size() != in_vect1.size()) || (out_vect.size() != in_vect1.size())){
    std::cout<<"ERROR: Vector sizes do not match"<<"\n";
    return;
    }
  for (int i = 0; i < in_vect1.size(); i++) {
    out_vect[i] = in_vect1[i].complex_mul(in_vect2[i]);
  }
}
// Compare the results of the two output vectors from parallel and scalar. They
// should be equal
int Compare(std::vector<Complex2> &v1, std::vector<Complex2> &v2) {
  int ret_code = 1;
  if(v1.size() != v2.size()){
    ret_code = -1;
  }
  for (int i = 0; i < v1.size(); i++) {
    if (v1[i] != v2[i]) {
      ret_code = -1;
      break;
    }
  }
  return ret_code;
}
int main() {
  // Declare your Input and Output vectors of the Complex2 class
  vector<Complex2> input_vect1;
  vector<Complex2> input_vect2;
  vector<Complex2> out_vect_parallel;
  vector<Complex2> out_vect_scalar;

  for (int i = 0; i < num_elements; i++) {
    input_vect1.push_back(Complex2(i + 2, i + 4));
    input_vect2.push_back(Complex2(i + 4, i + 6));
    out_vect_parallel.push_back(Complex2(0, 0));
    out_vect_scalar.push_back(Complex2(0, 0));
  }

  // Initialize your Input and Output Vectors. Inputs are initialized as below.
  // Outputs are initialized with 0
  try {
    // Pass in the name of the vendor for which the device you want to query
    std::string vendor_name = "Intel";
    // std::string vendor_name = "AMD";
    // std::string vendor_name = "Nvidia";
    // queue constructor passed exception handler
    CustomDeviceSelector selector(vendor_name);
    queue q(selector, dpc_common::exception_handler);
    // Call the DpcppParallel with the required inputs and outputs
    DpcppParallel(q, input_vect1, input_vect2, out_vect_parallel);
  } catch (...) {
    // some other exception detected
    std::cout << "Failure" << "\n";
    std::terminate();
  }

  std::cout
      << "****************************************Multiplying Complex numbers "
         "in Parallel********************************************************"
      << "\n";
  // Print the outputs of the Parallel function
  int indices[]{0, 1, 2, 3, 4, (num_elements - 1)};
  constexpr size_t indices_size = sizeof(indices) / sizeof(int);

  for (int i = 0; i < indices_size; i++) {
    int j = indices[i];
    if (i == indices_size - 1) std::cout << "...\n";
    std::cout << "[" << j << "] " << input_vect1[j] << " * " << input_vect2[j]
              << " = " << out_vect_parallel[j] << "\n";
  }
  // Call the DpcppScalar function with the required input and outputs
  DpcppScalar(input_vect1, input_vect2, out_vect_scalar);

  // Compare the outputs from the parallel and the scalar functions. They should
  // be equal

  int ret_code = Compare(out_vect_parallel, out_vect_scalar);
  if (ret_code == 1) {
    std::cout << "Complex multiplication successfully run on the device"
              << "\n";
  } else
    std::cout
        << "*********************************************Verification Failed. Results are "
           "not matched**************************"
        << "\n";

  return 0;
}

# -

# ### Build and Run
# Select the cell below and click run ▶ to compile and execute the code:

# ! chmod 755 q; chmod 755 run_complex_mult.sh; if [ -x "$(command -v qsub)" ]; then ./q run_complex_mult.sh; else ./run_complex_mult.sh; fi

# _If the Jupyter cells are not responsive or if they error out when you compile the code samples,please restart the Jupyter Kernel: 
# "Kernel->Restart Kernel and Clear All Outputs" and compile the code samples again_

# #### Solution
# - [complex_mult_solution.cpp](src/complex_mult_solution.cpp)

# ***
# # Summary
#
# In this module you learned:
# * The fundamental SYCL Classes
# * How to select the device to offload to kernel workloads
# * How to write a DPC++ program using Buffers, Accessors, Command Group handler, and kernel
# * How to use the Host accessors and Buffer destruction to do the synchronization
#

# <html><body><span style="color:green"><h1>Survey</h1></span></body></html>
#
# [We would appreciate any feedback you’d care to give, so that we can improve the overall training quality and experience. Thanks! ](https://intel.az1.qualtrics.com/jfe/form/SV_6zljPDDUQ0RBRsx)

# <html><body><span style="color:Red"><h1>Reset Notebook</h1></span></body></html>
#
# ##### Should you be experiencing any issues with your notebook or just want to start fresh run the below cell.
#
#

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
          # !rsync -a --size-only /data/oneapi_workshop/oneAPI_Essentials/02_DPCPP_Program_Structure/ ~/oneAPI_Essentials/02_DPCPP_Program_Structure
          print('Notebook reset -- now click reload on browser.')
# linking button and function together using a button's method
button.on_click(on_button_clicked)
# displaying button and its output together
widgets.VBox([button,out])
