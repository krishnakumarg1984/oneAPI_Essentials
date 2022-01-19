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

# # Introduction to oneAPI and DPC++

# ##### Sections
# - [oneAPI Programming Model Overview](#oneAPI-Software-Model-Overview)
# - [Programming Challenges for Multiple architectures](#Programming-Challenges-for-Multiple-architectures)
# - [Introducing oneAPI](#Introducing-oneAPI)
# - _Code:_ [DPC++ Hello World](#Simple-Exercise)
# - [What is Data Parallel C++](#What-is-Data-Parallel-C++)
# - [How to Compile & Run a DPC++ program](#How-to-Compile-&-Run-DPC++-program)
# - _Code:_ [Simple Vector Increment to Vector Add](#Lab-Exercise:-Simple-Vector-Increment-TO-Vector-Add)

# ## Learning Objectives
#
# * Explain how the __oneAPI__ programming model can solve the challenges of programming in a heterogeneous world 
# * Use oneAPI projects to enable your workflows
# * Understand the __Data Parallel C++ (DPC++)__ language and programming model
# * Familiarization on the use Jupyter notebooks for training throughout the course
#

# ## oneAPI Programming Model Overview
# The __oneAPI__ programming model provides a comprehensive and unified portfolio of developer tools that can
# be used across hardware targets, including a range of performance libraries spanning several workload
# domains. The libraries include functions custom-coded for each target architecture so the same
# function call delivers optimized performance across supported architectures. __DPC++__ is based on
# industry standards and open specifications to encourage ecosystem collaboration and innovation.
#
# ### oneAPI Distribution
# Intel&reg; oneAPI toolkits are available via multiple distribution channels:
# * Local product installation: install the oneAPI toolkits from the __Intel® Developer Zone__.
# * Install from containers or repositories: install the oneAPI toolkits from one of several supported
# containers or repositories.
# * Pre-installed in the __Intel® DevCloud__: a free development sandbox for access to the latest Intel® SVMS hardware and select oneAPI toolkits. 

# ## Programming Challenges for Multiple architectures
# Currently in the data centric space there is growth in specialized workloads. Each kind of data centric hardware typically needs to be programmed using different languages and libraries as there is no common programming language or APIs, this requires maintaining separate code bases. Developers have to learn a whole set of different tools as there is inconsistent tool support across platforms. Developing software for each hardware platform requires a separate investment, with little ability to reuse that work to target a different architecture. You will also have to consider the requirement of the diverse set of data-centric hardware.
#
# <img src="Assets/oneapi1.png">
#

# ## Introducing oneAPI
# __oneAPI__ is a solution to deliver unified programming model to __simplify development__ across diverse architectures. It includes a unified and simplified language and libraries for expressing __parallelism__ and delivers uncompromised native high-level language performance across a range of hardware including __CPUs, GPUs, FPGAs__. oneAPI initiative is based on __industry standards and open specifications__ and is interoperable with existing HPC programming models.
#
# <img src="Assets/oneapi2.png">
#

# ***
# # Simple Exercise
# This exercise introduces DPC++ to the developer by way of a small simple code. In addition, it introduces the developer to the Jupyter notebook environment for editing and saving code; and for running and submitting programs to the Intel® DevCloud.
#
# ##  Editing the simple.cpp code
# The Jupyter cell below with the gray background can be edited in-place and saved.
#
# The first line of the cell contains the command **%%writefile 'simple.cpp'** This tells the input cell to save the contents of the cell into a file named 'simple.cpp' in your current directory (usually your home directory). As you edit the cell and run it in the Jupyter notebook, it will save your changes into that file.
#
# The code below is some simple DPC++ code to get you started in the DevCloud environment. Simply inspect the code - there are no modifications necessary. Run the first cell to create the file, then run the cell below it to compile and execute the code.
# 1. Inspect the code cell below, then click run ▶ to save the code to a file
# 2. Run ▶ the cell in the __Build and Run__ section below the code snippet to compile and execute the code in the saved file

# %%writefile lab/simple.cpp
//==============================================================
// Copyright © 2020 Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================
#include <CL/sycl.hpp>
using namespace sycl;
static const int N = 16;
int main(){
  //# define queue which has default device associated for offload
  queue q;
  std::cout << "Device: " << q.get_device().get_info<info::device::name>() << "\n";

  //# Unified Shared Memory Allocation enables data access on host and device
  int *data = malloc_shared<int>(N, q);

  //# Initialization
  for(int i=0; i<N; i++) data[i] = i;

  //# Offload parallel computation to device
  q.parallel_for(range<1>(N), [=] (id<1> i){
    data[i] *= 2;
  }).wait();

  //# Print Output
  for(int i=0; i<N; i++) std::cout << data[i] << "\n";

  free(data, q);
  return 0;
}

# ### Build and Run
# Select the cell below and click Run ▶ to compile and execute the code above:

# ! chmod 755 q; chmod 755 run_simple.sh;if [ -x "$(command -v qsub)" ]; then ./q run_simple.sh; else ./run_simple.sh; fi

# _If the Jupyter cells are not responsive or if they error out when you compile the code samples, please restart the Jupyter Kernel: 
# "Kernel->Restart Kernel and Clear All Outputs" and compile the code samples again_

# ## SYCL
# __SYCL__ (pronounced ‘sickle’) represents an industry standardization effort that includes
# support for data-parallel programming for C++. It is summarized as “C++ Single-source
# Heterogeneous Programming for OpenCL.” The SYCL standard, like OpenCL*, is managed
# by the __Khronos Group*__.
#
# SYCL is a cross-platform abstraction layer that builds on OpenCL. It enables code
# for heterogeneous processors to be written in a “single source” style using C++. This is not
# only useful to the programmers, but it also gives a compiler the ability to analyze and
# optimize across the entire program regardless of the device on which the code is to be run.
#
# Unlike OpenCL, SYCL includes templates and lambda functions to enable higher-level application software to be cleanly coded with optimized acceleration of kernel code.
# Developers program at a higher level than OpenCL but always have access to lower-level code through seamless integration with OpenCL, as well as C/C++ libraries.

# ## What is Data Parallel C++
# __oneAPI__ programs are written in __Data Parallel C++ (DPC++)__. It takes advantage of modern C++ productivity benefits and familiar constructs, and incorporates the __SYCL*__ standard for data parallelism and heterogeneous programming. DPC++ is a __single source__ language where host code and __heterogeneous accelerator kernels__ can be mixed in same source files. A DPC++ program is invoked on the host computer and offloads the computation to an accelerator. Programmers use familiar C++ and library constructs with added functionliaties like a __queue__ for work targeting, __buffer__ for data management, and __parallel_for__ for parallelism to direct which parts of the computation and data should be offloaded.

# ## DPC++ extends SYCL 1.2.1
# DPC++ programs __enhance productivity__. Simple things should be simple to express and lower verbosity and programmer burden. They also __enhance performance__ by giving programmers control over program execution and by enabling hardware-specific features. It is a fast-moving open collaboration feeding into the __SYCL* standard__, and is an __open source__ implementation with the goal of upstreaming LLVM and DPC++ extensions to become core __SYCL*__, or __Khronos*__ extensions.

# ## HPC Single Node Workflow with oneAPI 
# Accelerated code can be written in either a kernel (DPC++) or __directive based style__. Developers can use the __Intel® DPC++ Compatibility tool__ to perform a one-time migration from __CUDA__ to __Data Parallel C++__. Existing __Fortran__ applications can use a __directive style based on OpenMP__. Existing __C++__ applications can choose either the __Kernel style__ or the __directive based style option__ and existing __OpenCL__ applications can remain in the OpenCL language or migrate to Data Parallel C++.
#
# __Intel® Advisor__ is recommended to  __Optimize__ the design for __vectorization and memory__ (CPU and GPU) and __Identify__ loops that are candidates for __offload__ and project the __performance on target accelerators.__
#
# The figure below shows the recommended approach of different starting points for HPC developers:
#
#
# <img src="Assets/workflow.png">
#

# ## oneAPI Programming models

# ### Platform Model
#
# The platform model for oneAPI is based upon the SYCL* platform model. It specifies a host controlling one or more devices. A host is the computer, typically a CPU-based system executing the primary portion of a program, specifically the application scope and the command group scope. 
#
# The host coordinates and controls the compute work that is performed on the devices. A device is an accelerator, a specialized component containing compute resources that can quickly execute a subset of operations typically more efficiently than the CPUs in the system. Each device contains one or more compute units that can execute several operations in parallel. Each compute unit contains one or more processing elements that serve as the individual engine for computation.
#
# The following figure provides a visual depiction of the relationships in the platform model. One host communicates with one or more devices. Each device can contain one or more compute units. Each compute unit can contain one or more processing elements. In this example, the CPU in a desktop computer is the host and it can also be made available as a device in a platform configuration.
#
# <img src="Assets/plat30.png">
#
#

# ### Execution Model
#
# The execution model is based upon the SYCL* execution model. It defines and specifies how code, termed kernels, execute on the devices and interact with the controlling host.
# The host execution model coordinates execution and data management between the host and devices via command groups. The command groups, which are groupings of commands like kernel invocation and accessors, are submitted to queues for execution.
#
# Accessors, which are formally part of the memory model, also communicate ordering requirements of execution. A program employing the execution model declares and instantiates queues. Queues can execute with an in-order or out-of-order policy controllable by the program. In-order execution is an Intel extension.
#
# The device execution model specifies how computation is accomplished on the accelerator. Compute ranging from small one-dimensional data to large multidimensional data sets are allocated across a hierarchy of ND-ranges, work-groups, sub-groups (Intel extension), and work-items, which are all specified when the work is submitted to the command queue.
#
# It is important to note that the actual kernel code represents the work that is executed for one work-item. The code outside of the kernel controls just how much parallelism is executed; the amount and distribution of the work is controlled by specification of the sizes of the ND-range and work-group.
#
#
# The following figure depicts the relationship between an ND-range, work-group, sub-group, and work-item. The total amount of work is specified by the ND-range size. The grouping of the work is specified by the work-group size. The example shows the ND-range size of X * Y * Z, work-group size of X’ * Y’ * Z’, and subgroup size of X’. Therefore, there are X * Y * Z work-items. There are (X * Y * Z) / (X’ * Y’ * Z’) work-groups and (X * Y * Z) / X’ subgroups.
#
# <img src="Assets/kernel30.png">
#
#

# ### Memory Model
#
# The memory model for oneAPI is based upon the SYCL* memory model. It defines how the host and devices interact with memory. It coordinates the allocation and management of memory between the host and devices. The memory model is an abstraction that aims to generalize across and be adaptable to the different possible host and device configurations.
#
# In this model, memory resides upon and is owned by either the host or the device and is specified by declaring a memory object. There are two different types of memory objects, buffers and images. Interaction of these memory objects between the host and device is accomplished via an accessor, which communicates the desired location of access, such as host or device, and the particular mode of access, such as read or write.
#
# Consider a case where memory is allocated on the host through a traditional malloc call. Once the memory is allocated on the host, a buffer object is created, which enables the host allocated memory to be communicated to the device. The buffer class communicates the type and number of items of that type to be communicated to the device for computation. Once a buffer is created on the host, the type of access allowed on the device is communicated via an accessor object, which specifies the type of access to the buffer.
#
# <img src="Assets/memory.png">

# ### Kernel Programming Model
# The kernel programming model for oneAPI is based upon the SYCL* kernel programming model. It enables explicit parallelism between the host and device. The parallelism is explicit in the sense that the programmer determines what code executes on the host and device; it is not automatic. The kernel code executes on the accelerator. 
#
# Programs employing the oneAPI programming model support single source, meaning the host code and device code can be in the same source file. However, there are differences between the source code accepted in the host code and the device code with respect to language conformance and language features. 
#
# The SYCL Specification defines in detail the required language features for host code and device code. The following is a summary that is specific to the oneAPI product.

# ## How to Compile & Run DPC++ program

# The three main steps of compiling and running a DPC++ program are:
# 1. Initialize environment variables
# 2. Compile the DPC++ source code
# 3. Run the application
#  
# #### Compiling and Running on Intel&reg; DevCloud:
#  
# For this training, we have written a script (q) to aid developers in developing projects on DevCloud. This script submits the `run.sh` script to a gpu node on DevCloud for execution, waits for the job to complete and prints out the output/errors. We will be using this command to run on DevCloud: `./q run.sh`
#
#
#
# #### Compiling and Running on a Local System:
#
# If you have installed the Intel&reg; oneAPI Base Toolkit on your local system, you can use the commands below to compile and run a DPC++ program:
#
#     source /opt/intel/inteloneapi/setvars.sh
#
#     dpcpp simple.cpp -o simple
#
#     ./simple
#     
# _Note: run.sh script is a combination of the three steps listec above._

# # Lab Exercise: Simple Vector Increment TO Vector Add
# ### Code Walkthrough
#
# __DPC++ programs__ are standard C++. The program is invoked on the __host__ computer, and offloads computation to the __accelerator__. You will use DPC++’s __queue, buffer, device, and kernel abstractions__ to direct which parts of the computation and data should be offloaded.
#
# The DPC++ compiler and the oneAPI libraries automate the tedious and error-prone aspects of compute and data offload, but still allow you to control how computation and data are distributed for best performance. The compiler knows how to generate code for both the host and the accelerator, how to launch computation on the accelerator, and how to move data back and forth. 
#
# In the program below you will use a data parallel algorithm with DPC++ to leverage the computational power in __heterogenous computers__. The DPC++ platform model includes a host computer and a device. The host offloads computation to the device, which could be a __GPU, FPGA, or a multi-core CPU__.
#
# As a first step in a DPC++ program, create a __queue__. Offload computation to a __device__ by submitting tasks to a queue. You can choose CPU, GPU, FPGA, and other devices through the __selector__. This program uses the default q here, which means the DPC++ runtime selects the most capable device available at runtime by using the default selector. You will learn more about devices, device selectors, and the concepts of buffers, accessors and kernels in the upcoming modules, but here is a simple DPC++ program to get you started.
#
# Device and host can either share physical __memory__ or have distinct memories. When the memories are distinct, offloading computation requires __copying data between host and device__. DPC++ does not require you to manage the data copies. By creating __Buffers and Accessors__, DPC++ ensures that the data is available to host and device without any effort on your part. DPC++ also allows you explicit control over data movement to achieve best peformance.
#
# In a DPC++ program, we define a __kernel__, which is applied to every point in an index space. For simple programs like this one, the index space maps directly to the elements of the array. The kernel is encapsulated in a __C++ lambda function__. The lambda function is passed a point in the index space as an array of coordinates. For this simple program, the index space coordinate is the same as the array index. The __parallel_for__ in the below program applies the lambda to the index space. The index space is defined in the first argument of the parallel_for as a 1 dimensional __range from 0 to N-1__.
#
# The __parallel_for__ is nested inside another lamba function, which is passed as an argument in the below program where we __submit to the queue__. The DPC++ runtime invokes the lambda when the accelerator connected to the queue is ready. The handler argument to the lambda allows operations inside the lambda to define the __data and dependences__ with other computation that may be executed on host or devices. You will see more of this in later modules.
#
# Finally, the program does a __q.wait()__ on the queue. The earlier submit operation queues up an operation to be performed at a later time and immmediately returns. If the host wants to see the result of the computation, it must wait for the work to complete with a wait. Sometimes the device will encounter an error. The q.wait_and_throw() is a way for the host to capture and handle the error that has happened on the device.
#
# ### Lab Exercise
# Vector increment is the “hello world” of data parallel computing. A vector is an array of data elements, and the program below performs the same computation on each element of the vector by adding 1. The code below shows Simple Vector Increment DPC++ code. You will change the program to create a new vector, then add the elements in the new vector to the existing vector using DPC++.
#
# 1. Select the code cell below, __follow the STEPS 1 to 6__ in the code comments to change from vector-increment to vector-add and click run ▶ to save the code to a file.
# 2. Next run ▶ the cell in the __Build and Run__ section below the code to compile and execute the code.

# +
# %%writefile lab/simple-vector-incr.cpp
//==============================================================
// Copyright © 2020 Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================

#include <CL/sycl.hpp>
using namespace sycl;
//N is set as 2 as this is just for demonstration purposes. Even if you make N bigger than 2 the program still
//counts N as only 2 as the first 2 elements are only initialized here and the rest all becomes zero.
static const size_t N = 2;

// ############################################################
// work

void work(queue &q) {
  std::cout << "Device : "
            << q.get_device().get_info<info::device::name>()
            << "\n";
  // ### Step 1 - Inspect
  // The code presents one input buffer (vector1) for which Sycl buffer memory
  // is allocated. The associated with vector1_accessor set to read/write gets
  // the contents of the buffer.
  int vector1[N] = {10, 10};
  auto R = range(N);
  
  std::cout << "Input  : " << vector1[0] << ", " << vector1[1] << "\n";

  // ### Step 2 - Add another input vector - vector2
  // Uncomment the following line to add input vector2
  //int vector2[N] = {20, 20};

  // ### Step 3 - Print out for vector2
  // Uncomment the following line
  //std::cout << "Input  : " << vector2[0] << ", " << vector2[1] << "\n";
  buffer vector1_buffer(vector1,R);

  // ### Step 4 - Add another Sycl buffer - vector2_buffer
  // Uncomment the following line
  //buffer vector2_buffer(vector2,R);
  q.submit([&](handler &h) {
    accessor vector1_accessor (vector1_buffer,h);

    // Step 5 - add an accessor for vector2_buffer
    // Uncomment the following line to add an accessor for vector 2
    //accessor vector2_accessor (vector2_buffer,h,read_only);

    h.parallel_for<class test>(range<1>(N), [=](id<1> index) {
      // ### Step 6 - Replace the existing vector1_accessor to accumulate
      // vector2_accessor 
      // Comment the following line
      vector1_accessor[index] += 1;

      // Uncomment the following line
      //vector1_accessor[index] += vector2_accessor[index];
    });
  });
  q.wait();
  host_accessor h_a(vector1_buffer,read_only);
  std::cout << "Output : " << vector1[0] << ", " << vector1[1] << "\n";
}

// ############################################################
// entry point for the program

int main() {  
  try {
    queue q;
    work(q);
  } catch (exception e) {
    std::cerr << "Exception: " << e.what() << "\n";
    std::terminate();
  } catch (...) {
    std::cerr << "Unknown exception" << "\n";
    std::terminate();
  }
}
# -

# ### Build and Run
# Select the cell below and click Run ▶ to compile and execute the code above:

# ! chmod 755 q; chmod 755 run_simple-vector-incr.sh; if [ -x "$(command -v qsub)" ]; then ./q run_simple-vector-incr.sh; else ./run_simple-vector-incr.sh; fi

# _If the Jupyter cells are not responsive or if they error out when you compile the code samples, please restart the Jupyter Kernel: 
# "Kernel->Restart Kernel and Clear All Outputs" and compile the code samples again_

# #### Solution
# - [simple-vector-add.cpp](src/simple-vector-add.cpp)

# # Summary
# In this module you will have learned the following:
# * How oneAPI solves the challenges of programming in a heterogeneous world 
# * Take advantage of oneAPI solutions to enable your workflows
# * Use the Intel® DevCloud to test-drive oneAPI tools and libraries
# * Basics of the DPC++ language and programming model
# * Become familiarized with the use of Juypter notebooks by editing of source code in context.
#

# <html><body><span style="color:green"><h1>Survey</h1></span></body></html>
#
# [Tell us how we did in this module with a short survey. We will use your feedback to improve the quality and impact of these learning materials. Thanks!](https://intel.az1.qualtrics.com/jfe/form/SV_6m4G7BXPNSS7FBz)
#
#

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
          # !rsync -a --size-only /data/oneapi_workshop/oneAPI_Essentials/01_oneAPI_Intro/ ~/oneAPI_Essentials/01_oneAPI_Intro
          print('Notebook reset -- now click reload on browser.')
# linking button and function together using a button's method
button.on_click(on_button_clicked)
# displaying button and its output together
widgets.VBox([button,out])

# ## Resources
#
# Check out these related resources
#
# #### Intel® oneAPI Toolkit documentation
# * [Intel® oneAPI main page](https://software.intel.com/oneapi "oneAPI main page")
# * [Intel® oneAPI programming guide](https://software.intel.com/sites/default/files/oneAPIProgrammingGuide_3.pdf "oneAPI programming guide")
# * [Intel® DevCloud Signup](https://software.intel.com/en-us/devcloud/oneapi "Intel DevCloud")  Sign up here if you do not have an account.
# * [Intel® DevCloud Connect](https://devcloud.intel.com/datacenter/connect)  Login to the DevCloud here.
# * [Get Started with oneAPI for Linux*](https://software.intel.com/en-us/get-started-with-intel-oneapi-linux)
# * [Get Started with oneAPI for Windows*](https://software.intel.com/en-us/get-started-with-intel-oneapi-windows)
# * [Intel® oneAPI Code Samples](https://software.intel.com/en-us/articles/code-samples-for-intel-oneapibeta-toolkits)
# * [oneAPI Specification elements](https://www.oneapi.com/spec/)
#
# #### SYCL 
# * [SYCL* Specification (for version 1.2.1)](https://www.khronos.org/registry/SYCL/specs/sycl-1.2.1.pdf)
#
# #### Modern C++
# * [CPPReference](https://en.cppreference.com/w/)
# * [CPlusPlus](http://www.cplusplus.com/)
#
# ***
