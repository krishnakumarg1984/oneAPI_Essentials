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

# # SYCL Task Scheduling and Data Dependences

# ##### Sections
# - [Buffers and Accessors](#Buffers-and-Accessors)
# - [Memory Management](#Memory-Management) 
# - [Explicit Data Movement](#Explicit-Data-Movement)
# - [Implicit data movement](#Implicit-data-movement) 
# - [What is USM?](#What-is-Unified-Shared-Memory?)
# - [Types of USM](#Types-of-USM)
#     - _Code:_ [USM Explicit data Movement](#USM-Explicit-data-Movement)
#     - _Code:_ [USM Implicit data Movement](#USM-Implicit-data-Movement)
# - [Accessors](#Accessors)
#     - [Access modes](#Access-modes)    
#     - [Graph Scheduling](#Execution-Graph-Scheduling)
#         - _Code:_ [RAW - Read after Write](#RAW-Read-after-Write)
#         - _Code:_ [WAR WAW- Write after Read and Write after Write](#WAR-WAW-Write-after-Read-and-Write-after-Write)
#     - _Code:_ [Implicit dependency with Accessors](#Implicit-dependency-with-Accessors) 
# - [Graphs and Dependencies](#Graphs-and-Dependencies)
# - [Graphs in DPC++](#Graphs-in-DPC++)
# - [Dependency in Graphs](#Dependency-in-Linear-dependency-chain-graphs-and-y-pattern-Graphs)
# - [In-Order Queues](#In-Order-Queues)
#     - _Code:_ [Linear dependence chain using in-order queues](#Linear-dependence-chain-using-in-order-queues)
#     - _Code:_ [Y Pattern using in-order queues](#Y-Pattern-using-in-order-queues)
# - [Event-based dependencies](#Event-based-dependencies) 
#     - _Code:_ [Linear dependence chain using events](#Linear-dependence-chain-using-events)
#     - _Code:_ [Y Pattern using events](#Y-Pattern-using-events)
# - _Code:_ [Linear dependence chain using Buffers and Accessors](#Linear-dependence-chain-using-Buffers-and-Accessors)
# - _Code:_ [Y Pattern using Buffers and Accessors](#Y-Pattern-using-Buffers-and-Accessors)
#

# ## Learning Objectives
# * Utilize USM and Buffers and Accessors to apply Memory management and take control over data movement   implicitly and explicitly
# * Utilize different types of data dependences that are important for ensuring execution of graph scheduling
# * Select the correct modes of dependences in Graphs scheduling.

# ## Buffers and Accessors
# __Buffers__ are high level abstarction for data and these are accessbile either on the host machine or on the devices. Buffers encapsulate data in a SYCL application across both devices and host. __Accessors__ is the mechanism to access buffer data. Buffers are 1, 2 or 3 dimensional data.

# ## Memory Management
# Managing multiple memories can be accomplished, broadly, in two ways: 
# * Explicitly by the programmer
# * Implicitly by the runtime.
#
# Each method has its advantages and drawbacks, and programmers may choose one or the other depending on circumstances or personal preference.
#
#

# ### Explicit Data Movement
# In a DPC++ program one option for managing multiple memories is for the programmer to explicitly copy data between host and the device and once the computation is done it needs to be copied back to the host from the device. This can be done explicitly by the programmer. 
#
# Also, once we offload computation to a device by submitting tasks to a queue and the kernel computes new results,
# the data needs to be copied back to the host program. One of the main advantages of explicit transfer is that the programmer has full control over when data is transferred between the device and the host and back to host from the device,  and this is important and can be essential to obtaining the best performance on some hardware.
#  
# The disadvantage of explicit data movement is that transferring explicitly by the programmer can be tedious process and error prone. Transferring incorrect data or transferring the data back to host at incorrect time can lead to incorrect results. 
#
# Getting all of the data movement correct up front can be a time-consuming task.
#
# ### Implicit data movement
# The alternative to explicit data movement is implicit data movement. This is controlled by the runtime or driver and here the runtime is responsible for ensuring that data is transferred to the appropriate memory before it is used. 
#
# The advantage of implicit data movement is that it requires less effort on the programmer’s	part and  all the heavy lifting is done by the runtime. This also reduces the opportunity to introduce errors into the program since the runtime will automatically identify both when data transfers must be performed and how much data must be transferred.
#
# The drawback of implicit data movement is that the programmer has less or no control over the behavior of the runtime’s implicit mechanisms. The runtime will provide functional correctness but may not move data in an optimal fashion that could have a negative impact on program performance.
#
# ### Selecting the right strategy: explicit or implicit
# A programmer might choose to begin using implicit data movement to simplify porting an application to a new device. As we 
# begin tuning the application for performance, we might start replacing implicit data movement with explicit in performance-critical parts of the code.
#

# ## What is Unified Shared Memory?

# Unified Shared Memory (USM) is a DPC++ tool for data management. USM is a
# __pointer-based approach__ that should be familiar to C and C++ programmers who use malloc
# or new to allocate data. USM __simplifies development__ for the programmer when __porting existing
# C/C++ code__ to DPC++.

# ### USM Explicit data Movement
#
# The DPC++ code below shows an implementation of USM using <code>malloc_device</code>, in which data movement between host and device should be done explicitly by developer using <code>memcpy</code>. This allows developers to have more __controlled movement of data__ between host and device.
#
# The DPC++ code below demonstrates USM Explicit Data Movement: Inspect code, there are no modifications necessary:
#
# 1. Inspect the code cell below and click run ▶ to save the code to file.
# 2. Next run ▶ the cell in the __Build and Run__ section below the code to compile and execute the code.

# +
# %%writefile lab/USM_explicit.cpp

// Copyright (C) 2020 Intel Corporation

// SPDX-License-Identifier: MIT

#include <CL/sycl.hpp>
#include<array>
using namespace sycl;
constexpr int N = 42;

int main() {
  queue Q;

  std::array<int,N> host_array;
  int *device_array = malloc_device<int>(N, Q);

  for (int i = 0; i < N; i++)
    host_array[i] = N;

  // Submit the queue
  Q.submit([&](handler &h) {
      // copy hostArray to deviceArray
      h.memcpy(device_array, &host_array[0], N * sizeof(int));
    });
  Q.wait();

  Q.submit([&](handler &h) {
      h.parallel_for(N, [=](id<1> i) { device_array[i]++; });
    });
  Q.wait();

  Q.submit([&](handler &h) {
      // copy deviceArray back to hostArray
      h.memcpy(&host_array[0], device_array, N * sizeof(int));
    });
  Q.wait();

  free(device_array, Q);
  return 0;
}
# -

# #### Build and Run
# Select the cell below and click run ▶ to compile and execute the code:

# ! chmod 755 q; chmod 755 run_usm_explicit.sh; if [ -x "$(command -v qsub)" ]; then ./q run_usm_explicit.sh; else ./run_usm_explicit.sh; fi

# ### USM Implicit data Movement
#
# The DPC++ code below shows an implementation of USM using <code>malloc_shared</code>, in which data movement happens implicitly between host and device. Useful to __get functional quickly with minimum amount of code__ and developers will not having worry about moving memory between host and device.

# +
# %%writefile lab/USM_implicit.cpp

// Copyright (C) 2020 Intel Corporation

// SPDX-License-Identifier: MIT

#include <CL/sycl.hpp>
using namespace sycl;
constexpr int N = 42;

int main() {
  queue Q;
  int *host_array = malloc_host<int>(N, Q);
  int *shared_array = malloc_shared<int>(N, Q);

  for (int i = 0; i < N; i++) {
    // Initialize hostArray on host
    host_array[i] = i;
  }

  // Submit the queue
  Q.submit([&](handler &h) {
      h.parallel_for(N, [=](id<1> i) {
          // access sharedArray and hostArray on device
          shared_array[i] = host_array[i] + 1;
        });
    });
  Q.wait();

  for (int i = 0; i < N; i++) {
    // access sharedArray on host
    host_array[i] = shared_array[i];
  }

  free(shared_array, Q);
  free(host_array, Q);
  return 0;
}

# -

# #### Build and Run
# Select the cell below and click run ▶ to compile and execute the code:

# ! chmod 755 q; chmod 755 run_usm_implicit.sh; if [ -x "$(command -v qsub)" ]; then ./q run_usm_implicit.sh; else ./run_usm_implicit.sh; fi

# # Accessors

# Data represented by a buffer cannot be directly accessed through the buffer object. Instead, we must create accessor objects that allow us to safely access a buffer’s data. Accessors inform the runtime where and how we want to access data, allowing the runtime to ensure that the right data is in the right place at the right time and the kernels dont run until the data is available.
#
# ### Access modes
#
# When creating an accessor, we must inform the runtime how we are going to use it by	specifying an access mode as described in the below table.
# Access modes are how the runtime is able to perform implicit data movement.
# When the accessor is created with __access::mode::read_write__ we intend to both read and write to the buffer through the accessor. 
#
# `read_only` tells the runtime that the data needs to be available on the device before this kernel can begin executing. Similarly, __`access::mode::write`__ lets the runtime know that we will modify the contents of a buffer and may need to copy the results back after computation has ended.
#
# The runtime uses accessors to order the use of data, but it can also use this data to optimize scheduling of kernels and data movement.
#
# | Access Mode | Description |
# |:---|:---|
# | __read__ | Read only Access|
# | __write__ | Write-only access. Previous contents not discarded |
# | __read_write__ | Read and Write access |
# | __atomic__ |Read and write atomic access |
#
#
#

# ###  Execution Graph Scheduling
# Execution  graphs are the mechanism that we use to achieve proper sequencing of kernels, and data movement in an application. Dependences between kernels are fundamentally based on what data a kernel accesses. A kernel needs to be certain that it reads the correct data before it can compute its output.
#
# There are three types of data dependences that are important for ensuring correct execution. 
#
# * Read-after-Write (RAW) : Occurs when one task needs to read data produced by a different task. This type of dependence describes the flow of data between two kernels.
# * Write-after-Read (WAR) : The second type of dependence happens when one task needs to update data after another task has read it.
# * Write-after-Write (WAW) : The final type of data dependence occurs when two tasks try to write the same data.

# #### RAW-Read after Write
# The DPC++ code below demonstrates creating accessors: Inspect code, there are no modifications necessary:
#
# 1. Inspect the code cell below and click run ▶ to save the code to file
# 2. Next run ▶ the cell in the __Build and Run__ section below the code to compile and execute the code.

# +
# %%writefile lab/accessors_RAW.cpp
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
  std::array<int,N> a, b, c;
  for (int i = 0; i < N; i++) {
    a[i] = b[i] = c[i] = 0;
  }

  queue Q;

  //Create Buffers
  buffer A{a};
  buffer B{b};
  buffer C{c};

  Q.submit([&](handler &h) {
      accessor accA(A, h, read_only);
      accessor accB(B, h, write_only);
      h.parallel_for( // computeB
        N,
        [=](id<1> i) { accB[i] = accA[i] + 1; });
    });

  Q.submit([&](handler &h) {
      accessor accA(A, h, read_only);
      h.parallel_for( // readA
        N,
        [=](id<1> i) {
          // Useful only as an example
          int data = accA[i];
        });
    });

  Q.submit([&](handler &h) {
      // RAW of buffer B
      accessor accB(B, h, read_only);
      accessor accC(C, h, write_only);
      h.parallel_for( // computeC
        N,
        [=](id<1> i) { accC[i] = accB[i] + 2; });
    });

  // read C on host
  host_accessor host_accC(C, read_only);
  for (int i = 0; i < N; i++) {
    std::cout << host_accC[i] << " ";
  }
  std::cout << "\n";
  return 0;
}
# -

# ### Build and Run
# Select the cell below and click run ▶ to compile and execute the code:

# ! chmod 755 q; chmod 755 run_accessor_RAW.sh;if [ -x "$(command -v qsub)" ]; then ./q run_accessor_RAW.sh; else ./run_accessor_RAW.sh; fi

# #### WAR WAW-Write after Read and Write after Write
#
# WAR happens when one task needs to update data after another task has read it and WAW occurs when two tasks try to write the same data.
#
# The DPC++ code below demonstrates creating accessors: Inspect code, there are no modifications necessary:
# 1. Inspect the code cell below and click run ▶ to save the code to file
# 2. Next run ▶ the cell in the __Build and Run__ section below the code to compile and execute the code.

# +
# %%writefile lab/accessors_WAR_WAW.cpp
//==============================================================
// Copyright © 2020 Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================


// Copyright (C) 2020 Intel Corporation

// SPDX-License-Identifier: MIT

#include <CL/sycl.hpp>
#include <array>
using namespace sycl;
constexpr int N = 42;

int main() {
  std::array<int,N> a, b;
  for (int i = 0; i < N; i++) {
    a[i] = b[i] = 0;
  }

  queue Q;
  buffer A{a};
  buffer B{b};

  Q.submit([&](handler &h) {
      accessor accA(A, h, read_only);
      accessor accB(B, h, write_only);
      h.parallel_for( // computeB
          N, [=](id<1> i) {
          accB[i] = accA[i] + 1;
          });
      });

  Q.submit([&](handler &h) {
      // WAR of buffer A
      accessor accA(A, h, write_only);
      h.parallel_for( // rewriteA
          N, [=](id<1> i) {
          accA[i] = 21 + 21;
          });
      });

  Q.submit([&](handler &h) {
      // WAW of buffer B
      accessor accB(B, h, write_only);
      h.parallel_for( // rewriteB
          N, [=](id<1> i) {
          accB[i] = 30 + 12;
          });
      });

  host_accessor host_accA(A, read_only);
  host_accessor host_accB(B, read_only);
  for (int i = 0; i < N; i++) {
    std::cout << host_accA[i] << " " << host_accB[i] << " ";
  }
  std::cout << "\n";
  return 0;
}
# -

# ### Build and Run
# Select the cell below and click run ▶ to compile and execute the code:

# ! chmod 755 q; chmod 755 run_accessor_WAW.sh;if [ -x "$(command -v qsub)" ]; then ./q run_accessor_WAW.sh; else ./run_accessor_WAW.sh; fi

# ## Graphs and Dependencies
#
# We discussed data management and ordering the uses of data and the abstraction behind graphs in DPC++: dependences. 
# Dependences between kernels are fundamentally based on what data a kernel accesses. A kernel needs to be certain that it reads the correct data before it can compute its output. 
#
# We described the three types of data dependences that are important for ensuring correct execution. The first, Read-after-Write (RAW), occurs when one task needs to read data produced by a different task. This type of dependence describes the flow of data between two kernels. The second type of dependence happens when one task needs to update data after another task has read it. We call that type of dependence a Write-afterRead (WAR) dependence. The final type of data dependence occurs when two tasks try to write the same data. This is known as a Write-after-Write (WAW) dependence.
#
# Data dependences are the building blocks we will use to build graphs. This set of dependences is all we need to express both simple linear chains of kernels and large, complex graphs with hundreds of kernels with elaborate dependences. No matter which types of graph a computation needs, DPC++ graphs ensure that a program will execute correctly based on the expressed dependences. However, it is up to the programmer to make sure that a graph correctly expresses all the dependences in a program.

# ## Graphs in DPC++
#
# A command group can contain three different things: an action, its dependences, and miscellaneous host code. Of these three things, the one that is always required is the action since without it, the command group really doesn’t do anything. Most command groups will also express dependences, but there are cases where they may not.
#
# Command groups are typically expressed as a C++ lambda expression passed to the submit method. Command groups can also be expressed through shortcut methods on queue objects that take a kernel and set of event-based dependences.
#
# There are two types of actions that may be performed by a command group: kernels and explicit memory operations. Kernels
# are defined through calls to a parallel_for or single_task method and express computations that we want to perform on our devices. Operations for explicit data movement are the second type of action. Examples from USM include memcpy, memset, and fill operations. Examples from buffers include copy, fill, and update_host.
#
#

# ## Dependency in Linear dependency chain graphs and y pattern Graphs
#
# The two patterns that are explained below are linear dependence chains where one task executes after another and a “Y” pattern where two independent tasks must execute before successive tasks. 
#
# In a  __linear dependence__ chain the first node represents the initialization of data, while the second node presents the
# reduction operation that will accumulate the data into a single result.
#
#    ![Linear Dependence](Assets/graphs_linear.png)
#
# In a __“Y” pattern__ we independently initialize two different pieces of data. After the data is initialized, an addition kernel
# will sum the two vectors together. Finally, the last node in the graph accumulates the result into a single value.
#
#    ![Y Pattern](Assets/graphs_y.png)
#
# In the below examples for each pattern we will see three different implementations.
# * In-order queues. 
# * Event-based dependences. 
# * Using buffers and accessors to express data dependences between command groups.

# ### In-Order Queues
# The other main component of a command group is the set of dependences that must be satisfied before the action defined by the group can execute. DPC++ allows these dependences to be specified in several ways. If a program uses in-order DPC++ queues, the in-order semantics of the queue specify implicit dependences between successively enqueued command groups. One task cannot execute until the previously submitted task has completed.

# ### Linear dependence chain using in-order queues
#
# In the below example the inorder queues already guarantee a sequential order of execution between command groups. The first kernel we submit initializes the elements of an array to 1. The next kernel then takes those elements and sums them together into the first element. 
# Since our queue is in order, we do not need to do anything else to express that the second kernel should not execute
# until the first kernel has completed. Finally, we wait for the queue to finish executing all its tasks, and we check that we obtained the expected result.

# The DPC++ code below demonstrates creating Linear dependence In-Order Queues: Inspect code, there are no modifications necessary:
# 1. Inspect the code cell below and click run ▶ to save the code to file
# 2. Next run ▶ the cell in the __Build and Run__ section below the code to compile and execute the code.

# +
# %%writefile lab/Linear_inorder_queues.cpp
//==============================================================
// Copyright © 2020 Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================

#include <CL/sycl.hpp>
using namespace sycl;
constexpr int N = 42;

int main() {
  queue Q{property::queue::in_order()};

  int *data = malloc_shared<int>(N, Q);

  Q.parallel_for(N, [=](id<1> i) { data[i] = 1; });

  Q.single_task([=]() {
      for (int i = 1; i < N; i++)
        data[0] += data[i];
    });
  Q.wait();

  assert(data[0] == N);
  return 0;
}
# -

# ### Build and Run
# Select the cell below and click run ▶ to compile and execute the code:

# ! chmod 755 q; chmod 755 run_linear_inorder.sh;if [ -x "$(command -v qsub)" ]; then ./q run_linear_inorder.sh; else ./run_linear_inorder.sh; fi

# ### Y Pattern using in-order queues
#
# In the below example we can see a “Y” pattern using in-order queues. In this example, we declare two arrays, data1 and data2. We then define two kernels that will each initialize one of the arrays. These kernels do not depend on each other, but because the queue is in order, the kernels must execute one after the other. 
#
# Note that you can swap the order of these two kernels in this example. After the second kernel has executed, the third kernel adds the elements of the second array to those of the first array. The final kernel sums up the elements of the first array
# to compute the same result we did in our examples for linear dependence chains. 
#
# This summation kernel depends on the previous kernel, but this linear chain is also captured by the in-order queue. Finally, we wait for all kernels to complete and validate that we successfully computed the final result.

# The DPC++ code below demonstrates creating Linear dependence In-Order Queues: Inspect code, there are no modifications necessary:
# 1. Inspect the code cell below and click run ▶ to save the code to file
# 2. Next run ▶ the cell in the __Build and Run__ section below the code to compile and execute the code.

# +
# %%writefile lab/y_pattern_inorder_queues.cpp
//==============================================================
// Copyright © 2020 Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================

#include <CL/sycl.hpp>
using namespace sycl;
constexpr int N = 42;

int main() {
  queue Q{property::queue::in_order()};
 
  int *data1 = malloc_shared<int>(N, Q);
  int *data2 = malloc_shared<int>(N, Q);

  Q.parallel_for(N, [=](id<1> i) { data1[i] = 1; });

  Q.parallel_for(N, [=](id<1> i) { data2[i] = 2; });

  Q.parallel_for(N, [=](id<1> i) { data1[i] += data2[i]; });

  Q.single_task([=]() {
      for (int i = 1; i < N; i++)
        data1[0] += data1[i];

      data1[0] /= 3;
    });
  Q.wait();

  assert(data1[0] == N);
  return 0;
}
# -

# ### Build and Run
# Select the cell below and click run ▶ to compile and execute the code:

# ! chmod 755 q; chmod 755 run_y_inorder.sh;if [ -x "$(command -v qsub)" ]; then ./q run_y_inorder.sh; else ./run_y_inorder.sh; fi

# ## Event-based dependencies
#
# Event-based dependences are another way to specify what must be complete before a command group may execute. These event-based
# dependences may be specified in two ways. The first way is used when a command group is specified as a lambda passed to a queue’s submit method. In this case, the programmer invokes the depends_on method of the command group handler object, passing either an event or vector of events as parameter. 
#
# The other way is used when a command group is created from the shortcut methods defined on the queue object. When the
# programmer directly invokes parallel_for or single_task on a queue, an event or vector of events may be passed as an extra parameter.

# ### Linear dependence chain using events
# In the below example we can see usage of  an out-of-order queue and event-based dependences. Here, we capture the event returned by the first call to parallel_for. The second kernel is then able to specify a dependence on that event and the kernel execution it represents by passing it as a parameter to depends_on.

#
# The DPC++ code below demonstrates creating In-Order Queues: Inspect code, there are no modifications necessary:
#
# 1. Inspect the code cell below and click run ▶ to save the code to file
# 2. Next run ▶ the cell in the __Build and Run__ section below the code to compile and execute the code.

# +
# %%writefile lab/linear_event_graphs.cpp
//==============================================================
// Copyright © 2020 Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================

#include <CL/sycl.hpp>
using namespace sycl;
constexpr int N = 42;

int main() {
  queue Q;

  int *data = malloc_shared<int>(N, Q);

  auto e = Q.parallel_for(N, [=](id<1> i) { data[i] = 1; });

  Q.submit([&](handler &h) {
      h.depends_on(e);
      h.single_task([=]() {
          for (int i = 1; i < N; i++)
            data[0] += data[i];
        });
    });
  Q.wait();

  assert(data[0] == N);
  return 0;
}
# -

# ### Build and Run
# Select the cell below and click run ▶ to compile and execute the code:

# ! chmod 755 q; chmod 755 run_linear_events.sh;if [ -x "$(command -v qsub)" ]; then ./q run_linear_events.sh; else ./run_linear_events.sh; fi

# ### Y Pattern using events
#
# Below is a  “Y” pattern example with out-of-order queues instead of in-order queues. Since the dependences are no longer implicit
# due to the order of the queue, we must explicitly specify the dependences between command groups using events. 
#
# We define two independent kernels that have no initial dependences. We represent these kernels by two events, e1 and e2. When we define our third kernel, we must specify that it depends on the first two kernels. We do this by saying that it depends on events e1 and e2 to complete before it may execute. 
#
# However, in this example, we use a shortcut form to specify these dependences instead of the handler’s depends_on method. Here, we
# pass the events as an extra parameter to parallel_for. Since we want to
# pass multiple events at once, we use the form that accepts a std::vector of events, as modern C++ simplifies this by automatically
# converting the expression {e1, e2} into the appropriate vector.
#

# The DPC++ code below demonstrates creating Linear dependence In-Order Queues: Inspect code, there are no modifications necessary:
# 1. Inspect the code cell below and click run ▶ to save the code to file
# 2. Next run ▶ the cell in the __Build and Run__ section below the code to compile and execute the code.

# +
# %%writefile lab/y_pattern_events.cpp
//==============================================================
// Copyright © 2020 Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================

#include <CL/sycl.hpp>
using namespace sycl;
constexpr int N = 42;

int main() {
  queue Q;
 
  int *data1 = malloc_shared<int>(N, Q);
  int *data2 = malloc_shared<int>(N, Q);

  auto e1 = Q.parallel_for(N, [=](id<1> i) { data1[i] = 1; });

  auto e2 = Q.parallel_for(N, [=](id<1> i) { data2[i] = 2; });

  auto e3 = Q.parallel_for(range{N}, {e1, e2},
                           [=](id<1> i) { data1[i] += data2[i]; });

  Q.single_task(e3, [=]() {
      for (int i = 1; i < N; i++)
        data1[0] += data1[i];

      data1[0] /= 3;
    });
  Q.wait();

  assert(data1[0] == N);
  return 0;
}
# -

# ### Build and Run
# Select the cell below and click run ▶ to compile and execute the code:

# ! chmod 755 q; chmod 755 run_y_events.sh;if [ -x "$(command -v qsub)" ]; then ./q run_y_events.sh; else ./run_y_events.sh; fi

# ### Linear dependence chain using Buffers and Accessors
# In the below example we show how to write linear dependence chain using buffers and accessors instead of USM pointers. Here we use an outof-order queue but use data dependences specified through accessors instead of event-based dependences to order the execution of the command groups.
#
# The second kernel reads the data produced by the first kernel, and the runtime can see this because we declare accessors based
# on the same underlying buffer object. Unlike the previous examples, we do not wait for the queue to finish executing all its tasks. Instead, we declare a host accessor that defines a data dependence between the output of the second kernel and our assertion that we computed the correct answer on the host.

# The DPC++ code below demonstrates creating Linear Dependence chain using Buffers and Accessors: Inspect code, there are no modifications necessary:
# 1. Inspect the code cell below and click run ▶ to save the code to file
# 2. Next run ▶ the cell in the __Build and Run__ section below the code to compile and execute the code.

# +
# %%writefile lab/linear_buffers_graphs.cpp
//==============================================================
// Copyright © 2020 Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================

#include <CL/sycl.hpp>
using namespace sycl;
constexpr int N = 42;

int main() {
  queue Q;

  buffer<int> data{range{N}};

  Q.submit([&](handler &h) {
      accessor a{data, h};
      h.parallel_for(N, [=](id<1> i) { a[i] = 1; });
    });

  Q.submit([&](handler &h) {
      accessor a{data, h};
      h.single_task([=]() {
          for (int i = 1; i < N; i++)
            a[0] += a[i];
        });
    });

  host_accessor h_a{data};
  assert(h_a[0] == N);
  return 0;
}
# -

# ### Build and Run
# Select the cell below and click run ▶ to compile and execute the code:

# ! chmod 755 q; chmod 755 run_linear_buffer.sh;if [ -x "$(command -v qsub)" ]; then ./q run_linear_buffer.sh; else ./run_linear_buffer.sh; fi

# ### Y Pattern using Buffers and Accessors
#
# Below is a  “Y” pattern example with buffers and Accessors. We replace USM pointers and events with buffers and accessors. This example represents the two arrays data1 and data2 as buffer objects. Our kernels no longer use the shortcut methods for defining kernels since we must associate accessors with a command group handler. 
#
# The third kernel must capture the dependence on the first two kernels. Here this is accomplished by declaring accessors for our buffers. Since we have previously declared accessors for these buffers, the runtime is able to properly order the execution of these kernels. 
#
# As we saw in our buffer and accessor example for linear dependence chains, our final kernel orders itself by updating the values produced in the third kernel. We retrieve the final value of our computation by declaring a host accessor that will wait for the final kernel to finish executing before moving the data back to the host where we can read it and assert we computed the correct result.
#
#

# The DPC++ code below demonstrates creating Linear dependence In-Order Queues: Inspect code, there are no modifications necessary:
# 1. Inspect the code cell below and click run ▶ to save the code to file
# 2. Next run ▶ the cell in the __Build and Run__ section below the code to compile and execute the code.

# +
# %%writefile lab/y_pattern_buffers.cpp
//==============================================================
// Copyright © 2020 Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================

#include <CL/sycl.hpp>
using namespace sycl;
constexpr int N = 42;

int main() {
  queue Q;
 
  buffer<int> data1{range{N}};
  buffer<int> data2{range{N}};

  Q.submit([&](handler &h) {
      accessor a{data1, h};
      h.parallel_for(N, [=](id<1> i) { a[i] = 1; });
    });

  Q.submit([&](handler &h) {
      accessor b{data2, h};
      h.parallel_for(N, [=](id<1> i) { b[i] = 2; });
    });

  Q.submit([&](handler &h) {
      accessor a{data1, h};
      accessor b{data2, h, read_only};
      h.parallel_for(N, [=](id<1> i) { a[i] += b[i]; });
    });

  Q.submit([&](handler &h) {
      accessor a{data1, h};
      h.single_task([=]() {
          for (int i = 1; i < N; i++)
            a[0] += a[i];

          a[0] /= 3;
        });
    });

  host_accessor h_a{data1};
  assert(h_a[0] == N);
  return 0;
}
# -

# ### Build and Run
# Select the cell below and click run ▶ to compile and execute the code:

# ! chmod 755 q; chmod 755 run_y_buffer.sh;if [ -x "$(command -v qsub)" ]; then ./q run_y_buffer.sh; else ./run_y_buffer.sh; fi

#
# # Summary
# In this module you learned:
# * How to utilize USM and Buffers and Accessors to apply Memory management and take control over data movement  implicitly and explicitly
# * How to utilize different types of data dependences that are important for ensuring execution of graph scheduling
# * Select the correct modes of dependences in Graphs scheduling.
#
