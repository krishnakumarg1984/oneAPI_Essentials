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
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# # Data Parallel C++ Essentials Modules

# + [markdown] tags=[]
# The concepts build on top of each other introducing and reinforcing the concepts of Data Parallel C++.
#
# ## Module 0 - [Introduction to Jupyter Notebook (Optional) ](00_Introduction_to_Jupyter/Introduction_to_Jupyter.ipynb)
# `Optional` This module explains how to use Jupyter Notebook which is used in all of the modules to edit and run coding excecises, this can be skipped if you are already familiar with using Jupyter Notebooks.
#
# ## Module 1 - [Introduction to oneAPI and DPC++ ](01_oneAPI_Intro/oneAPI_Intro.ipynb)
# These initial hands-on exercises introduce you to DPC++ and the goal of oneAPI. In addition, it familiarizes you with the use of Jupyter notebooks as a front-end for all training exercises. This workshop is designed to be used on the DevCloud and includes details on how to submit batch jobs on DevCloud environment.
#
# ## Module 2 - [DPC++ Program Structure](02_DPCPP_Program_Structure/DPCPP_Program_Structure.ipynb)
# These hands-on exercises present six basic DPC++ programs that illustrate the elements of a DPC++ application. You can modify the source code in some of the exercises to become more familiar with DPC++ programming concepts.
#
# ## Module 3 - [DPC++ Unified Shared Memory](03_DPCPP_Unified_Shared_Memory/Unified_Shared_Memory.ipynb)
# These hands-on exercises show how to implement Unified Shared Memory (USM) in DPC++ code, as well as demonstrate how to solve for data dependencies for in-order and out-of-order queues.
#
# ## Module 4 - [DPC++ Subgroups](04_DPCPP_Sub_Groups/Sub_Groups.ipynb)
# These hands-on exercises demonstrate the enhanced features that DPC++ brings to sub-groups. The code samples demonstrate how to implement a query for sub-group info, sub-group collectives, and sub-group shuffle operations.
#
# ## Module 5 - [Demonstration of Intel® Advisor](05_Intel_Advisor/offload_advisor.ipynb)
# This set of hand-on exercises demonstrates various aspects of Intel® Advisor. The first uses Intel® Advisor to show performance offload opportunities found in a sample application, and then additional command-line options for getting offload advisor results. The second, [roofline analysis](05_Intel_Advisor/roofline_analysis.ipynb), gives an example of roofline analysis and command line options for getting advisor results. For both exercises, the results are rendered inside of the notebook. These notebooks are meant for exploration and familiarization, and do not require any cdoe modification.
#
# ## Module 6 - [Intel® Vtune™ Profiler on DevCloud](06_Intel_VTune_Profiler/Intel_VTune_Profiler.ipynb)
# This hands-on exercise demonstrates using Intel® Vtune™ Profiler on the command-line to collect and analyze gpu_hotspots. You will learn how to collect performance metrics and explore the results with the HTML output rendered inside of the notebook.  This module meant for exploration and familiarization, and does not require any code modification.
#
# ## Module 7 - [Intel® oneAPI DPC++ Library](07_DPCPP_Library/oneDPL_Introduction.ipynb)
# This hands-on exercise demonstrates using Intel® oneAPI DPC++ Library (oneDPL) for heterogeneous computing. You will learn how to use various Parallel STL algorithms for heterogeneous computing and also look at gamma-correction sample code that uses oneDPL.
#
# ## Module 8 - [DPC++ Reductions](08_DPCPP_Reduction/Reductions.ipynb)
# This hands-on exercise demonstrates various ways to optimizes reduction operations using DPC++. You will learn how to use ND-Range kernels to parallelize reductions on accelerators and also learn how to optimize reductions using new reduction extensions in DPC++.
#
# ## Module 9 - [Explore Buffers and Accessors in depth](09_DPCPP_Buffers_And_Accessors_Indepth/DPCPP_Buffers_accessors.ipynb)
# This hands-on exercise demonstrates various ways to create Buffers in DPC++. You will learn how to use sub-buffers, buffer properties and when to use_host_ptr, set_final_data and set_write_data. You will also learn accessors, host_accessors and its usecases.
#
# ## Module 10 - [SYCL Task Scheduling and Data Dependences](10_DPCPP_Graphs_Scheduling_Data_management/DPCPP_Task_Scheduling_Data_dependency.ipynb)
# This hands-on exercise demonstrates how to utilize USM and Buffers and accessors to apply Memory management and take control over data movement implicitly and explicitly, utilize different types of data dependences that are important for ensuring execution of graph scheduling. You will also learn to select the correct modes of dependences in Graphs scheduling.
#
# ## Module 11 - [Intel® Distribution for GDB on DevCloud](11_Intel_Distribution_for_GDB/gdb_oneapi.ipynb)
# This hands-on exercise demonstrates how to use the Intel® Distribution for GDB to debug kernels running on GPUs.
