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

# # Intel® Advisor - Offload Advisor

# These sections demonstrate how to collect and generate a roofline report using Intel® Advisor, below we will examine our "offload" report.
#
# ##### Sections
# - [What is Offload Advisor?](#What-is-Offload-Advisor?)
# - _Analysis:_ [Offload Advisor Analysis](#Offload-Advisor-Analysis)
# - [Analysis of Top Offload Regions](#Analysis-of-Top-Offload-Regions)
# - [What Kernels Should Not Be Offloaded?](#What-Kernels-Should-Not-Be-Offloaded?)
# - [Command line options](#Command-line-options)
#
#

# ## Learning Objectives
# The goal of this notebook is to show how Intel® Advisor can help deciding what part of the code should or should not be offloaded on the GPU. At the end of this, you will be able:
# <ul>
#     <li>To run Offload Advisor and generate a HTML report</li>
#     <li>To read and understand the metrics in the report</li>
#     <li>To get a performance estimation of your application on the target hardware</li>
#     <li>To decide which loops are good candidate for offload</li>
# </ul>

# ## What is Offload Advisor?

# Offload Advisor allows you to collect performance predictor data in addition to the profiling capabilities of Intel® Advisor. View output files containing metrics and performance data such as total speedup, fraction of code accelerated, number of loops and functions offloaded, and a call tree showing offloadable and accelerated regions.

# ## Offload Advisor Analysis

# The below HTML report is <span style="color:blue"><b>live</b></span>, click navigation to see output.

# [Intel Advisor Offload report](assets/offload.html)

# ### View the Report
# Select the cell below and click run ▶ to view the analysis.

import os
os.system('/bin/echo $(whoami) is running DPCPP_Essentials Module5 -- Intel Advisor - 1 of 2 offload.html')
from IPython.display import IFrame
IFrame(src='assets/offload.html', width=1024, height=1280)

# ## Using Intel® Advisor to increase performance
# __Intel® Advisor__ is recommended to  __Optimize__ the design for __vectorization and memory__ (CPU and GPU) and __Identify__ loops that are candidates for __offload__ and project the __performance on target accelerators.__
#
# __offload Advisor__ can help determine what kernels should be offloaded and can predict the speedup that can be expected.
#
# Developers can use the __Intel® DPC++ Compatibility tool__ to perform a one-time migration from __CUDA__ to __Data Parallel C++__. Existing __Fortran__ applications can use a __directive style based on OpenMP__. Existing __C++__ applications can choose either the __Kernel style__ or the __directive based style option__.
#
# Once you wirte the DPC++ code,  __GPU roofline analyis__ helps to develop an optimization strategy and see potential bottlenecks relative to target maximums.
#
# Finally the GPU analysis using VTune can help optimize for the target.
#
# <img src="assets/a1.png">

# ### Intel® Advisor - Offload Advisor: Find code that can be profitably offloaded
#
# From the below fugure we can clearly observe that the the workload was accelerated by 3.5x. You can see in program metrics that the original workload ran in 18.51s and the accelerated workload ran in 5.45s
#
#
#
# <img src="assets/a4.png">

# ### Offload Advisor: Will Offload Increase Performance?
#
# From the below figure we can clearly observe the good candidates for offloading and the bad candidates to offload. You can also observe what your workload is bounded by.
#
#
# <img src="assets/a5.png">

# ## Analysis of Top Offload Regions
#
# Provides a detailed description of each loop interesting for offload. You can view the Timings (total time, time on the accelerator, speedup), the Offload metrics like the offload taxe and the data transfers, Memory traffic (DRAM, L3, L2, L1) and the trip count. It also highlighst which part of the code should run on the accelerator.
#   
#   <img src="assets/a6.png">
#

# ## What Kernels Should Not Be Offloaded?
#
# Below explains why Intel Advisor does not recommend a given loop for offload. The possible reason can be dependency issues, that loops are not profitable, or the total time is too small.
#   
#   <img src="assets/a7.png">
#   
#

# ## Compare Acceleration on Different GPUs
#
# Below compares acceleration on Gen9 and Gen11. You can observe from the below picture that its not efficient to offload on Gen 9
# whereas in Gen11 there is one offload with 98% of code accelerated and by 1.6x.
#
#   
#   <img src="assets/a8.png">
#   
#

# ## What Is the Workload Bounded By?
#
# The performance will ultimately have an upper bound based on your hardware’s limitations. There are several limitations that Offload Advisor can indicate but they generally  come down to compute, memory and data transfer. Knowing what your application is bounded by is critical to developing an optimization strategy. In the below example 95% of workload bounded by L3 bandwidth but you may have several bottlenecks.
#
#   
#   <img src="assets/a9.png">
#   
#

# ## Program Tree
#
# The program tree offers another view of the proportion of code that can be offloaded to the accelerator
#
# ![image](assets/programtree.png)

# ## Command line options
#
#
#
# The application runs on a CPU and is actually need not be threaded. For Intel® Offload Advisor, it doesn't matter if your code is already threaded. Advisor will run several analyses on your application to extract several metric such as the number of operations, the number of memory transfers, data dependencies and many more.
# Remember that our goal here is to decide if some of our loops are good candidates for offload. In this section, we will generate the report assuming that we want to offload our computations on a Gen Graphic (gen9) which is the hardware available on DevCloud.
# Keep in mind that if you want Advisor to extract as much information as possible, you need to compile your application with debug information (-g with intel compilers).
#
# The easiest way to run Offload Advisor is to use the batch mode that consists in running 2 scripts available is the folder $APM ($APM is available when Advisor is sourced).
# <ul>
#     <li>collect.py: Used to collect data such as timing, flops, tripcounts and many more</li>
#     <li>analyze.py: Creating the report</li>
# </ul>
#
# To be more specific, collect.py runs the following analyses:
# <ul>
#     <li>survey: Timing your application functions and loops, reading compiler diagnostics</li>
#     <li>tripcount: With flops and cache simulation to count the number of iterations in the loops as well as the number of operations and memory transfers</li>
#     <li>dependency: Check if you have data dependency in your loops, preventing it to be good candidates for offloading or vectorization</li>
# </ul>
#
# Offload Advisor is currently run from the command-line as below. Once the run is complete you can view the generated report.html.
#
# * Clone official GitHub samples repository
#      git clone https://github.com/oneapi-src/oneAPI-samples.git
#         
# * Go into Project directory to the matrix multiply advisor sample 
#
#     ``cd oneAPI-samples/Tools/Advisor/matrix_multiply_advisor/``
#     
# * Build the application and generate the matrix multiplication binary
#
#     ``cmake .``    
#     ``make``
#
# ```
#
# advixe-python $APM/collect.py advisor_project --config gen9 -- ./matrix.dpcpp
# advixe-python $APM/analyze.py advisor_project --config gen9 --out-dir ./analyze
#
# ```
#

# +
# %%writefile advisor_offload.sh
# #!/bin/bash

advixe-python $APM/collect.py advisor_project --config gen9 -- ./matrix.dpcpp
advixe-python $APM/analyze.py advisor_project --config gen9 --out-dir ./analyze

# -

# ### Generating the HTML report

# The last step is to generate our HTML report for offloading on gen9. This report will show us:
# <ul>
#     <li>What is the expected speedup on Gen9</li>
#     <li>What will most likely be our bottleneck on Gen9</li>
#     <li>What are the good candidates for offload</li>
#     <li>What are the loops that should not be offloaded</li>
# </ul>

# ## Offload Advisor Output Overview
#
# <span style="color:blue">report.html</span>: Main report in HTML format
#
# <span style="color:blue">report.csv</span> and <span style="color:blue">whole_app_metric.csv</span>: Comma-separated CSV files
#
# <span style="color:blue">program_tree.dot:</span> A graphical representation of the call tree showing the offloadable and accelerated regions
#
# <span style="color:blue">program_tree.pdf:</span> A graphical representation of the call tree generated if the DOT\(GraphViz*) utility is installed and a 1:1 conversion from the <span style="color:blue">program_tree.dot</span> file
#
# <span style="color:blue">JSON</span> and <span style="color:blue">LOG</span> files that contain data used to generate the HTML report and logs, primarily used for debugging and reporting bugs and issues
#
#

# ## Summary
#
#   * Ran the Offload Advisor report.
#   * Analyzed various outputs.
#   * Learned about additional command line options and how to speed up collection time.

# <html><body><span style="color:green"><h1>Survey</h1></span></body></html>
#
# [We would appreciate any feedback you’d care to give, so that we can improve the overall training quality and experience. Thanks! ](https://intel.az1.qualtrics.com/jfe/form/SV_0OZVTLvFGI2e0Id)

# ## Continue to Roofline Analysis
# [Roofline Analysis](roofline_analysis.ipynb)

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
          # !rsync -a --size-only /data/oneapi_workshop/oneAPI_Essentials/05_Intel_Advisor/ ~/oneAPI_Essentials/05_Intel_Advisor
          print('Notebook reset -- now click reload on browser.')
# linking button and function together using a button's method
button.on_click(on_button_clicked)
# displaying button and its output together
widgets.VBox([button,out])
