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

# # Intel® Advisor - Roofline Analysis

# This sections demonstrates how to collect and generate a roofline report using Intel Advisor.
#
# ##### Sections
# - [What is the Roofline Model?](#What-is-the-Roofline-Model?)
# - _Analysis:_ [Roofline Analysis Report](#Roofline-Analysis-Report)
# - [Finding Effective Optimization Strategies](#Finding-Effective-Optimization-Strategies)
# - [Command Line Options for GPU Roofline Analysis](#Command-Line-Options-for-GPU-Roofline-Analysis)
# - [Using Roofline Analysis on Intel GPU](#Using-Roofline-Analysis-on-Intel-GPU)

# ## Learning Objectives
# - Explain how Intel® Advisor performs GPU Roofline Analysis.
# - Run the GPU Roofline Analysis using command line syntax.
# - Use GPU Roofline Analysis to identify effective optimization strategies.
#

# ## What is the Roofline Model?

# A Roofline chart is a visual representation of application performance in relation to hardware limitations, including memory bandwidth and computational peaks.  Intel Advisor includes an automated Roofline tool that measures and plots the chart on its own, so all you need to do is read it.
#
# The chart can be used to identify not only where bottlenecks exist, but what’s likely causing them, and which ones will provide the most speedup if optimized.
#

# ## Requirements for a Roofline Model on a GPU
#
# In order to generate a roofline analysis report ,application must be at least partially running on a GPU, Gen9 or Gen11 integrated graphics and the Offload must be implemented with OpenMP, SYCL, DPC++, or OpenCL and a recent version of Intel® Advisor 
#
# Generating a Roofline Model on GPU generates a multi-level roofline where a single loop generates several dots and each dot can be compared to its own memory (GTI/L3/DRAM/SLM)
#

# ## Gen9 Memory Hierarchy
#
# ![image](assets/gen9.png)

# ## Roofline Analysis Report

# Let's run a roofline report -- this is another <b>live</b> report that is interactive.

# [Intel Advisor Roofline report](assets/roofline.html)

import os
os.system('/bin/echo $(whoami) is running DPCPP_Essentials Module5 -- Roofline_Analysis - 2 of 2 roofline.html')
from IPython.display import IFrame
IFrame(src='assets/roofline.html', width=1024, height=769)

# # Finding Effective Optimization Strategies
#  Here are the GPU Roofline Performance Insights, it highlights poor performing loops and shows performance ‘headroom’  for each loop which can be improved and which are worth improving. The report shows likely causes of bottlenecks where it can be Memory bound vs. compute bound. It also suggests next optimization steps
#
#   
#   <img src="assets/r1.png">
#  
#

# ### Running the Survey

# The Survey is usually the first analysis you want to run with Intel® Advisor. The survey is mainly used to time your application as well as the different loops and functions. 

# ### Running the trip count

# The second step is to run the trip count analysis. This step uses instrumentation to count how many iterations you are running in each loops. Adding the option -flop will also provide the precise number of operations executed in each of your code sections.

# ## Advisor Command-Line for generating "roofline" on the CLI

# * Clone official GitHubb samples repository
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
# * To run the GPU Roofline analysis in the Intel® Advisor CLI:
#   Run the Survey analysis with the --enable-gpu-profiling option    
#      ``advixe-cl –collect=survey --enable-gpu-profiling --project-dir=./adv -- ./matrix.dpcpp``
#         
# * Run the Trip Counts and FLOP analysis with --enable-gpu-profiling option:
#
#     ``advixe-cl -–collect=tripcounts --stacks --flop --enable-gpu-profiling --project-dir=./adv -- ./matrix.dpcpp``
#
# *Generate a GPU Roofline report:
#     ``advixe-cl --report=roofline --gpu  --project-dir=./adv`` 
#     
# * Open the generated roofline.html in a web browser to visualize GPU performance.

# +
# %%writefile advisor_roofline.sh
# #!/bin/bash

advixe-cl –collect=survey --enable-gpu-profiling --project-dir=./adv -- ./matrix.dpcpp

advixe-cl -–collect=tripcounts --stacks --flop --enable-gpu-profiling --project-dir=./adv -- ./matrix.dpcpp

advixe-cl --report=roofline --gpu  --project-dir=./adv


# -

# ## Using Roofline Analysis on Intel GPU
# You can see how close you are to the system maximums. The roofline indicates maximum room for improvement
#
# <img src="assets/r2.png">
#

# ## Showing Dots for all Memory Sub-systems
#
# ![alt text](assets/roofline3.png "More info.")

# ## Add Labels
#
# ![alt text](assets/roofline4.png "Labeling.")

# ## Clean the View
#
# ![alt text](assets/roofline5.png "Clean View.")

# ## Show the Guidance
#
# ![alt text](assets/roofline6.png "Guidance.")

# ## Summary
#
#   * We ran a roofline report.
#   * Explored the features of the roofline report and learned how to interpret the report.
#   * Examined the information to determine where speedup opportunites exist.

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
