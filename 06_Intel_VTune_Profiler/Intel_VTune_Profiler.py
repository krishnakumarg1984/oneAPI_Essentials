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

# # VTune™ Profiling on Intel® DevCloud

# ##### Sections
# - [What is VTune™ Profiler?](#What-is-VTune™-Profiler?)
# - [VTune™ Command-line Options](#VTune™-Command-line-Options)
# - _Lab Exercise:_ [VTune™ gpu_hotspots profiling with iso3dfd sample](#Lab-Exercise:-VTune™-Profiling)

# ## Learning Objectives

# - Profile a DPC++ application using the VTune™ profiling tool on Intel® DevCloud
# - Understand the basics of VTune™ command line options for collecting data and generating reports

# ## What is VTune™ Profiler?

# VTune™ allows DPC++ Profiling capabilities so you can tune for CPU, GPU, and FPGA.
#
# ![VTune UI](vtuneui.png)
#
# __Analyze Data Parallell C++__ :
# See the lines of DPC++ that consume the most time
#
# __Tune for CPU, GPU & FPGA__ :
# Optimize for any supported hardware accelerator
#
# __Optimize Offload__ :
# Tune OpenMP offload performance
#
# __Wide Range of Performance Profiles__ :
# CPU, GPU, FPGA, threading, memory, cache, storage…
#
# __Most Popular Languages__ :
# DPC++, C, C++, Fortran*, Python*, Go*, Java*, or a mix
#

# ## VTune™ Command-line Options

# ### Run and collect VTune™ data
# ```vtune -collect gpu_hotspots -result-dir vtune_data a.out```
#
# Various types of profiling data can be collected like `hotspots`, `memory-consumption`, `memory-access`, `threading`…
#
# Use the command line help to find out more:
#
# ```vtune --help -collect```
#
# ### Generate html report for collected VTune™ data:
# ```vtune -report summary -result-dir vtune_data -format html -report-output $(pwd)/summary.html```
#
# Various types of report can be generated like `summary`, `top-down`, `callstacks`…
#
# Use the command line help to find out more:
#
# ```vtune --help -report```
#

# ## When to use VTune™ Command line

# VTune™ Command-line is useful when on __Intel® DevCloud__ or you only have __SSH__ access to development system.
#
# However, it is recommended to install the __full VTune™ version__ on a local system and use the __UI rich experience__ of VTune Profiling Tool.
#
# ![VTune UI](vtuneui.png)
#

# ## Lab Exercise: VTune™ Profiling
# - Build, run, collect VTune™ data and display VTune summary when running on gpu and cpu.

# ### Test Application: DPC++ implementation of iso3dfd

# DPC++ implementation of iso3dfd will be used to collect VTune™ data and analyze the generated result. Below are source code to iso3dfd application:
# - [iso3dfd.cpp](src/iso3dfd.cpp)
# - [iso3dfd_kernels.cpp](src/iso3dfd_kernels.cpp)
#

# ### Build and Run

# +
# %%writefile run_iso3dfd.sh
# #!/bin/bash
source /opt/intel/inteloneapi/setvars.sh > /dev/null 2>&1

dpcpp src/iso3dfd.cpp src/utils.cpp src/iso3dfd_kernels.cpp -o iso3dfd

./iso3dfd 256 256 256 8 8 8 20 sycl gpu



# -

# **STEP 1:** Build and Run the iso3dfd app by running ▶ the command below:

# ! chmod 755 q; chmod 755 run_iso3dfd.sh; if [ -x "$(command -v qsub)" ]; then ./q run_iso3dfd.sh; else ./run_iso3dfd.sh; fi

# _If the Jupyter cells are not responsive or if they error out when you compile the code samples, please restart the Jupyter Kernel: 
# "Kernel->Restart Kernel and Clear All Outputs" and compile the code samples again_.

# ## Analyze performance with VTune™

# Use VTune™ command line to analyze performace on GPU vs CPU and display the summary

# ### VTune™ Command Line for collecting and reporting

# +
# %%writefile vtune_collect.sh
# #!/bin/bash
source /opt/intel/inteloneapi/setvars.sh
/bin/echo "##" $(whoami) is compiling DPCPP_Essentials Module6 -- Intel Vtune profiler - 1 of 1 Vtune_Profiler
#vtune
#type=hotspots
#type=memory-consumption
#type=uarch-exploration
#type=memory-access
#type=threading
#type=hpc-performance
#type=system-overview
#type=graphics-rendering
#type=io
#type=fpga-interaction
#type=gpu-offload
type=gpu-hotspots
#type=throttling
#type=platform-profiler
#type=cpugpu-concurrency
#type=tsx-exploration
#type=tsx-hotspots
#type=sgx-hotspots

# rm -r vtune_data

# echo "Vtune Collect $type"
vtune -collect $type -result-dir vtune_data $(pwd)/iso3dfd 256 256 256 8 8 8 20 sycl gpu

# echo "Vtune Summary Report"
vtune -report summary -result-dir vtune_data -format html -report-output $(pwd)/summary.html


# -

# ### Run VTune™ to Collect Hotspots and Generate Report

# **STEP 2:** Collect VTune™ data and generate report by running ▶ the command below:

# ! chmod 755 vtune_collect.sh; if [ -x "$(command -v qsub)" ]; then ./q vtune_collect.sh; else ./vtune_collect.sh; fi

# _If the Jupyter cells are not responsive or if they error out when you compile the code samples, please restart the Jupyter Kernel: 
# "Kernel->Restart Kernel and Clear All Outputs" and compile the code samples again_.

# ## Display VTune™ Summary

# Display VTune™ summary report generated in html format

# ### Display VTune™ Report for GPU

# **STEP 3:** Display VTune™ summary report by running ▶ the command below 

from IPython.display import IFrame
IFrame(src='summary.html', width=960, height=600)

# ## Summary

# VTune™ command line is useful for quick analysis of DPC++ application to get performance metric and tune applications.

# <html><body><span style="color:green"><h1>Survey</h1></span></body></html>
#
# [We would appreciate any feedback you’d care to give, so that we can improve the overall training quality and experience. Thanks! ](https://intel.az1.qualtrics.com/jfe/form/SV_5jyMumvDk1YKDeR)

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
          # !rsync -a --size-only /data/oneapi_workshop/oneAPI_Essentials/06_Intel_VTune_Profiler/ ~/oneAPI_Essentials/06_Intel_VTune_Profiler
          print('Notebook reset -- now click reload on browser.')
# linking button and function together using a button's method
button.on_click(on_button_clicked)
# displaying button and its output together
widgets.VBox([button,out])
