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

# # Intel® Distribution for GDB*

# In this notebook, we will cover using the Intel® Distribution for GDB* to debug oneAPI applications on the GPU.
#
# ##### Sections
# - [Intel Distribution for GDB Overview](#Intel-Distribution-for-GDB-Overview)
# - [How does the Intel Distribution for GDB debug GPUs?](#How-does-Intel-Distribution-for-GDB-debug-GPUs?)
# - [GDB Commands](#GDB-Commands)
# - [Debug an Application](#Debug-an-Application)
# - [Multi-Device Debugging](#Multi-Device-Debugging)
#
# Note: Unlike other modules in the oneAPI Essentials series, this notebook is designed for the DevCloud and cannot be run in a local environment. This is because when GDB pauses the GPU execution, display rendering is also interrupted.

# ## Learning Objectives
# The goal of this notebook is to show how the Intel® Distribution for GDB* can help you debug GPU kernels. At the end of module, you will be able to:
# <ul>
#     <li>Run the Intel Distribution for GDB.</li>
#     <li>Understand inferiors, threads, and SIMD lanes as shown in GDB.</li>
#     <li>Use different methods to examine local variables for different threads and lanes.</li>
# </ul>

# ## Intel Distribution for GDB Overview
#
# Intel® Distribution for GDB* (*gdb-oneapi* executable) is part of the Intel® oneAPI Base Toolkit. It can be used to debug oneAPI applications written in several different languages targeting various different accelerator devices.
#
# <img src="assets/gdb_overview.jpg">
#
# ### Major Features
# * Multi-target: The debugger can orchestrate multiple targets for different architectures. This feature allows you to debug the "host" portion and the "kernel" of a DPC++ program in the same GDB* session.
# * Auto-attach: The debugger automatically creates an inferior that attaches itself to the Intel® Graphics Technology target to be able to receive events and control the GPU for debugging.
# * Thread and SIMD lanes: The debugger displays SIMD lane information for the GPU threads on the command line interface. You can switch among active threads and lanes.
# * Support for debugging a kernel offloaded to a CPU, GPU, or FPGA-emulation device.
#
#
# ## How does the Intel Distribution for GDB debug GPUs?
#
#
# ### Compilation and Execution for Debug
# When debugging oneAPI applications with gdb-oneapi, debug information for the GPU needs to be generated and embedded in the appplication. The compilation and execution process looks like the following.
#
# <img src="assets/gpu_debug.jpg">
#
# 1. Source code is compiled. Host code is compiled normally while kernel code is compiled with debug info into SPIR-V intermediate representation format embedded in the host binary.
#     * Use -g (generate debug info) and -O0 (disable optimization) compiler options to debug source.
#     * May use -O2 to debug optimized code at assembly level.
#     * Use same optimization level when linking, if compiling and linking separately.
#     * Ahead-of-time (AOT) compilation also works with GPU debug and can be utilize to avoid JIT compilation everytime application is run.
# 2. Launch appliction with `gdb-oneapi`
#     * `gdb-oneapi <your_app>`
# 3. Application runtime compiles SPIR-V and debug info into ELF and DWARF formats.
# 4. GPU kernel code is executed and debugged.
#
# ### Inferiors for GPUs
#
# GDB creates objects called *inferior*s to represent the state of each program execution. An inferior usually corresponds to a debugee process. For oneAPI applications, GDB will create one inferior for the native host target and additional inferiors for each GPU or GPU tile. When a GPU application is debugged, the debugger, by default, automatically launches a `gdbserver-gt` process to listen to GPU debug events. The `gdbserver-gt` target is then added to the debugger as an inferior.
#
# <img src="assets/gdb_gpu_inferior.jpg">
#
# To see information about the inferiors while debugging. Use the `info inferiors` GDB command.
#
# ### Debugging Threaded GPU SIMD Code
#
# GPU kernel code is written for a single work-item. When executing, the code is implicitly threaded and widened to vectors of work-items. In the Intel Distribution for GDB, variable locations are expressed as functions of the SIMD lane. The lane field is added to the thread representation in the form of `<inferior>.<thread>:<lane>`.
#
# Users can use the `info threads` command to see information about the various active threads. The `thread` command can be used to switching among active threads and SIMD lanes. The `thread apply <thread>:<lane> <cmd>` command can be used to apply the specified command to the specified lanes.
#
# SIMD Lanes Support:
#
#     * Only enabled SIMD lanes are displayed
#     * SIMD width is not fixed
#     * User can switch between enabled SIMD lanes
#     * After a stop, GDB switches to an enabled SIMD lane

# ## GDB Commands
#
# The following table lists some common GDB commands. If a command has special functionality for GPU debugging, description will be shown in orange. You may also consult the [Intel Distribution for GDB Reference Sheet](https://software.intel.com/content/www/us/en/develop/download/gdb-reference-sheet.html).
#
# | Command | Description |
# | ---: | :--- |
# | help \<cmd> | Print help information. |
# | run [arg1, ... argN] | Start the program, optionally with arguments. |
# | break \<file>:\<line> | Define a breakpoint at a specified line. |
# | info break | Show defined breakpoints. |
# | delete \<N> | Remove Nth breakpoint. |
# | step / next | Single-step a source line, stepping into / over function calls. |
# | info args/locals | Show the arguments/local variables of the current function. |
# | print \<exp> | Print value of expression. |
# | x/\<format> \<addr> | Examine the memory at \<addr>. |
# | up, down | Go one level up/down the function call stack |
# | disassemble | Disassemble the current function.  <font color='orange'> If inside a GPU kernel, GPU instructions will be shown. </font> |
# | backtrace | Shown the function call stack. |
# | info inferiors | Display information about the inferiors. <font color='orange'> GPU debugging will display additional inferior(s) (gdbserver-gt). </font> |
# | info threads \<thread> | Display information about threads, including their <font color='orange'> active SIMD lanes. </font> |
# | thread \<thread>:\<lane> | Switch context to the <font color='orange'> SIMD lane of the specified thread. <font> |
# | thread apply \<thread>:\<lane> \<cmd> | Apply \<cmd> to specified lane of the thread. |
# | set scheduler-locking on/step/off | Lock the thread scheduler. Keep other threads stopped while current thread is stepping (step) or resumed (on) to avoid interference. Default (off). |
# | set nonstop on/off | Enable/disable nonstop mode. Set before program starts. <br> (off) : When a thread stops, all other threads stop. Default. <br> (on) : When a thread stops, other threads keep running. |
# | print/t $emask | Inspect the execution mask to show active SIMD lanes. | 

# ## Debug an Application
# The kernel we're going to debug is a simple array transform function where the kernel adds 100 to even elements of the array and sets the odd elements to be -1. Below is the kernel code, the entire source code is [here](src/array-transform.cpp).
# ``` cpp
# 54        h.parallel_for(data_range, [=](id<1> index) {
# 55            size_t id0 = GetDim(index, 0);
# 56            int element = in[index]; // breakpoint-here
# 57            int result = element + 50;
# 58            if (id0 % 2 == 0) {
# 59                result = result + 50; // then-branch
# 60            } else {
# 61                result = -1; // else-branch
# 62            }
# 63            out[index] = result;
# 64        });
# ```
#
# ### Compile the Code
# Execute the following cell to compile the code. Notice the compiler options used to disable optimization and enable debug information.

# ! dpcpp -O0 -g src/array-transform.cpp -o bin/array-transform

# ### Create a debug script
# To debug on the GPU, we're going to write the GDB debug commands to a file and then submit the execution of the debugger to a node with GPUs.
#
# In our first script, we'll get take a look at how inferiors, threads, and SIMD lanes are represented. Our debug script will perform the following tasks. 
# 1. Set a temporary breakpoint in the DPCPP kernel at line 59.
# 2. Run the application in the debugger.
# 3. Display information about the active inferiors once the breakpoint is encountered.
# 4. Display information about the active threads and SIMD lanes.
# 5. Display the execution mask showing which SIMD lanes are active.
# 6. Remove breakpoint.
# 7. Continue running.
#
# Execute the following cell to write the debug commands to file.

# +
# %%writefile lab/array-transform.gdb
#Set Breakpoint in the Kernel
echo ================= (1) tbreak 59 ===============\n
tbreak 59

# Run the application on the GPU
echo ================= (2) run gpu ===============\n
run gpu

echo ================= (3)  info inferiors ============\n
info inferiors

echo ================= (4) info threads ============\n
info threads

# Show execution mask that show active SIMD lanes.
echo ================= (5) print/t $emask ============\n
print/t $emask

echo ================= (6) c ==========================\n
c 
# -

# ### Start the Debugger
# The [run_debug.sh](run_debug.sh) script runs the *gdb-oneapi* executable with our debug script on the compiled application.
#
# Execute the following cell to submit the debug job to a node with a GPU.

# ! chmod 755 q; chmod 755 run_debug.sh; if [ -x "$(command -v qsub)" ]; then ./q run_debug.sh; else ./run_debug.sh; fi

# #### Explanation of Output
# 1. You should see breakpoint 1 created at line 59.
# 2. Application is run with the *gpu* argument to execute on the GPU device. Program should stop at the kernel breakpoint.
# 3. With context now automatically switched to the device. The *info inferiors* command will show the active GDB inferior(s). Here, you should see two, one corresponds to the host portion, another, the active one, for gdbserver-gt which is debugging the GPU kernel. 
# 4. The *info threads* command allows you to examine the active threads and SIMD lanes. There should be 8 threads active. Notice that only even SIMD lanes are active, this is because only the even work-items encounter the breakpoint at line 59.
# 5. Printing the $emask execution mask also shows the even lanes being active.
# 6. Continue running the program.

# ## Debug the Application Again
#
# Now, we will debug the application again. This time, we'll switch threads, use the scheduler-locking feature, and print local variables.
# Run the following cell to write new GDB commands to array-transform.gdb.

# +
# %%writefile lab/array-transform.gdb
#Set Breakpoint in the Kernel
echo ================= (1) break 59 ===============\n
break 59
echo ================= (2) break 61 ===============\n
break 61

# Run the application on the GPU
echo ================= (3) run gpu ===============\n
run gpu

# Keep other threads stopped while current thread is stepped
echo ================= (4) set scheduler-locking step ===============\n
set scheduler-locking step 

echo ================= (5) next ===============\n
next 

echo ================= (6) info threads 2.* ===============\n
info threads 2.*

echo ================= (7) Print element ============\n
print element

# Switch thread
echo ================= (8) thread 2.1:5 =======================\n
thread 2.1:4

echo ================= (9) Print element ============\n
print element

echo ================= (10) thread apply 2.1:* print element =======================\n
thread apply 2.1:* print element

# Inspect vector of a local variable, 8 elements, integer word
echo ================= (11) x/8dw &result =======================\n
x /8dw &result

echo ================= (12) d 1 =======================\n
d 1
echo ================= (13) d 2 =======================\n
d 2
echo ================= (14) c ==========================\n
c 
# -

# ### Start Debugger Again To Examine Variables, Memories
# Run the following cell to run the debugger for the second time.

# ! chmod 755 q; chmod 755 run_debug.sh; if [ -x "$(command -v qsub)" ]; then ./q run_debug.sh; else ./run_debug.sh; fi

# ### Explanation of Output
# 1. Set Breakpoint at line 59 for the even lanes.
# 2. Set Breakpoint at line 61 for the odd lanes.
# 3. Start the application, it will stop at breakpoint 2. At this point all the threads should be stopped and active for the odd SIMD lanes.
# 4. Set schedule-locking so that when the current thread is stepped all other threads remain stopped.
# 5. Step the current thread (thread 2.1), breakpoint at line 59 is encountered only for current thread.
# 6. Show the threads and where each thread is stopped. Notice the current thread is stopped and active at the even SIMD lanes.
# 7. Print local variable element.
# 8. Switch to a different lane.
# 9. Print local variable element again, this time you should see a different value.
# 10. Use thread apply to print element for all lanes of the 2.1 thread.
# 11. Print vectorized result.
# 12. Delete breakpoint.
# 13. Delete breakpoint.
# 14. Run until the end.

# ## Multi-Device Debugging
#
# The Intel Distribution for GDB can debug applications that offload a kernel to multiple GPU devices. Each GPU device appear as a separate inferior within the debugger. Users can switch to the context of a thread that corresponds to a particular GPU or CPU using the `inferior <id>` command. Threads of the GPUs can be independently resumed and the thread state can be individually examined.

# ## References
# * [Intel Distribution for GDB Landing Page](https://software.intel.com/content/www/us/en/develop/tools/oneapi/components/distribution-for-gdb.html)
# * [Intel Distribution for GDB Release Notes](https://software.intel.com/content/www/us/en/develop/articles/gdb-release-notes.html)
# * [Intel Distribution for GDB Reference Sheet](https://software.intel.com/content/www/us/en/develop/download/gdb-reference-sheet.html)

# ## Summary
#
#   * Used Intel Distribution for GDB to debug a GPU application.
#   * Used various-GPU related GDB commands.
