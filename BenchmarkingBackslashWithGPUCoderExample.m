%% Benchmark A\b by Using GPU Coder
% This example shows how to benchmark solving a linear system
% by generating GPU code.  Use matrix left division, also known as
% <docid:matlab_ref#btg5qam mldivide> or the backslash operator (\), to 
% solve the system of linear equations $A*x = b$ for $x$ (that is, 
% compute $x = A \backslash b$).
%

% Copyright 2017-2019 The MathWorks, Inc.

%% Prerequisites
% * CUDA(R) enabled NVIDIA(R) GPU with compute capability 3.5 or higher.
% * NVIDIA CUDA toolkit and driver.
% * Environment variables for the compilers and libraries. For information 
% on the supported versions of the compilers and libraries, see 
% <docid:gpucoder_gs#mw_aa8b0a39-45ea-4295-b244-52d6e6907bff
% Third-Party Products>. For setting up the environment variables, see 
% <docid:gpucoder_gs#mw_453fbbd7-fd08-44a8-9113-a132ed383275
% Environment Variables>.

%% Verify GPU Environment
% To verify that the compilers and libraries necessary for running this example
% are set up correctly, use the <docid:gpucoder_ref#mw_0957d820-192f-400a-8045-0bb746a75278 coder.checkGpuInstall>
% function.
envCfg = coder.gpuEnvConfig('host');
envCfg.BasicCodegen = 1;
envCfg.Quiet = 1;
coder.checkGpuInstall(envCfg);

%% Determine the Maximum Data Size
% 0Choose the appropriate matrix size for the computations by specifying 
% the amount of system memory in GB available to the CPU and the GPU.  The 
% default value is based only on the amount of memory available on the GPU. 
% You can specify a value that is appropriate for your system.
g = gpuDevice; 
maxMemory = 0.25*g.AvailableMemory/1024^3;

%% The Benchmarking Function
% This example benchmarks matrix left division (\) including the cost of
% transferring data between the CPU and GPU, to get a clear view of the
% total application time when using GPU Coder(TM). The application time 
% profiling must not include the time to create sample input data. The 
% |genData.m| function separates generation of test data from the 
% entry-point function that solves the linear system. 
type getData.m

%% The Backslash Entry-Point Function
% The |backslash.m| entry-point function encapsulates the (\) operation for 
% which you want to generate code.
type backslash.m

%% Generate the GPU Code
% Create a function to generate the GPU MEX function based on the
% particular input data size. 
type genGpuCode.m

%% Choose a Problem Size
% The performance of the parallel algorithms that solve a linear system 
% depends greatly on the matrix size. This example compares the performance 
% of the algorithm for different matrix sizes.

% Declare the matrix sizes to be a multiple of 1024.
sizeLimit = inf;
if ispc
    sizeLimit = double(intmax('int32'));
end
maxSizeSingle = min(floor(sqrt(maxMemory*1024^3/4)),floor(sqrt(sizeLimit/4)));
maxSizeDouble = min(floor(sqrt(maxMemory*1024^3/8)),floor(sqrt(sizeLimit/8)));
step = 1024;
if maxSizeDouble/step >= 10
    step = step*floor(maxSizeDouble/(5*step));
end
sizeSingle = 1024:step:maxSizeSingle;
sizeDouble = 1024:step:maxSizeDouble;
numReps = 5;

%% Compare Performance: Speedup
% Use the total elapsed time as a measure of performance because that
% enables you to compare the performance of the algorithm for different
% matrix sizes. Given a matrix size, the benchmarking function creates the
% matrix |A| and the right-side |b| once, and then solves |A\b| a few
% times to get an accurate measure of the time it takes. 
%
type benchFcnMat.m

%%
% Create a different function for GPU code execution that
% invokes the generated GPU MEX function.
type benchFcnGpu.m

%% Execute the Benchmarks
% When you execute the benchmarks, the computations can take a long time to 
% complete. Print some intermediate status information as you complete the
% benchmarking for each matrix size.  Encapsulate the loop over
% all the matrix sizes in a function to benchmark single- and double-precision 
% computations.
%
% Actual execution times can vary across different hardware configurations. 
% This benchmarking was done by using MATLAB R2020a on a machine with a 
% 6 core, 3.5GHz Intel(R) Xeon(R) CPU and an NVIDIA TITAN Xp GPU.
type executeBenchmarks.m

%%
% Execute the benchmarks in single and double precision.
[cpu, gpu] = executeBenchmarks('single', sizeSingle, numReps);
results.sizeSingle = sizeSingle;
results.timeSingleCPU = cpu;
results.timeSingleGPU = gpu;
[cpu, gpu] = executeBenchmarks('double', sizeDouble, numReps);
results.sizeDouble = sizeDouble;
results.timeDoubleCPU = cpu;
results.timeDoubleGPU = gpu;

%% Plot the Performance
% Plot the results and compare the performance on the CPU and
% the GPU for single and double precision.

%%
% First, look at the performance of the backslash operator in single
% precision.
fig = figure;
ax = axes('parent', fig);
plot(ax, results.sizeSingle, results.timeSingleGPU, '-x', ...
     results.sizeSingle, results.timeSingleCPU, '-o')
grid on;
legend('GPU', 'CPU', 'Location', 'NorthWest');
title(ax, 'Single-Precision Performance')
ylabel(ax, 'Time (s)');
xlabel(ax, 'Matrix Size');
drawnow;

%%
% Now, look at the performance of the backslash operator in double
% precision.
fig = figure;
ax = axes('parent', fig);
plot(ax, results.sizeDouble, results.timeDoubleGPU, '-x', ...
     results.sizeDouble, results.timeDoubleCPU, '-o')
legend('GPU', 'CPU', 'Location', 'NorthWest');
grid on;
title(ax, 'Double-Precision Performance')
ylabel(ax, 'Time (s)');
xlabel(ax, 'Matrix Size');
drawnow;

%%
% Finally, look at the speedup of the backslash operator when comparing
% the GPU to the CPU.
speedupDouble = results.timeDoubleCPU./results.timeDoubleGPU;
speedupSingle = results.timeSingleCPU./results.timeSingleGPU;
fig = figure;
ax = axes('parent', fig);
plot(ax, results.sizeSingle, speedupSingle, '-v', ...
     results.sizeDouble, speedupDouble, '-*')
grid on;
legend('Single-precision', 'Double-precision', 'Location', 'SouthEast');
title(ax, 'Speedup of Computations on GPU Compared to CPU');
ylabel(ax, 'Speedup');
xlabel(ax, 'Matrix Size');
drawnow;
