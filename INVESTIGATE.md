# Investigate notes I got from learning the idol repo.

- The first time I work with MPS backend (for metal)
- Really easy to understand and easy to setup

## Example for layernorm
- The underlying for offset when get the tensor:
```python
B, T, C = 2, 3, 4
a = torch.randn(B, T, C)
print(a)

tensor([[[ 1.9269,  1.4873,  0.9007, -2.1055],
         [ 0.6784, -1.2345, -0.0431, -1.6047],
         [ 0.3559, -0.6866, -0.4934,  0.2415]],

        [[-1.1109,  0.0915, -2.3169, -0.2168],
         [-0.3097, -0.3957,  0.8034, -0.6216],
         [-0.5920, -0.0631, -0.8286,  0.3309]]])

# To get the a[b,t,c], we get the offset: b*T*C + t*C + c -> by this way we can access the tensor in C

# Here is the code to access one batch of input in C(pointer):
# seek to the input position inp[b,t,:]
#   float* x = inp + b * T * C + t * C;
```
- Understand how the LayerNorm is added in to each transformer block and make the training phase more stable

## train_gpt2

- Defined each struct and malloc memory in first init for tensors (16) and activation (23). Also the params for backward
- There are several forwards and backwards implementation for each layers:
```cpp
layernorm_forward()
matmul_forward() // Most of the time is spent here.
attention_forward()
residual_forward()
gelu_forward()
softmax_forward()
crossentropy_forward()

crossentropy_softmax_backward()
layernorm_backward()
residual_backward()
gelu_backward()
matmul_backward() // Most of the time is spent here.
attention_backward()
encoder_backward()
```

- Using omp to parallel the for loops:
```cpp
    #pragma omp parallel for collapse(2)
    for (int b = 0; b < B; b++) {
        for (int t = 0; t < T; t++) {
            float* dout_bt = dout + b * T * OC + t * OC;
            float* dinp_bt = dinp + b * T * C + t * C;
            for (int o = 0; o < OC; o++) {
                float* wrow = weight + o*C;
                float d = dout_bt[o];
                for (int i = 0; i < C; i++) {
                    dinp_bt[i] += wrow[i] * d;
                }
            }
        }
    }
```
`for` mean this omp will do the for loop, collapse(2) mean each thread will take 2 nested for loop to process.

## CUDA Programming
### Example for layernorm_forward

The initialize phase, we need to allocate memory for host and device
```cpp
//For the first trial, using the version 1: 
//parallelizes over B,T, loops over C

// parameters
int B = 8;
int T = 1024;
int C = 768;

// First we need to create host memory, for example:
float* out = (float*)malloc(B * T * C * sizeof(float)); // Need to cast into float* because malloc auto return void*

// Now we need to move to GPU
float* d_out;
float* d_mean;
float* d_rstd;
float* d_inp;
float* d_weight;
float* d_bias;
cudaCheck(cudaMalloc(&d_out, B * T * C * sizeof(float))); // No value, so only need to malloc the memory
cudaCheck(cudaMalloc(&d_mean, B * T * sizeof(float)));
cudaCheck(cudaMalloc(&d_rstd, B * T * sizeof(float)));
cudaCheck(cudaMalloc(&d_inp, B * T * C * sizeof(float)));
cudaCheck(cudaMalloc(&d_weight, C * sizeof(float)));
cudaCheck(cudaMalloc(&d_bias, C * sizeof(float)));
cudaCheck(cudaMemcpy(d_inp, inp, B * T * C * sizeof(float), cudaMemcpyHostToDevice)); // Need input value, so need to copy from host to device
cudaCheck(cudaMemcpy(d_weight, weight, C * sizeof(float), cudaMemcpyHostToDevice));
cudaCheck(cudaMemcpy(d_bias, bias, C * sizeof(float), cudaMemcpyHostToDevice));
```

Run the funtion (layernorm_forward1)
```cpp

// Blocksize = 256
// grid_size is calulated to determind how much block need to cover all N elements (N is batch size * sequence length)
// Parallel for B, T <number of blocks, number of thread per block>
// The grid_size is round up to 32 to cover all threads needed for all elements.
void layernorm_forward1(float* out, float* mean, float* rstd,
                           float* inp, float* weight, float* bias,
                           int B, int T, int C,
                           const int block_size) {
    const int N = B * T;
    const int grid_size = CEIL_DIV(N, block_size);
    layernorm_forward_kernel1<<<grid_size, block_size>>>(out, mean, rstd, inp, weight, bias, N, C);
    cudaCheck(cudaGetLastError());
}
```

Move to the GPU kernel to handle each element
```cpp
__global__ void layernorm_forward_kernel1(float* out, float* mean, float* rstd,
                                 float* inp, float* weight, float* bias,
                                 int N, int C) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    // blockIdx.x * blockDim.x to get the correct block id, after that is the threadIdx
    if (idx < N) {
        // seek to the input position inp[idx,:]
        float* x = inp + idx * C;
        // More code ...
    }
}
```

For the layernorm_forward2 kernel, we divide into 3 kernels: mean_kernel, rstd_kernel, and normalize_kernel. Let's dive in the mean_kernel
```cpp
mean_kernel<<<B * T, block_size, block_size * sizeof(float)>>>(mean, inp, N, C,block_size);

__global__ void mean_kernel(float* mean, float* inp, int N, int C, int block_size) {
    extern __shared__ float shared[]; // Faster than the global memory, but only for threads in the same block 
    int idx = blockIdx.x; // range [0, B*T)
    int tid = threadIdx.x; // range [0, block_size)
    float* x = inp + idx * C;
    // thread coarsening -> this make sure that one thread can handle multiple values
    // each thread processes a subset of the array elements in each pass, potentially looping over the array multiple times but doing more work in each pass
    float sum = 0.0f;
    for (int i = tid; i < C; i += block_size) {
        sum += x[i];
    }
    // 256 threads, each handle 3 elements due to C is 768
    shared[tid] = sum;
    __syncthreads(); // Make sure all threads are done
    // reductions
    // combine the partial sums computed by individual threads
    // This is achieved in a step-wise fashion, halving the number of active threads in each step until only one thread (thread 0) remains to write the final result.
    for (int stride = block_size / 2; stride >= 1; stride /= 2) {
        __syncthreads();
        if (tid < stride) {
            shared[tid] += shared[tid + stride];
        }
    }
    // write the final result (at thread 0) to global memory
    if (tid == 0) {
        mean[idx] = shared[0] / C;
    }
}
```
The rstd_kernel and normalization_kernel have the same intuative.
