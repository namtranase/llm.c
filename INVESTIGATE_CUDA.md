# CUDA Programming
## Example for layernorm_forward

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

## Example for gelu_forward

The initilize phase is simple as the layernorm, so I skip this part.

There is nothing special, the gelu cu is straight forward: only need to define the grid_size and block_size for the kernel
```cpp
__global__ void gelu_kernel(float* out, const float* inp, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    float s = sqrtf(2.0f / M_PI);
    if (i < N) {
        float xi = inp[i];
        float cube = 0.044715f * xi * xi * xi;
        out[i] = 0.5f * xi * (1.0f + tanhf(s * (xi + cube)));
    }
}
```

For the benchmark, from block_size 512 the results is almost the same
```
block_size   32 | time 0.309282 ms | bandwidth 162.736969 GB/s
block_size   64 | time 0.095865 ms | bandwidth 525.027222 GB/s
block_size  128 | time 0.070493 ms | bandwidth 713.993103 GB/s
block_size  256 | time 0.069624 ms | bandwidth 722.908569 GB/s
block_size  512 | time 0.072110 ms | bandwidth 697.983582 GB/s
block_size 1024 | time 0.075942 ms | bandwidth 662.765259 GB/s
```

## Example of matmul_forward

CPU code, loop over batch, sequence length, and output channel.
```cpp
// inp is (B,T,C), weight is (OC, C), bias is (OC)
// out will be (B,T,OC)

void matmul_forward_cpu(float* out,
                    float* inp, float* weight, float* bias,
                    int B, int T, int C, int OC) {
    // OC is short for "output channels"
    // inp is (B,T,C), weight is (OC, C), bias is (OC)
    // out will be (B,T,OC)
    #pragma omp parallel for collapse(2)
    for (int b = 0; b < B; b++) {
        for (int t = 0; t < T; t++) {
            float* out_bt = out + b * T * OC + t * OC;
            float* inp_bt = inp + b * T * C + t * C;
            for (int o = 0; o < OC; o++) {
                float val = (bias != NULL) ? bias[o] : 0.0f;
                float* wrow = weight + o*C;
                for (int i = 0; i < C; i++) {
                    val += inp_bt[i] * wrow[i];
                }
                out_bt[o] = val;
            }
        }
    }
}
```

Naive kernel, each thread handles one element of out (one channel)
```cpp
// two dimension of gridDim and blockDim for convenient with matrix multiply
dim3 gridDim(CEIL_DIV(B * T, sqrt_block_size), CEIL_DIV(OC, sqrt_block_size));
dim3 blockDim(sqrt_block_size, sqrt_block_size);

// this access to a row of weight matrix (one OC)
int bt = blockIdx.x * blockDim.x + threadIdx.x;
int oc = blockIdx.y * blockDim.y + threadIdx.y;

// actually I don't really understand this line, but for now I assume that it refer to the batch and the token position
int b = bt / BT;
int t = bt % BT;

// The bias should be add with other kernel, the author seem to find the best way to fuse this operation into matmul
int idx = blockIdx.x * blockDim.x + threadIdx.x;
int stride = blockDim.x * gridDim.x;
for (int i = idx; i < B * T * OC; i += stride) {
    int col = i % OC;
    out[i] += bias[col];
}
// Other code is the same as CPU code
```

cuBLAS kernel is fucking insane, it faster than normal kernel ~50 times

```cpp
// inp is (B*T, C), weight is (OC, C), out is (B*T, OC)
// cuBLAS dose: C = alpha *A * B + beta * C
// where A is mxk, B is kxn, C is mxn

// In Pytorch, we want: inp @ weight.T
// but cuBLAS is columm major, -> out.T = weight @ inp.T
// in this repo, the variables look transposed -> this become out.T = weight.T @ inp

// Need to call cuBLAS with A = weight, B = inp

const float alpha = 1.0f;
const float beta = 0.0f;
stat = cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, OC, B*T, C, &alpha, weight, C, inp, C, &beta, out, OC);
// A need to transpose via CUBLAS_OP_T
```


