# Investigate notes I got from learning the idol repo.

- The first time I work with MPS backend (for metal)
- Really easy to understand and easy to setup

## Example for layernorm
- The underlying for offset when get the tensor:
```
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
#// seek to the input position inp[b,t,:]
#   float* x = inp + b * T * C + t * C;
```
- Understand how the LayerNorm is added in to each transformer block and make the training phase more stable

## train_gpt2

- Defined each struct and malloc memory in first init for tensors (16) and activation (23). Also the params for backward
- There are several forwards and backwards implementation for each layers:
```
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
```
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