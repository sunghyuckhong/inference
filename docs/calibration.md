# MLPerf Inference Calibration and Quantization

---

## FuriosaAI MLPerf Quantization

The post-training quantization process involves calculating the quantization parameters based on the dynamic ranges of the weight and activation tensors. Symmetric quantization is applied to the weights, while asymmetric quantization is applied to the activations.

### Weight Quantization

We use *per-channel* quantization for the weights, determining the quantization parameters from the maximum absolute values(*AMAX*) of the channels of each weight tensor. For GPT-J, we calculate the dynamic range values of the weights after applying *SmoothQuant*. 


### Activation Quantization
The dynamic range values of the activations are obtained through the use of a calibration dataset. Asymmetric quantization is applied to the activation tensors, with its granularity determined as either *per-channel* or *per-tensor* depending on the type of models. The quantization parameters of BERT are determined by the *PERCENTILE* calibration method, which involves creating histograms based on the tensor elements and selecting the dynamic range that covers 99.99% of the elements.


### Additional Details

 - Quantizing all the tensors to low bit precision can lead to latency improvement. However, in order to achieve the favorable tradeoff between latency and accuracy, we quantize the tensors related to linear layers as INT8 and other intermediate tensors as either FP32, BF16, or INT8. 

- We employ INT8 KV cache quantization to cope with the memory bottleneck of large language model (LLM) inference.



### Quantization Details for BERT & GPT-J 

Note: applied selectively based on accuracy impact.

The following quantization methods are applied to each block of BERT:

- Weight: INT8, per-channel symmetric quantization, *AMAX*
- Activation: INT8, per-channel asymmetric quantization, *PERCENTILE*


The following quantization methods are applied to each decoder block of GPT-J:

- Weight: INT8, per-channel symmetric quantization with SmoothQuant, *AMAX*
- Activation: INT8, per-tensor asymmetric quantization with SmoothQuant, *MINMAX*
- KV cache: INT8, per-head asymmetric quantization


### SmoothQuant

In MLPerf Inference v4.1, the migration strength for SmoothQuant is set as $\alpha$ = 0.5.
