# MLPerf Inference Calibration and Quantization Details

---

## FuriosaAI MLPerf Quantization

In the post-training quantization process, a dynamic range search for each weight and activation tensor is required.
Symmetric quantization is applied to the weights, while asymmetric quantization is applied to the activations.

### Weights


dynamic range values are determined by searching for the maximum absolute value(*AMAX*), with each weight tensor having its own *per-channel* *AMAX* value. For GPT-J, *SmoothQuant* is applied initially, after which the dynamic range values are defined.

### Activations


For each activation tensor, it is necessary to search for dynamic range values using a calibration dataset. Activation quantization follows an asymmetric scheme and the dynamic range value is initialized either *per-channel* or *per-tensor*, depending on the model and the architecture of the network. For BERT, dynamic range values are determined using *PERCENTILE* method. This method involves creating histograms based on the elements in the tensor bins, selecting a range that covers 99.99% of the element count in the tensor.

### Additional Details


 - From an acceleration perspective, quantizing all tensors to low bit precision would be ideal. However, considering the trade-off with accuracy, tensors related to linear layers are quantized to INT8 but other intermediate tensors are pre-determined to use appropriate representations such as FP32, BF16, or INT8.

 - Using KV cache quantization to cache values in INT8.

### BERT, GPT-J Qunatization Details

Note: applied selectively based on accuracy impact

When quantizing BERT, the following details are applied in each block:

- Weight: INT8, per-channel symmetric quantization, *AMAX*
- Activation: INT8, per-channel asymmetric quantization, *PERCENTILE*


When quantizing GPT-J, the following details are applied in each decoder block:

- Weight: INT8, per-channel symmetric quantization with SmoothQuant, *AMAX*
- Activation: INT8, per-tensor asymmetric quantization with SmoothQuant, *MINMAX*
- KV caching: INT8, per-head asymmetric quantization


### SmoothQuant

In MLPerf Inference v4.1, $\alpha$ = 0.5.
