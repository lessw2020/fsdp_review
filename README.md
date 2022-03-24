# fsdp_review
Some eval and profile routines for fsdp

## /mnist holds fsdp_mnist.py  
This runs an fsdp CNN model for a default of 2 epochs to provide a review framework for FSDP.

#### Usage:</br>
~~~
python fsdp_mnist.py  <optional flags> 
~~~
#### Flags:
-- use_offload = Activates cpu_offload </br>
-- profile = Run PyTorch profile tracing </br>
-- save-model = Save model (only working on nightlies) </br>
</br>
<img width="585" alt="fsdp_mnist_use_offload" src="https://user-images.githubusercontent.com/46302957/159985372-dae03e6c-e527-4a89-8dbb-f7897b4bb672.png">

