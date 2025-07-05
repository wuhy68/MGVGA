# Circuit Representation Learning with Masked Gate Modeling and Verilog-AIG Alignment (ICLR'25)


## Data
We provide an example of data in /data folder.

## Weights
We provide the pre-trained weights in /weights folder.

The following contains a code snippet illustrating how to load the model.

```python
from aigmae.configuration_vgmae import AIGMAEConfig
from aigmae.modeling_vgmae import AIGMAEEmbeddingModel

model_config = AIGMAEConfig(
    num_classes = 4,
    num_encoder_layers = 7,
    hidden_size = 64,
)

pretrained_path = "weights/deepgcn_e7_cd2_pretrain_mgm0.3_vga0.5-checkpoint-4983/"
model = AIGMAEEmbeddingModel.from_pretrained(pretrained_path, config=model_config)
```

