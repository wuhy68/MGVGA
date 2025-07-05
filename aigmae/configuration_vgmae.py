from transformers.configuration_utils import PretrainedConfig

class AIGMAEConfig(PretrainedConfig):
    def __init__(
        self,
        cross_hidden_size=3584,
        cross_num_heads=8,
        freeze: false,
        hidden_size=64,
        model_type="AIGMAE",
        num_classes=4,
        num_cross_decoder_layers=2,
        num_encoder_layers=7,
        **kwargs,
    ):
        self.cross_hidden_size=cross_hidden_size
        self.cross_num_heads=cross_num_heads
        self.freeze=freeze
        self.hidden_size=hidden_size
        self.num_classes=num_classes
        self.num_cross_decoder_layers=num_cross_decoder_layers
        self.num_encoder_layers=num_encoder_layers

        super().__init__(
            **kwargs,
        )