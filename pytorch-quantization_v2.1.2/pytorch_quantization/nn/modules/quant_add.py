
import torch
import torch.nn

from pytorch_quantization import tensor_quant

from . import _utils

# ref https://pytorch.org/docs/stable/generated/torch.add.html

# L1Loss
class QuantResizeAdd(torch.nn.L1Loss, _utils.QuantInputMixin):
    default_quant_desc_input = tensor_quant.QUANT_DESC_8BIT_PER_TENSOR

    def __init__(self):
        super(QuantResizeAdd, self).__init__()
        self.init_quantizer(self.default_quant_desc_input)

    def forward(self, input, other):
        quant_input = self._input_quantizer(input)
        quant_other = self._input_quantizer(other)

        output = torch.add(quant_input, quant_other)
        return output

class QuantAdd(torch.nn.MSELoss, _utils.QuantInputMixin):
    default_quant_desc_input = tensor_quant.QUANT_DESC_8BIT_PER_TENSOR

    def __init__(self, **kwargs):
        quant_desc_input = _utils.pop_quant_desc_in_kwargs(self.__class__, input_only=True, **kwargs)
        super(QuantAdd, self).__init__()
        self.init_quantizer(quant_desc_input)

    def forward(self, input, other):
        quant_input = self._input_quantizer(input)
        output = torch.add(quant_input, other)
        return output
