import sys
from typing import Any
# from torch.nn import Module, Linear, Parameter, Conv2d, ReLU
import paddle
# from torch import Tensor, FloatTensor
# from torch.autograd import Function
from paddle.autograd import PyLayer
# from torch.autograd.function import _ContextMethodMixin
from functools import partial
import paddle.nn.functional as F
from paddle.nn import Layer, Linear, Conv2D, ReLU
import paddle.nn as nn
import math
from paddle import Tensor

calibrated_num = 0
total_num = 0


class LoRALayer:
    def __init__(
        self, 
        r: int, 
        lora_alpha: int, 
        lora_dropout: float,
        merge_weights: bool,
    ):
        self.r = r
        self.lora_alpha = lora_alpha
        self.lora_dropout = nn.Dropout(p=lora_dropout) if lora_dropout > 0. else nn.Identity()
        self.merged = False
        self.merge_weights = merge_weights


class LoRALinear(nn.Layer, LoRALayer):
    """LoRA-Enhanced Linear Layer with Pretrained Weights"""
    def __init__(
        self, 
        pretrained_layer: nn.Linear,  # 预训练好的线性层
        r: int = 0,                   # LoRA秩
        lora_alpha: int = 1, 
        lora_dropout: float = 0.,
        fan_in_fan_out: bool = False,
        merge_weights: bool = True,
    ):
        super().__init__()
        LoRALayer.__init__(self, r=r, lora_alpha=lora_alpha, 
                          lora_dropout=lora_dropout, merge_weights=merge_weights)
        
        # 从预训练层继承参数
        self.in_features = pretrained_layer.in_features
        self.out_features = pretrained_layer.out_features
        self.fan_in_fan_out = fan_in_fan_out
        
        # 继承原始权重和偏置
        self.weight = paddle.create_parameter(
            shape=pretrained_layer.weight.shape,
            default_initializer=paddle.nn.initializer.Assign(pretrained_layer.weight.clone())
        )
        if pretrained_layer.bias is not None:
            self.bias = paddle.create_parameter(
                shape=pretrained_layer.bias.shape,
                default_initializer=paddle.nn.initializer.Assign(pretrained_layer.bias.clone())
            )
        else:
            # 修改点: register_parameter('bias', None) -> 直接设置为 None
            self.bias = None
        
        # 初始化LoRA参数
        if r > 0:
            self.lora_A = paddle.create_parameter(shape=[r, self.in_features], default_initializer=paddle.nn.initializer.Constant(0.))
            self.lora_B = paddle.create_parameter(shape=[self.out_features, r], default_initializer=paddle.nn.initializer.Constant(0.))
            self.scaling = lora_alpha / r
            kaiming_uniform_ = paddle.nn.initializer.KaimingUniform(negative_slope=math.sqrt(5))
            kaiming_uniform_(self.lora_A)
            self.weight.stop_gradient = True # 冻结原始权重
            if self.bias is not None:
                self.bias.stop_gradient = True
                
        # 处理fan_in_fan_out
        if fan_in_fan_out:
            self.weight.data = self.weight.data.T

    def reset_lora_parameters(self):
        """仅重置LoRA参数"""
        if hasattr(self, 'lora_A'):
            kaiming_uniform_ = paddle.nn.initializer.KaimingUniform(negative_slope=math.sqrt(5))
            kaiming_uniform_(self.lora_A)
            # 修改点: nn.init.zeros_ -> paddle.nn.initializer
            zeros_ = paddle.nn.initializer.Constant(0.)
            zeros_(self.lora_B)

    def train(self, mode: bool = True):
        """模式切换时处理权重合并"""
        self.training = mode
        if mode:  # 训练模式
            if self.merge_weights and self.merged:
                if self.r > 0:
                    self.weight.data -= self._get_lora_weights()
                self.merged = False
        else:      # 评估模式
            if self.merge_weights and not self.merged:
                if self.r > 0:
                    self.weight.data += self._get_lora_weights()
                self.merged = True
        return self

    def _get_lora_weights(self) -> paddle.Tensor:
        """计算LoRA权重调整量"""
        lora_weights = (self.lora_B @ self.lora_A) * self.scaling
        return lora_weights.T if self.fan_in_fan_out else lora_weights

    def forward(self, x: paddle.Tensor) -> paddle.Tensor:
        if self.r > 0 and not self.merged:
            # 原始前向计算
            result = F.linear(x, self._get_effective_weight(), self.bias)
            # LoRA分支
            x_drop = self.lora_dropout(x)
            lora_adjust = (x_drop @ self.lora_A.T @ self.lora_B.T) * self.scaling
            return result + lora_adjust
        else:
            return F.linear(x, self._get_effective_weight(), self.bias)

    def _get_effective_weight(self) -> paddle.Tensor:
        """根据模式返回有效权重"""
        if self.fan_in_fan_out:
            return self.weight.T
        return self.weight



def quant(input:paddle.Tensor, lb:float, ub:float, bit:int):
    input = paddle.clip(input,lb, ub)
    # print(ub.device, lb.device, bit.device)
    s = (ub - lb) / (2 ** bit -1)
    input = (input - lb)/s
    input = input.round()
    input = input * s + lb
    return input

def cal_mse(input:paddle.Tensor, lb:float, ub:float, bit:int):
    quant_input = quant(input, lb, ub, bit)
    res = float(paddle.norm(input - quant_input))
    return res

def DOBI(input:paddle.Tensor, bit:int, one_direction = False, num:int=100):
    min_value = paddle.min(input)
    max_value = paddle.max(input)
    
    diff = (max_value - min_value) / (2 * num)
    
    history_min = float('inf')
    input = input.cuda()
    
    if one_direction:
        diff = (max_value - min_value) / num
        for i in range(num):
            lb = min_value
            ub = max_value - diff * i
            cur_value = cal_mse(input, lb, ub, bit)
            if cur_value < history_min:
                best_lb = lb
                best_ub = ub
                history_min = cur_value
    else:
        diff = (max_value - min_value) / (2 * num)
        for i in range(num):
            lb = min_value + diff * i
            ub = max_value - diff * i
            cur_value = cal_mse(input, lb, ub, bit)
            if cur_value < history_min:
                best_lb = lb
                best_ub = ub
                history_min = cur_value
    global calibrated_num
    global total_num
    calibrated_num += 1
    print(f'calibration:{calibrated_num}/{total_num}')
    
    return float(best_lb), float(best_ub)

class Differentiable_Round(PyLayer):
    @staticmethod
    def forward(ctx, x: Tensor):
        return x.round()

    @staticmethod
    def backward(ctx, grad_outputs):
        return grad_outputs


class Differentiable_Clip(PyLayer):
    @staticmethod
    def forward(
        ctx,
        input: Tensor,
        min_val: Tensor,
        max_val: Tensor,
    ) -> Any:
        ctx.save_for_backward(input, min_val, max_val)
        # if isinstance(min_val, Tensor):
        #     min_val = min_val.item()
        # if isinstance(max_val, Tensor):
        #     max_val = max_val.item()
        if min_val > max_val:
            return paddle.full_like(input, fill_value=max_val)
        output = paddle.clip(input, min_val.item(), max_val.item())
        return output

    @staticmethod
    def backward(ctx, grad_outputs: Tensor) -> Any:
        input, min_val, max_val = ctx.saved_tensor()

        grad_input = grad_outputs.clone()
        grad_input[(input < min_val) | (input > max_val)] = 0
        
        grad_min = grad_outputs.clone()
        grad_min[input > min_val] = 0
        grad_min = grad_min.sum().reshape([1])

        grad_max = grad_outputs.clone()
        grad_max[input < max_val] = 0
        grad_max = grad_max.sum().reshape([1])
        return grad_input, grad_min, grad_max


class FakeQuantizerBase(Layer):
    def __init__(self, int_quant: bool = True, bit:int=4) -> None:
        super().__init__()
        self.lower_bound = paddle.create_parameter(
            shape=[],
            dtype='float32',
            default_initializer=paddle.nn.initializer.Normal()
        )
        self.upper_bound = paddle.create_parameter(
            shape=[],
            dtype='float32',
            default_initializer=paddle.nn.initializer.Normal()
        )
        self.n_bit = paddle.create_parameter(
            shape=[],
            dtype='float32',
            default_initializer=paddle.nn.initializer.Normal()
        )
        # self.n_bit = bit
        self.set_n_bit_manually(bit)
        
        self.bit2bound = {}
        self.use_bit2bound = False
        self.size_of_input = None
        
        self.int_quant = int_quant

        self.clip = Differentiable_Clip.apply
        self.round = Differentiable_Round.apply
        
        self.calibrated = False
        self.one_direction_search = False
        
        global total_num 
        total_num += 1
        

    def set_int_quant(self, enable: bool):
        self.int_quant = enable

    def set_require_grad(self, enable_lb: bool, enable_up: bool, enable_nbit: bool):
        self.lower_bound.requires_grad = enable_lb
        self.upper_bound.requires_grad = enable_up
        # self.n_bit.requires_grad = enable_nbit

    def set_params_manually(self, lb: Tensor, ub: Tensor, n_bit: Tensor):
        device = self.lower_bound.place
        self.lower_bound.set_value(paddle.to_tensor(lb, dtype='float32'))
        self.upper_bound.set_value(paddle.to_tensor(ub, dtype='float32'))
        
    def set_params_lb_manually(self, lb: Tensor):
        device = self.lower_bound.place
        self.lower_bound.set_value(paddle.to_tensor(lb, dtype='float32'))
    
    def set_params_ub_manually(self, ub: Tensor):
        device = self.upper_bound.place
        self.upper_bound.set_value(paddle.to_tensor(ub, dtype='float32'))

    def set_n_bit_manually(self, n_bit):
        device = self.n_bit.place
        self.n_bit.set_value(paddle.to_tensor(n_bit, dtype='float32'))


class FakeQuantizerWeight(FakeQuantizerBase):
    def __init__(self,bit=4) -> None:
        super(FakeQuantizerWeight, self).__init__(bit=bit)

    def forward(self, x:paddle.Tensor):
        if not self.calibrated:
            lb, rb = DOBI(x, bit=self.n_bit, one_direction=self.one_direction_search)
            self.set_params_lb_manually(lb)
            self.set_params_ub_manually(rb)
            self.calibrated = True
            return x
        if self.size_of_input is None:
            self.size_of_input = x.numel()
        
        n_bits = self.n_bit if not self.int_quant else self.round(self.n_bit)
        
        if self.use_bit2bound:
            try:
                lb, ub = self.bit2bound[int(n_bits.item())]
                self.set_params_lb_manually(lb)
                self.set_params_ub_manually(ub)
            except Exception as e:
                print(f'use bit 2 bound.{int(n_bits.item())} not found.')

        # (u-l)/(2^n-1)
        s = (self.upper_bound - self.lower_bound) / ((2 ** n_bits - 1))

        # clip(x,l,u)
        c = self.clip(x, self.lower_bound, self.upper_bound)

        # int value \in [0,2^n-1]
        r = self.round((c - self.lower_bound) / s)

        return s * r + self.lower_bound
    
        # n_bits = self.n_bit if not self.int_quant else self.round(self.n_bit)

        # # (u-l)/(2^n-1)
        # s = (self.upper_bound - self.lower_bound) / torch.pow(2, n_bits)

        # # clip(x,l,u)
        # c = self.clip(x, self.lower_bound, self.upper_bound)

        # # int value \in [0,2^n-1]
        # # r = self.clip(self.round((c - self.lower_bound) / s - 0.5, 0, torch.pow(2, n_bits)-1))
        # r = self.clip(
        #     self.round((c - self.lower_bound) / s - 0.5),
        #     0, torch.pow(2, n_bits)-1
        #     )

        # return s * (r+0.5) + self.lower_bound


# class FakeQuantizer_Lora_QuantWeight(FakeQuantizerWeight):
#     def __init__(self,q_weight:FakeQuantizerWeight,
#                  weight,
#                  r=0) -> None:
#         super(FakeQuantizer_Lora_QuantWeight, self).__init__()
#         self.n_bit = q_weight.n_bit
#         self.calibrated=q_weight.calibrated
#         self.set_params_manually(q_weight.lower_bound,
#                                  q_weight.upper_bound,
#                                  q_weight.n_bit)
#         if len(weight.shape)==3:
#             weight = weight[0]

#         if r > 0:
#             self.lora_A = nn.Parameter(weight.new_zeros((r, weight.shape[0])))
#             self.lora_B = nn.Parameter(weight.new_zeros((weight.shape[1], r)))
#         if hasattr(self, 'lora_A'):
#             # initialize A the same way as the default for nn.Linear and B to zero
#             nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
#             nn.init.zeros_(self.lora_B)
        
        

#     def forward(self, x:paddle.Tensor):
#         if not self.calibrated:
#             lb, rb = DOBI(x, bit=self.n_bit, one_direction=self.one_direction_search)
#             self.set_params_lb_manually(lb)
#             self.set_params_ub_manually(rb)
#             self.calibrated = True
#             return x
#         if self.size_of_input is None:
#             self.size_of_input = x.numel()
        
#         n_bits = self.n_bit if not self.int_quant else self.round(self.n_bit)
        
#         if self.use_bit2bound:
#             try:
#                 lb, ub = self.bit2bound[int(n_bits.item())]
#                 self.set_params_lb_manually(lb)
#                 self.set_params_ub_manually(ub)
#             except Exception as e:
#                 print(f'use bit 2 bound.{int(n_bits.item())} not found.')

#         # (u-l)/(2^n-1)
#         s = (self.upper_bound - self.lower_bound) / (torch.pow(2, n_bits) - 1)

#         # clip(x,l,u)
        

#         c = self.clip(x+(self.lora_A.transpose(0, 1) @ self.lora_B.transpose(0, 1)), self.lower_bound, self.upper_bound)

#         # int value \in [0,2^n-1]
#         r = self.round((c - self.lower_bound) / s)

#         return s * r + self.lower_bound


class FakeQuantizerAct(FakeQuantizerBase):
    def __init__(self,bit=4) -> None:
        """
        if dynamic, bound will be the minmax value of the input values.
        dynamic is only used for act.
        if running_stat, bound will be set by moving average.
        """
        super(FakeQuantizerAct, self).__init__(bit=bit)

        self.running_stat = False  # initial boundary
        self.first_iter = False  # initial boundary
        self.dynamic = False
        self.beta = 0.995
        self.identity = False

    def forward(self, x):
        if not self.calibrated:
            lb, rb = DOBI(x, bit=self.n_bit, one_direction=self.one_direction_search)
            self.set_params_lb_manually(lb)
            self.set_params_ub_manually(rb)
            self.calibrated = True
            return x

        if self.size_of_input is None:
            self.size_of_input = x.numel()

        if self.identity:
            return x
        
        if self.dynamic:
            '''
                use min,max as clip boundary
                TODO: use other observer later.
            '''
            n_bits = self.n_bit if not self.int_quant else self.round(self.n_bit)

            lb = paddle.min(x).detach()
            ub = paddle.max(x).detach()
            n_bits = n_bits.detach()


            # (u-l)/(2^n-1)
            s = (ub - lb) / ((2 ** n_bits - 1))

            

            # clip(x,l,u)
            c = self.clip(x, lb, ub)

            # int value \in [0,2^n-1]
            r = self.round((c - lb) / s)


            return s * r + lb

        if self.running_stat:
            if self.first_iter:
                self.lower_bound.data = paddle.min(x).detach().clone()
                self.upper_bound.data = paddle.max(x).detach().clone()
                self.first_iter = False
            else:
                self.lower_bound =  self.beta * self.lower_bound + (1-self.beta) *  paddle.min(x)
                self.upper_bound =  self.beta * self.upper_bound + (1-self.beta) *  paddle.max(x)
            return x

        n_bits = self.n_bit if not self.int_quant else self.round(self.n_bit)
        if self.use_bit2bound:
            try:
                lb, ub = self.bit2bound[int(n_bits.item())]
                self.set_params_lb_manually(lb)
                self.set_params_ub_manually(ub)
            except Exception as e:
                print(f'use bit 2 bound.{int(n_bits.item())} not found.')

        # (u-l)/(2^n-1)
        s = (self.upper_bound - self.lower_bound) / ((2 ** n_bits - 1))


        # clip(x,l,u)
        c = self.clip(x, self.lower_bound, self.upper_bound)


        # int value \in [0,2^n-1]
        r = self.round((c - self.lower_bound) / s)

        return s * r + self.lower_bound

        
        # n_bits = self.n_bit if not self.int_quant else self.round(self.n_bit)

        # # (u-l)/(2^n-1)
        # s = (self.upper_bound - self.lower_bound) / torch.pow(2, n_bits)

        # # clip(x,l,u)
        # c = self.clip(x, self.lower_bound, self.upper_bound)

        # # int value \in [0,2^n-1]
        #     r = self.clip(
        #     self.round((c - self.lower_bound) / s - 0.5),
        #     0, torch.pow(2, n_bits)-1
        #     )

        # return s * (r+0.5) + self.lower_bound
    
##########################################################
##########################################################


class FakeQuantizerWeightParam2(FakeQuantizerBase):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x):

        # n_bits = self.n_bit if not self.int_quant else self.round(self.n_bit)

        # (u-l)/(2^n-1)
        s = (self.upper_bound - self.lower_bound) / self.n_bit

        # clip(x,l,u)
        c = self.clip(x, self.lower_bound, self.upper_bound)

        # int value \in [0,2^n-1]
        r = self.round((c - self.lower_bound) / s)

        return s * r + self.lower_bound


class FakeQuantizerActParam2(FakeQuantizerBase):
    def __init__(self) -> None:
        """
        if dynamic, bound will be the minmax value of the input values.
        dynamic is only used for act.
        if running_stat, bound will be set by moving average.
        """
        super().__init__()

        self.running_stat = True # initial boundary
        self.first_iter = True # initial boundary
        self.dynamic = False
        self.beta = 0.995

    def forward(self, x):
        if self.dynamic:
            '''
                use min,max as clip boundary
                TODO: use other oberver later.
            '''
            # n_bits = self.n_bit if not self.int_quant else self.round(self.n_bit)
            n_bits = self.n_bit
            lb = paddle.min(x).detach()
            ub = paddle.max(x).detach()
            n_bits = n_bits.detach()

            # (u-l)/(2^n-1)
            s = (ub - lb) / n_bits

            # clip(x,l,u)
            c = self.clip(x, lb, ub)

            # int value \in [0,2^n-1]
            r = self.round((c - lb) / s)

            return s * r + lb

        if self.running_stat:
            if self.first_iter:
                self.lower_bound.data = paddle.min(x).detach().clone()
                self.upper_bound.data = paddle.max(x).detach().clone()
                self.first_iter = False
            else:
                self.lower_bound =  self.beta * self.lower_bound + (1-self.beta) *  paddle.min(x)
                self.upper_bound =  self.beta * self.upper_bound + (1-self.beta) *  paddle.max(x)
            return x


        # n_bits = self.n_bit if not self.int_quant else self.round(self.n_bit)

        # (u-l)/(2^n-1)
        s = (self.upper_bound - self.lower_bound) / n_bits

        # clip(x,l,u)
        c = self.clip(x, self.lower_bound, self.upper_bound)

        # int value \in [0,2^n-1]
        r = self.round((c - self.lower_bound) / s)

        return s * r + self.lower_bound


class QuantBase(Layer):
    def __init__(self,config):
        super().__init__()
        self.quant = True
        self.bit = config['bit']
        self.weight_quantizer = FakeQuantizerWeight(self.bit)
        self.act_quantizer = FakeQuantizerAct(self.bit)
        # if config['param'] == 1:
        #     self.weight_quantizer = FakeQuantizerWeight()
        #     self.act_quantizer = FakeQuantizerAct()
        # elif config['param'] == 2:
        #     self.weight_quantizer = FakeQuantizerWeightParam2()
        #     self.act_quantizer = FakeQuantizerActParam2()

    def get_weight_quantizer(self):
        return self.weight_quantizer

    def get_act_quantizer(self):
        return self.act_quantizer

    def set_quant_flag(self, enable: bool):
        self.quant = enable

    def set_require_grad(self, enable: bool):
        # 似乎没有必要分别设置.
        self.weight_quantizer.set_require_grad(enable,enable, enable)
        self.act_quantizer.set_require_grad(enable,enable, enable)

    def set_weight_bias_grad(self, enable: bool):
        self.weight.requires_grad = enable
        if self.bias:
            self.bias.requires_grad = enable

    def get_quant_weight_bias(self):
        quant_weight = self.weight_quantizer(self.weight)

        return (quant_weight, self.bias)


class QuantLinear(QuantBase):
    def __init__(self,config):
        super().__init__(config)
    
    def load_values(self, value):
        min_value, max_value = value
        self.act_quantizer.set_params_lb_manually(min_value)
        self.act_quantizer.set_params_ub_manually(max_value)

    def set_param(self, linear: Linear):
        """
        must be called before forward.
        """

        self.in_feature = linear.weight.shape[0]
        self.out_feature = linear.weight.shape[1]

        self.weight = paddle.create_parameter(
            shape=linear.weight.shape,
            dtype='float32',
            default_initializer=paddle.nn.initializer.Assign(linear.weight.clone())
        )
        if linear.bias is not None:
            self.bias = paddle.create_parameter(
                shape=linear.bias.shape,
                dtype='float32',
                default_initializer=paddle.nn.initializer.Assign(linear.bias.clone())
            )
        else:
            self.bias = linear.bias

    def forward(self, x):
        if not self.quant:
            return F.linear(x, self.weight, self.bias)
        quant_act = self.act_quantizer(x)
        quant_weight = self.weight_quantizer(self.weight)
        # bias don't need to be quant
        return F.linear(quant_act, quant_weight, self.bias)


class Quant4LoRA_QuantLinear(QuantLinear, LoRALayer):
    """支持四个LoRA矩阵的量化线性层，并用可训练归一化权重加权"""
    def __init__(self, 
                 quant_linear: QuantLinear,  # 预训练好的量化线性层
                 config,  
                 r: int = 0,                 # LoRA秩
                 lora_alpha: int = 1,
                 lora_dropout: float = 0.,
                 merge_weights: bool = False,
                 num_lora: int = 4,
                 is_finetune_flag: bool = False):          # 新增四个LoRA矩阵
        QuantLinear.__init__(self, config=config)
        LoRALayer.__init__(self, r=r, lora_alpha=lora_alpha, 
                          lora_dropout=lora_dropout, merge_weights=merge_weights)

        quant_linear.in_features = quant_linear.in_feature
        quant_linear.out_features = quant_linear.out_feature
        self.set_param(quant_linear)  # 继承原始权重和量化参数
        self.quant = quant_linear.quant
        self.act_quantizer.set_params_manually(lb=quant_linear.act_quantizer.lower_bound, 
                                               ub=quant_linear.act_quantizer.upper_bound, 
                                               n_bit=quant_linear.act_quantizer.n_bit)
        self.act_quantizer.calibrated = quant_linear.act_quantizer.calibrated
        self.weight_quantizer.set_params_manually(lb=quant_linear.weight_quantizer.lower_bound, 
                                                  ub=quant_linear.weight_quantizer.upper_bound, 
                                                  n_bit=quant_linear.weight_quantizer.n_bit)
        self.weight_quantizer.calibrated = quant_linear.weight_quantizer.calibrated
        self.top_k = 4
        self.is_finetune = is_finetune_flag
        # 初始化四个LoRA矩阵
        self.num_lora = num_lora
        if r > 0:
            self.lora_A = nn.ParameterList([
                paddle.create_parameter(
                    shape=[r, self.in_feature], 
                    dtype='float32', 
                    default_initializer=paddle.nn.initializer.Constant(0.)
                ) for _ in range(num_lora)
            ])
            self.lora_B = nn.ParameterList([
                paddle.create_parameter(
                    shape=[self.out_feature, r], 
                    dtype='float32', 
                    default_initializer=paddle.nn.initializer.Constant(0.)
                ) for _ in range(num_lora)
            ])
            self.scaling = self.lora_alpha / r
            kaiming_initializer = paddle.nn.initializer.KaimingUniform(negative_slope=math.sqrt(5))
            for i in range(num_lora):
                kaiming_initializer(self.lora_A[i])

            self.weight.stop_gradient = True
            if self.bias is not None:
                self.bias.stop_gradient = True

            self.lora_alpha_params = paddle.create_parameter(
                shape=[num_lora], 
                dtype='float32', 
                default_initializer=paddle.nn.initializer.Constant(1.0 / num_lora)
            )

    def _get_lora_weights(self) -> paddle.Tensor:
        weights = F.softmax(self.lora_alpha_params, axis=0)  # 归一化
        lora_sum = sum(weights[i] * (self.lora_B[i] @ self.lora_A[i]) for i in range(self.num_lora))
        return lora_sum 

    def forward(self, x):
        if not self.quant:
            return F.linear(x, self.weight, self.bias)
        else:
            quant_act = self.act_quantizer(x)
            if self.r > 0 and not self.merged:
                quant_weight = self.weight_quantizer(self.weight + self._get_lora_weights().T)
            else:
                quant_weight = self.weight_quantizer(self.weight)
            
            return F.linear(quant_act, quant_weight, self.bias)
    

class QuantLinearQKV(Layer):
    def __init__(self,config):
        super().__init__()
        self.q = QuantLinear(config)
        self.k = QuantLinear(config)
        self.v = QuantLinear(config)
    
    def load_values(self, value):
        min_value, max_value = value
        self.q.act_quantizer.set_params_lb_manually(min_value)
        self.q.act_quantizer.set_params_ub_manually(max_value)
        self.k.act_quantizer.set_params_lb_manually(min_value)
        self.k.act_quantizer.set_params_ub_manually(max_value)
        self.v.act_quantizer.set_params_lb_manually(min_value)
        self.v.act_quantizer.set_params_ub_manually(max_value)
        
    def set_quant_flag(self, enable: bool):
        self.q.set_quant_flag(enable)
        self.k.set_quant_flag(enable)
        self.v.set_quant_flag(enable)
        
    def set_require_grad(self, enable: bool):
        # 似乎没有必要分别设置.
        self.q.set_require_grad(enable)
        self.k.set_require_grad(enable)
        self.v.set_require_grad(enable)


    def set_weight_bias_grad(self, enable: bool):
        self.q.set_weight_bias_grad(enable)
        self.k.set_weight_bias_grad(enable)
        self.v.set_weight_bias_grad(enable)
    

    def get_quant_weight_bias(self):
        w_q,b_q = self.q.get_quant_weight_bias()
        w_k,b_k = self.k.get_quant_weight_bias()
        w_v,b_v = self.v.get_quant_weight_bias()
        
        quant_weight = paddle.concat([w_q, w_k, w_v], axis=0)
        if b_q is not None:
            bias = paddle.concat([b_q,b_k,b_v])

        return (quant_weight, bias)

    def set_param(self, linear: Linear):
        """
        must be called before forward.
        """

        self.in_feature = linear.weight.shape[0]
        self.out_feature = linear.weight.shape[1] // 3
        
        linear_q = Linear(self.in_feature, self.out_feature)
        linear_k = Linear(self.in_feature, self.out_feature)
        linear_v = Linear(self.in_feature, self.out_feature)
        
        
        linear_q.weight.data, linear_k.weight.data, linear_v.weight.data = (linear.weight.data.clone().transpose([1, 0]).reshape(3, self.out_feature, self.in_feature).transpose([0, 2, 1]))
        
        
        if linear.bias is not None:
            linear_q.bias.data, linear_k.bias.data, linear_v.bias.data = linear.bias.data.clone().reshape(3,self.out_feature)
    
        self.q.set_param(linear_q)
        self.k.set_param(linear_k)
        self.v.set_param(linear_v)


    def forward(self, x):
        q = self.q(x)
        k = self.k(x)
        v = self.v(x)
        # print('qsize',q.size())
        # print('ksize',k.size())
        # print('vsize',v.size())
        return paddle.concat([q,k,v], dim=-1)
    
        quant_act = self.act_quantizer(x)
        quant_weight = self.weight_quantizer(self.weight)
        # bias don't need to be quant

        return F.linear(quant_act, quant_weight, self.bias)


class QuantConv2d(QuantBase):
    def __init__(self,config):
        super().__init__(config)

    def set_param(self, conv: Conv2D):
        """
        must be called before forward.
        """

        self.in_channels = conv._in_channels
        self.out_channels = conv._out_channels
        self.kernel_size = conv._kernel_size
        self.conv_kwargs = {
            "stride": conv._stride,
            "padding": conv._padding,
            "dilation": conv._dilation,
            "groups": conv._groups,
        }
        self.weight = paddle.create_parameter(
            shape=conv.weight.shape,
            dtype='float32',
            default_initializer=paddle.nn.initializer.Assign(conv.weight.clone())
        )
        if conv.bias is not None:
            self.bias = paddle.create_parameter(
                shape=conv.bias.shape,
                dtype='float32',
                default_initializer=paddle.nn.initializer.Assign(conv.bias.clone())
            )
        else:
            self.bias = conv.bias

    def forward(self, x):
        if not self.quant:
            return F.conv2d(x, self.weight, self.bias, **self.conv_kwargs)

        quant_act = self.act_quantizer(x)
        quant_weight = self.weight_quantizer(self.weight)
        # bias doesn't need to be quant

        return F.conv2d(quant_act, quant_weight, self.bias, **self.conv_kwargs)



if __name__ == "__main__":

    def differentiable_test():
        x = Tensor((1.0, 2.0, 3.0, 4.0, 5.0))
        lb = Tensor((2.0,))
        ub = Tensor((4.0,))

        n = Tensor((2.0,))

        x.requires_grad = True
        lb.requires_grad = True
        ub.requires_grad = True
        n.requires_grad = True

        s = (ub - lb) / (paddle.pow(2, n) - 1)
        print(f"s:{s}")
        round = Differentiable_Round.apply
        clip = Differentiable_Clip.apply

        clipped = clip(x, lb, ub)
        subbed = clipped - lb
        normd = subbed / s
        roundded = round(normd)
        unnormed = s * roundded + lb

        x_hat_sum = unnormed.sum()
        x_hat_sum.backward()

        print("clipped", clipped)
        print("subbed", subbed)
        print(subbed[2] / s)
        print(type(s))
        print("normd", normd, float(normd[2]))
        print("roundded", roundded)
        print("unnormed", unnormed)
        print("=" * 10)
        print(x_hat_sum)
        print(x.grad)
        print(lb.grad)
        print(ub.grad)
        print(n.grad)

    def linear_test():
        l = Linear(10, 10, True)
        l_nb = Linear(10, 10, False)

        print(l.weight)
        print(l.bias)

        print(l_nb.weight)
        print(l_nb.bias)
        if l_nb.bias:
            print(1)
        else:
            print(2)

    # differentiable_test()
    # linear_test()

    def conv_test():
        c = Conv2D(2, 6, (3, 4), 2, 1, bias=True)
        c_nb = Conv2D(3, 6, 3, 2, 1, bias=False)

        print(c.weight.shape)
        # n_kernal(out_channels),in_channels, kerner_shape_H, kernel_shape_W
        print(c.bias.shape)  # n_kernal(out_channels)

        print(c_nb.weight.shape)
        print(c_nb.bias)

        # print(c.)

    def quant_sim_test():
        paddle.seed(1234)
        l = Linear(10, 10, True)

        l_q = QuantLinear()
        l_q.set_param(l)
        l_q.set_quant_flag(False)

        input = paddle.randn((16, 3, 10))
        out1 = l(input)
        out2 = l_q(input)

        # No quant: tensor(0., grad_fn=<SumBackward0>)
        print("No quant:", (out1 - out2).sum())

        l_q.set_quant_flag(True)
        act_min_val, act_max_val = input.min(), input.max()
        weight_min_val, weight_max_val = l_q.weight.min(), l_q.weight.max()

        l_q.get_weight_quantizer().set_params_manually(
            weight_min_val, weight_max_val, 4
        )
        l_q.get_act_quantizer().set_params_manually(act_min_val, act_max_val, 4)

        out1 = l(input)
        out2 = l_q(input)

        # 4bit quant: tensor(0.7627, grad_fn=<SumBackward0>)
        print("4bit quant:", (out1 - out2).sum())

        l_q.get_weight_quantizer().set_params_manually(
            weight_min_val, weight_max_val, 8
        )
        l_q.get_act_quantizer().set_params_manually(act_min_val, act_max_val, 8)

        out1 = l(input)
        out2 = l_q(input)
        # 8bit quant: tensor(-0.1381, grad_fn=<SumBackward0>)
        print("8bit quant:", (out1 - out2).sum())

        ######################################
        ######################################
        c = Conv2D(3, 8, 3)

        c_q = QuantConv2d()
        c_q.set_param(c)
        c_q.set_quant_flag(False)

        input = paddle.randn((16, 3, 15, 15))
        c_q.set_quant_flag(False)
        out1 = c(input)
        out2 = c_q(input)

        # No quant: tensor(0., grad_fn=<SumBackward0>)
        print("No quant:", (out1 - out2).sum())

        c_q.set_quant_flag(True)
        act_min_val, act_max_val = input.min(), input.max()
        weight_min_val, weight_max_val = c_q.weight.min(), c_q.weight.max()

        c_q.get_weight_quantizer().set_params_manually(
            weight_min_val, weight_max_val, 4
        )
        c_q.get_act_quantizer().set_params_manually(act_min_val, act_max_val, 4)

        out1 = c(input)
        out2 = c_q(input)

        # 4bit quant: tensor(0.7627, grad_fn=<SumBackward0>)
        print("4bit quant:", (out1 - out2).sum())

        c_q.get_weight_quantizer().set_params_manually(
            weight_min_val, weight_max_val, 8
        )
        c_q.get_act_quantizer().set_params_manually(act_min_val, act_max_val, 8)

        out1 = c(input)
        out2 = c_q(input)
        # 8bit quant: tensor(-0.1381, grad_fn=<SumBackward0>)
        print("8bit quant:", (out1 - out2).sum())

        # import matplotlib.pyplot as plt

        # plt.subplot(1, 2, 1)
        # plt.hist(c.weight.reshape(-1).detach().numpy(), bins=40)
        # plt.subplot(1, 2, 2)
        # plt.hist(c_q.get_quant_weight_bias()[0].reshape(-1).detach().numpy(), bins=40)
        # plt.show()

        c_q.get_weight_quantizer().set_params_manually(
            weight_min_val, weight_max_val, 16
        )
        c_q.get_act_quantizer().set_params_manually(act_min_val, act_max_val, 16)

        out1 = c(input)
        out2 = c_q(input)
        # 8bit quant: tensor(-0.1381, grad_fn=<SumBackward0>)
        print("16bit quant:", (out1 - out2).sum())

    # quant_sim_test()

    def replace_test():

        class test_module(Layer):
            def __init__(self):
                super().__init__()

                self.c1 = Conv2D(3, 16, 3)
                self.act = ReLU()
                self.l = Linear(2704, 15)

            def forward(self, x):
                x = self.c1(x)
                x = self.act(x).view(x.size(0), -1)
                print(x.shape)
                x = self.l(x)

                return x

        input = paddle.randn((16, 3, 15, 15))

        m = test_module()
        out1 = m(input)

        for name, module in m.named_modules():
            if isinstance(module, Linear):
                new_module = QuantLinear()
                new_module.set_param(module)
                module = new_module
                module.set_quant_flag(True)
            elif isinstance(module, Conv2D):
                new_module = QuantConv2d()
                new_module.set_param(module)
                module = new_module
                module.set_quant_flag(True)

        # for name, module in m.named_modules():
        #     if isinstance(module, Linear):
        #         print(1)
        #     elif isinstance(module, Conv2d):
        #         print(2)
        print(m)

        out2 = m(input)

        print("8bit quant:", (out1 - out2).sum())
    # replace_test()
    def linear_shape_test():
        l = Linear(2, 6, True)
        print(l.weight)
        print(l.weight.size())
        print(l.bias.size())
        print(l.weight.reshape(3,2,2))
        a,b,c = l.weight.reshape(3,2,2)
        print(a,b,c)
    # linear_shape_test()
    
    def qkv_equal_test():
        paddle.seed(3407)
        config = {'param':1}
        in_feature = 30
        out_feature = 90
        l = Linear(3, 9, False)
        l_q = QuantLinearQKV(config)
        l_qq = QuantLinear(config)
        print(l.weight.size())
        
        l_qq.set_param(l)
        l_q.set_param(l)
        
        l_qq.set_quant_flag(False)
        l_q.q.set_quant_flag(False)
        l_q.k.set_quant_flag(False)
        l_q.v.set_quant_flag(False)
        
        input = paddle.randn(5, 3)
        
        out1 = l(input)
        out2, q, k ,v = l_q(input)
        out3 = l_qq(input)
        
        q_1, k_1, v_1 = out1.reshape(5,3,3).permute(1,0,2)
        
        print(out1.size())
        print(out2.size())
        print(out3.size())
        print(F.l1_loss(out1, out2))
        print(F.l1_loss(out1, out3))
        print(F.l1_loss(q, q_1))
        print(F.l1_loss(q, k_1))
        print(F.l1_loss(q, v_1))
        print(F.l1_loss(k, k_1))
        print(F.l1_loss(v, v_1))
        
        weight_raw = l.weight.data
        weight_q = l_q.q.weight
        weight_k = l_q.k.weight
        weight_v = l_q.v.weight
        print('weight loss')
        print(F.l1_loss(weight_q, weight_raw[:3,:]))
        print(F.l1_loss(weight_k, weight_raw[3:6,:]))
        print(F.l1_loss(weight_v, weight_raw[6:,:]))
        
        out_ma_1 = input @ weight_q.transpose(0,1)
        print('mannul')
        print(F.l1_loss(out_ma_1, q))
        print(F.l1_loss(out_ma_1, q_1))
        print(F.l1_loss(out_ma_1, out1[:,:3]))
        print(F.l1_loss(out_ma_1, out1[:,3:6]))
        print(F.l1_loss(out_ma_1, out1[:,6:]))
        
        print('linear self')
        out_ma = input @ weight_raw.transpose(0,1) # 这就是linear的计算方式
        out_q_real = out_ma[:,:3] # linear得到的q
        out_q = input @ weight_raw[:3,:].transpose(0,1)
        out_q1 = input @ weight_raw.transpose(0,1)[:,:3]
        out_q2 = input @ weight_raw.reshape(3, 3,3 ).permute(1, 0, 2)[0].transpose(0,1)
        print(F.l1_loss(out_ma, out1))
        print(F.l1_loss(out_q_real, out_q))
        print(F.l1_loss(out_q_real, out_q1))
        print(F.l1_loss(out_q_real, out_q2))
        
        print('detail')
        
        qkv = l(input).reshape(5, 3, 3).permute(1, 0, 2)
        
        q, k, v = (
            qkv[0],
            qkv[1],
            qkv[2],
        )  # make torchscript happy (cannot use tensor as tuple)
        print(F.l1_loss(q, out_q_real))
        print('weight show')
        print(weight_raw.transpose(0,1))
        print(weight_raw[:3,:].transpose(0,1))
        
        print(input@weight_raw.transpose(0,1))
        print((input@weight_raw.transpose(0,1))[:,:3])
        print(input@weight_raw[:3,:].transpose(0,1))
        
        w1 = weight_raw.transpose(0,1)[:,:3]
        w2 = weight_raw[:3,:].transpose(0,1)
        
        print(F.l1_loss(w1, w2))
        print(F.l1_loss(input@w1, input@w2))
        print(F.l1_loss(input@w1, q))
        print(F.l1_loss(input@w2, q))
        
        print(F.l1_loss((input@(weight_raw.transpose(0,1))[:,:3]), input@(weight_raw[:3,:].transpose(0,1))))
        
    # qkv_equal_test()
    
    def qkv_equal_gpt():

        # 定义一个函数来拆分线性层成三个线性层
        def split_linear_layer(linear_layer, split_size):
            in_features = linear_layer.in_features
            out_features = linear_layer.out_features
            q_linear = nn.Linear(in_features, split_size)
            k_linear = nn.Linear(in_features, split_size)
            v_linear = nn.Linear(in_features, split_size)
            q_linear.weight.data = linear_layer.weight.data[:split_size, :]
            k_linear.weight.data = linear_layer.weight.data[split_size:2*split_size, :]
            v_linear.weight.data = linear_layer.weight.data[2*split_size:, :]
            q_linear.bias.data = linear_layer.bias.data[:split_size]
            k_linear.bias.data = linear_layer.bias.data[split_size:2*split_size]
            v_linear.bias.data = linear_layer.bias.data[2*split_size:]
            return q_linear, k_linear, v_linear

        # 创建一个示例线性层
        linear_layer = nn.Linear(100, 300)

        # 将线性层拆分成三个线性层
        q_linear, k_linear, v_linear = split_linear_layer(linear_layer, 100)

        # 创建一些输入数据
        batch_size = 16
        input_data = paddle.randn(batch_size, 100, dtype=paddle.float64)  # 使用 torch.float64

        # 使用拆分前的线性层进行前向传播
        output_before_split = linear_layer(input_data)

        # 使用拆分后的线性层进行前向传播
        q_output, k_output, v_output = q_linear(input_data), k_linear(input_data), v_linear(input_data)
        output_after_split = paddle.concat([q_output, k_output, v_output], axis=1)

        # 计算 L1 损失来验证结果一致性
        l1_loss = nn.L1Loss()
        loss = l1_loss(output_before_split, output_after_split)

        print("L1 损失:", loss.item())

    # qkv_equal_gpt()
    
    def param_test():
        paddle.seed(3407)
        
        config = {'param':1}
        l_raw = Linear(10,20)
        l_q = QuantLinear(config)
        l_q.set_param(l_raw)
        
        input = paddle.randn(5, 10)
        
        l_q.set_quant_flag(False)
        
        out1 = l_raw(input)
        out2 = l_q(input)
        
        print(F.l1_loss(out1, out2))
        
        # print(l_q.named_parameters())
        for name, param in l_q.named_parameters():
            print(name)
        
        
    param_test()