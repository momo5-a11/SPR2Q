import torch
import torch.nn as nn
import math
from quamba.qLinearLayer import QLinearLayer
from quamba.qSelectiveScan import QSScan
from quamba.qActLayer import QActLayer

class QuantizedSelectiveScan(nn.Module):
    def __init__(self, d_model, d_state=16, expand=1., dt_rank="auto", q_config=None, **kwargs):
        super().__init__()
        if q_config is None:
            raise ValueError("q_config must be provided for QuantizedSelectiveScan.")
        self.d_model = d_model
        self.d_state = d_state
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_inner / 16) if dt_rank == "auto" else dt_rank
        self.x_proj = QLinearLayer(self.d_inner, self.dt_rank + self.d_state * 2, bias=False, w_qconfig=q_config['weight'])
        self.dt_proj = QLinearLayer(self.dt_rank, self.d_inner, bias=True, w_qconfig=q_config['weight'])
        self.selective_scan = QSScan(d_inner=self.d_inner, d_state=self.d_state, w_qconfig=q_config['ssm'], a_qconfig=q_config['activation'])
        self.act_quant_u = QActLayer(q_config['activation'])
        self.act_quant_dt = QActLayer(q_config['activation'])
        self.act_quant_B = QActLayer(q_config['activation'])
        self.act_quant_C = QActLayer(q_config['activation'])

    def forward(self, x: torch.Tensor, prompt: torch.Tensor):
        B, L, C = x.shape
        x_dbl = self.x_proj(x)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        dts = self.dt_proj(dts)
        u_q = self.act_quant_u(x)
        dt_q = self.act_quant_dt(dts)
        B_q = self.act_quant_B(Bs)
        C_q = self.act_quant_C(Cs + prompt)
        out_y = self.selective_scan(u_q, dt_q, B_q, C_q)
        return out_y