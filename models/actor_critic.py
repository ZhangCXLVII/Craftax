import jax
import jax.numpy as jnp
import flax.linen as nn
import numpy as np
from flax.linen.initializers import constant, orthogonal
from typing import Sequence

import distrax



class ActorCriticConv(nn.Module):
    action_dim: Sequence[int]
    layer_width: int
    activation: str = "tanh"

    @nn.compact
    def __call__(self, obs):
        x = nn.Conv(features=32, kernel_size=(5, 5))(obs)
        x = nn.relu(x)
        x = nn.max_pool(x, window_shape=(3, 3), strides=(3, 3))
        x = nn.Conv(features=32, kernel_size=(5, 5))(x)
        x = nn.relu(x)
        x = nn.max_pool(x, window_shape=(3, 3), strides=(3, 3))
        x = nn.Conv(features=32, kernel_size=(5, 5))(x)
        x = nn.relu(x)
        x = nn.max_pool(x, window_shape=(3, 3), strides=(3, 3))

        embedding = x.reshape(x.shape[0], -1)

        actor_mean = nn.Dense(
            self.layer_width, kernel_init=orthogonal(2), bias_init=constant(0.0)
        )(embedding)
        actor_mean = nn.relu(actor_mean)

        actor_mean = nn.Dense(
            self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0)
        )(actor_mean)
        actor_mean = nn.relu(actor_mean)

        actor_mean = nn.Dense(
            self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0)
        )(actor_mean)

        pi = distrax.Categorical(logits=actor_mean)

        critic = nn.Dense(
            self.layer_width, kernel_init=orthogonal(2), bias_init=constant(0.0)
        )(embedding)
        critic = nn.relu(critic)
        critic = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(
            critic
        )

        return pi, jnp.squeeze(critic, axis=-1)


class ActorCritic_mlp(nn.Module):
    action_dim: Sequence[int]
    layer_width: int
    activation: str = "tanh"

    @nn.compact
    def __call__(self, x):
        if self.activation == "relu":
            activation = nn.relu
        else:
            activation = nn.tanh

        actor_mean = nn.Dense(
            self.layer_width,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )(x)
        actor_mean = activation(actor_mean)

        actor_mean = nn.Dense(
            self.layer_width,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )(actor_mean)
        actor_mean = activation(actor_mean)

        actor_mean = nn.Dense(
            self.layer_width,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )(actor_mean)
        actor_mean = activation(actor_mean)

        actor_mean = nn.Dense(
            self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0)
        )(actor_mean)
        pi = distrax.Categorical(logits=actor_mean)

        critic = nn.Dense(
            self.layer_width,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )(x)
        critic = activation(critic)

        critic = nn.Dense(
            self.layer_width,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )(critic)
        critic = activation(critic)

        critic = nn.Dense(
            self.layer_width,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )(critic)
        critic = activation(critic)

        critic = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(
            critic
        )

        return pi, jnp.squeeze(critic, axis=-1)


class TransformerActor(nn.Module):
    action_dim: int
    layer_width: int
    num_heads: int = 2  # Transformer 头数
    num_layers: int = 2  # Transformer 层数
    activation: str = "tanh"

    @nn.compact
    def __call__(self, x):
        x = x[:, None, :]  # (batch, seq_len=1, dim)
        
        # **确保输入维度可以被 num_heads 整除**
        proj_dim = (x.shape[-1] // self.num_heads) * self.num_heads
        x = nn.Dense(proj_dim)(x)  # 投影到合适维度
        
        for _ in range(self.num_layers):
            x = nn.SelfAttention(num_heads=self.num_heads)(x)
            x = nn.LayerNorm()(x)
            x = nn.Dense(self.layer_width)(x)
            x = nn.tanh(x) if self.activation == "tanh" else nn.relu(x)

        x = nn.Dense(self.action_dim)(x)
        x = jnp.squeeze(x, axis=1)  # 变回 (batch, action_dim)
        return distrax.Categorical(logits=x)


class testBlock_1(nn.Module):
    embed_dim: int
    expansion: int = 2  # 固定扩展因子为 2

    @nn.compact
    def __call__(self, x):
        # x 的形状: (batch, seq_len, embed_dim)
        # 1. 线性投影到 2*embed_dim
        y = nn.Dense(self.expansion * self.embed_dim,
                     kernel_init=orthogonal(np.sqrt(2)),
                     bias_init=constant(0.0))(x)
        # 2. 分割为两部分
        u, v = jnp.split(y, 2, axis=-1)
        # 3. 门控激活: 使用 SiLU (Swish) 激活 v，再与 u 相乘
        gated = u * nn.silu(v)
        # 4. 投影回 embed_dim
        out = nn.Dense(self.embed_dim,
                       kernel_init=orthogonal(np.sqrt(2)),
                       bias_init=constant(0.0))(gated)
        # 5. 残差连接
        return x + out


class Test_1(nn.Module):
    action_dim: int
    layer_width: int
    num_layers: int = 8
    activation: str = "tanh"  

    @nn.compact
    def __call__(self, x):
        # 输入 x 的形状为 (batch, input_dim)
        # 扩展序列维度，变为 (batch, seq_len=1, input_dim)
        x = x[:, None, :]
        # 投影到ayer_width
        x = nn.Dense(self.layer_width,
                     kernel_init=orthogonal(np.sqrt(2)),
                     bias_init=constant(0.0))(x)
        # 堆叠
        for _ in range(self.num_layers):
            x = MambaBlock(embed_dim=self.layer_width)(x)
            x = nn.LayerNorm()(x)
        # 最后映射到动作维度
        x = nn.Dense(self.action_dim,
                     kernel_init=orthogonal(0.01),
                     bias_init=constant(0.0))(x)
        # 去除 seq_len 维度，得到 (batch, action_dim)
        x = jnp.squeeze(x, axis=1)
        return distrax.Categorical(logits=x)



# ------------------- RMSNorm -------------------
class RMSNorm(nn.Module):
    d_model: int
    eps: float = 1e-5

    @nn.compact
    def __call__(self, x):
        norm = jnp.sqrt(jnp.mean(x**2, axis=-1, keepdims=True) + self.eps)
        return x / norm

# ------------------- MambaBlock -------------------
class MambaBlock(nn.Module):
    d_model: int      # 模型输入输出维度
    d_inner: int      # 内部隐藏维度（投影后维度）
    d_conv: int       # 卷积核大小
    dt_rank: int      # 投影到 Δ 的维度
    d_state: int      # SSM 状态维度（n）
    conv_bias: bool = True
    bias: bool = True

    @nn.compact
    def __call__(self, x):
        # x: (b, l, d_model)
        b, l, _ = x.shape

        # 1. RMSNorm 预处理
        x_norm = RMSNorm(self.d_model)(x)

        # 2. in_proj: 投影到 2 * d_inner
        x_and_res = nn.Dense(self.d_inner * 2,
                             kernel_init=orthogonal(np.sqrt(2)),
                             bias_init=nn.initializers.constant(0.0) if self.bias else None)(x_norm)
        # 分割成 x_part 和 res，形状均 (b, l, d_inner)
        x_part, res = jnp.split(x_and_res, 2, axis=-1)

        # 3. 对 x_part 直接进行 1D 卷积
        conv_out = nn.Conv(features=self.d_inner,
                           kernel_size=(self.d_conv,),
                           padding='SAME',
                           feature_group_count=self.d_inner,
                           use_bias=self.conv_bias,
                           kernel_init=orthogonal(np.sqrt(2)),
                           bias_init=nn.initializers.constant(0.0) if self.conv_bias else None
                          )(x_part)
        # 4. SiLU 激活得到 u
        u = jax.nn.silu(conv_out)

        # 5. selective SSM: 调用 ssm() 函数，得到 y_ssm
        y_ssm = self.ssm(x_part, u)  # 注意：x_part用于生成投影参数，u作为 selective SSM 输入

        # 6. 对 res 直接 SiLU 激活
        res_act = jax.nn.silu(res)
        # 7. 门控：y = y_ssm * res_act
        y = y_ssm * res_act

        # 8. out_proj: 投影回 d_inner
        y = nn.Dense(self.d_inner,
                     kernel_init=orthogonal(np.sqrt(2)),
                     bias_init=nn.initializers.constant(0.0) if self.bias else None)(y)
        # 9. 残差连接：加上原始 x_part
        y = x_part + y
        # 10. RMSNorm 后处理
        y = RMSNorm(self.d_inner)(y)
        # 11. 最后 Linear Project 到 d_model，并 softmax
        y = nn.Dense(self.d_model,
                     kernel_init=orthogonal(1.0),
                     bias_init=nn.initializers.constant(0.0) if self.bias else None)(y)
        y = nn.softmax(y, axis=-1)
        return y

    def ssm(self, x, u):
        """
        selective SSM 实现，参照 PyTorch 代码。
        Args:
            x: (b, l, d_inner) 原始投影后的 x 部分，用于生成选择性参数。
            u: (b, l, d_inner) 经过卷积和激活后的 u。
        Returns:
            y: (b, l, d_inner) selective SSM 输出
        """
        b, l, d_in = x.shape
        n = self.d_state  # 状态维度

        # 计算 x_proj: 将 x 投影到 (dt_rank + 2*n) 维
        x_proj_out = nn.Dense(self.dt_rank + 2 * n,
                              use_bias=False,
                              kernel_init=orthogonal(np.sqrt(2)))(x)  # (b, l, dt_rank + 2*n)
        # 分割得到 delta, B, C
        delta, B, C = jnp.split(x_proj_out, [self.dt_rank, self.dt_rank + n], axis=-1)
        # dt_proj: 将 delta 投影到 d_in
        delta = nn.Dense(d_in,
                         use_bias=True,
                         kernel_init=orthogonal(np.sqrt(2)))(delta)
        # softplus 激活
        delta = jax.nn.softplus(delta)  # (b, l, d_in)

        # selective_scan 实现，参考 PyTorch 代码
        y = self.selective_scan(u, delta, B, C)
        return y

    def selective_scan(self, u, delta, B, C):
        """
        selective_scan 实现：
        对于时间步 t, 更新状态：
            x_state = deltaA[t] * x_state + deltaB_u[t]
            y[t] = einsum(x_state, C[t])
        最后 y = y + u * D，其中 D 为可训练参数
        """
        b, l, d_in = u.shape
        n = self.d_state

        # A_log: (d_in, n)
        A_log = self.param('A_log', orthogonal(np.sqrt(2)), (d_in, n))
        # D: (d_in,)
        D = self.param('D', nn.initializers.ones, (d_in,))
        # A = -exp(A_log)
        A = -jnp.exp(A_log)  # (d_in, n)

        # deltaA: (b, l, d_in, n) = exp(einsum(delta, A))
        deltaA = jnp.exp(jnp.einsum('bld,dn->bldn', delta, A))
        # deltaB_u: (b, l, d_in, n) = einsum(delta, B, u)
        deltaB_u = jnp.einsum('bld,bln,bld->bldn', delta, B, u)

        def scan_fn(carry, inputs):
            # carry: x_state, shape (b, d_in, n)
            # inputs: (deltaA_t, deltaB_u_t, C_t)
            deltaA_t, deltaB_u_t, C_t = inputs  # deltaA_t, deltaB_u_t: (b, d_in, n); C_t: (b, n)
            new_state = deltaA_t * carry + deltaB_u_t  # (b, d_in, n)
            # y_t: (b, d_in) = einsum(new_state, C_t)
            y_t = jnp.einsum('bdn,bn->bd', new_state, C_t)
            return new_state, y_t

        # 初始化 x_state: (b, d_in, n) zeros
        init_state = jnp.zeros((b, d_in, n))
        # 准备 scan 输入：沿时间维度 l
        # deltaA: (b, l, d_in, n) -> (l, b, d_in, n)
        # deltaB_u: (b, l, d_in, n) -> (l, b, d_in, n)
        # C: (b, l, n) -> (l, b, n)
        scan_inputs = (jnp.transpose(deltaA, (1, 0, 2, 3)),
                       jnp.transpose(deltaB_u, (1, 0, 2, 3)),
                       jnp.transpose(C, (1, 0, 2)))
        # 使用 jax.lax.scan 进行递归计算
        final_state, ys = jax.lax.scan(scan_fn, init_state, scan_inputs)
        # ys: (l, b, d_in) -> 转换为 (b, l, d_in)
        ys = jnp.transpose(ys, (1, 0, 2))
        # 最后加上 u * D (广播 D)
        y_out = ys + u * D
        return y_out

# ------------------- MambaActor -------------------
class MambaActor(nn.Module):
    action_dim: int
    layer_width: int
    num_layers: int = 2  # 两层 MambaBlock

    @nn.compact
    def __call__(self, x):
        # x: (b, d_model) 输入为向量
        # 扩展序列维度为 (b, 1, d_model)
        x = x[:, None, :]
        # 初始投影到 layer_width
        x = nn.Dense(self.layer_width,
                     kernel_init=orthogonal(np.sqrt(2)),
                     bias_init=nn.initializers.constant(0.0))(x)
        # 堆叠 num_layers 层 MambaBlock
        for _ in range(self.num_layers):
            x = MambaBlock(d_model=self.layer_width,
                           d_inner=self.layer_width,
                           d_conv=3,
                           dt_rank=4,      # 可调
                           d_state=16,     # 可调
                           conv_bias=True,
                           bias=True)(x)
        # 映射到动作维度
        x = nn.Dense(self.action_dim,
                     kernel_init=orthogonal(0.01),
                     bias_init=nn.initializers.constant(0.0))(x)
        # 去掉序列维度，得到 (b, action_dim)
        x = jnp.squeeze(x, axis=1)
        return distrax.Categorical(logits=x)

class ActorCritic(nn.Module):
    action_dim: int
    layer_width: int
    activation: str = "tanh"

    @nn.compact
    def __call__(self, x):
        #pi = TransformerActor(self.action_dim, self.layer_width, activation=self.activation)(x)
        pi = MambaActor(self.action_dim, self.layer_width)(x)


        # Critic 结构保持不变
        if self.activation == "relu":
            activation = nn.relu
        else:
            activation = nn.tanh

        critic = nn.Dense(
            self.layer_width,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )(x)
        critic = activation(critic)

        critic = nn.Dense(
            self.layer_width,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )(critic)
        critic = activation(critic)

        critic = nn.Dense(
            self.layer_width,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )(critic)
        critic = activation(critic)

        critic = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(
            critic
        )

        return pi, jnp.squeeze(critic, axis=-1)