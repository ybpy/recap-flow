"""Exercise large multi-GPU collectives and bf16 training kernels before model loading."""

import json

import jax
from jax import lax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_compilation_cache", val=False)
devices = jax.devices("gpu")
print("devices", list(map(str, devices)), flush=True)
x = jax.device_put_sharded([np.ones((1024, 1024), np.float32) * (i + 1) for i in range(len(devices))], devices)
summed = jax.pmap(lambda a: lax.psum(a, "replicas"), axis_name="replicas")(x)
summed.block_until_ready()
assert np.all(np.asarray(summed) == sum(range(1, len(devices) + 1)))
print("allreduce_fp32_pass", flush=True)


def step(w):
    loss, grad = jax.value_and_grad(lambda a: jnp.mean(jnp.square(a @ a)))(w)
    return loss, w - lax.pmean(grad, "replicas") * 0.001


w = jax.device_put_sharded([np.ones((1024, 1024), np.float32) * 0.001 for _ in devices], devices)
w = jax.tree.map(lambda a: a.astype(jnp.bfloat16), w)
f = jax.pmap(step, axis_name="replicas")
for _i in range(3):
    loss, w = f(w)
    loss.block_until_ready()
    assert np.isfinite(np.asarray(loss)).all()
print(
    json.dumps({"complete": True, "devices": len(devices), "loss": np.asarray(loss).astype(float).tolist()}), flush=True
)
