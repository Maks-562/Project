import jax
import jax.numpy as jnp
from jax import grad, jit, vmap, lax
import equinox as eqx
import optax
import pandas as pd
import numpy as np

# ---------- DMN model as an Equinox Module ----------
class DMN(eqx.Module):
    theta: jax.Array
    activation: jax.Array
    weight: jax.Array
    fractions: jax.Array 
    left: jax.Array = eqx.field(static=True)
    right: jax.Array = eqx.field(static=True)
    layer: jax.Array = eqx.field(static=True)

    def __call__(self, phase1, phase2):
        num_nodes = self.theta.shape[0]
        compliance = jnp.zeros((num_nodes, 6))
        
        ws = self.weight
        fs = self.fractions
        
        def propagate_weights(i,ws):
            
            j = num_nodes-i
            
            l = self.left[j]
            r = self.right[j]
           
            act = self.activation
         
            is_leaf = (l == -1) & (r == -1)
           
            
            def handle_leaf(_):
                return relu(act[j])
                
            def handle_inner(_):
                return ws[l] + ws[r]  
            
            updated = lax.cond(is_leaf, handle_leaf, handle_inner, operand=None)
           
            return ws.at[j].set(updated)
        
        # propagete the volume fractions f
        def propagate_fs(i,fs):
            j = num_nodes-i
            l = self.left[j]
            r = self.right[j]
            
            def handle_leaf(_):
                return 0.0
            def handle_inner(_):
                return ws[l]/ws[j]
                
            is_not_leaf =  ~((l == -1) & (r == -1))
            
            updated = lax.cond(is_not_leaf, handle_inner,handle_leaf, operand=None)
            fs = fs.at[l].set(updated)
            fs = fs.at[r].set(updated)
            return  fs
        
        def body_fun(i, compliance):
            j = num_nodes - i
            l = self.left[j]
            r = self.right[j]
            theta_i = self.theta[j]
            fs = self.fractions

            is_leaf = (l == -1) & (r == -1)
           
            def handle_leaf(_):
                is_even = (j % 2) == 0
                base = lax.cond(is_even, lambda _: phase1, lambda _: phase2, operand=None)
                return rotate(base, theta_i)


            def handle_inner(_):
                D1 = compliance[l]
                D2 = compliance[r]
                f1 = fs[l]
                f2 = fs[r]
               
                D_h = homogenise(D1, D2, f1, f2)
                return rotate(D_h, theta_i)

            updated = lax.cond(is_leaf, handle_leaf, handle_inner, operand=None)
            return compliance.at[i].set(updated)
         
        ws = lax.fori_loop(0,num_nodes+1, propagate_weights, ws)
        fs = lax.fori_loop(5,num_nodes+1, propagate_fs, fs)
        compliance = lax.fori_loop(0,num_nodes, body_fun, compliance)
       
      
        return compliance[-1]

# ---------- Utility functions (ported to JAX) ----------

@jit
def relu(x):
    return jnp.maximum(0, x)

@jit
def relu_prime(x):
    return jnp.where(x > 0, 1.0, 0.0)

@jit
def R(theta):
    c = jnp.cos(theta)
    s = jnp.sin(theta)
    root2 = jnp.sqrt(2.0)
    return jnp.array([
        [c**2, s**2, root2 * c * s],
        [s**2, c**2, -root2 * c * s],
        [-root2 * s * c, root2 * s * c, c**2 - s**2]
    ])

@jit
def convert_to_matrix(v):
    return jnp.array([
        [v[0], v[1], v[2]],
        [v[1], v[3], v[4]],
        [v[2], v[4], v[5]]
    ])

@jit
def convert_to_vector(M):
    return jnp.array([M[0, 0], M[0, 1], M[0, 2], M[1, 1], M[1, 2], M[2, 2]])

@jit
def rotate(D, theta):
    Dmat = convert_to_matrix(D)
    Rm = R(theta)
    Rn = R(-theta)
    Drot = Rn @ Dmat @ Rm
    return convert_to_vector(Drot)

@jit
def homogenise(D1, D2, f1, f2):
    gamma = f1 * D2[0] + f2 * D1[0]
    Dr = jnp.zeros(6)
    Dr = Dr.at[0].set((D1[0] * D2[0]) / gamma)
    Dr = Dr.at[1].set((f1 * D1[1] * D2[0] + f2 * D2[1] * D1[0]) / gamma)
    Dr = Dr.at[2].set((f1 * D1[2] * D2[0] + f2 * D2[2] * D1[0]) / gamma)
    Dr = Dr.at[3].set(f1 * D1[3] + f2 * D2[3] - (1/gamma) * (f1*f2) * (D1[1] - D2[1])**2)
    Dr = Dr.at[4].set(f1 * D1[4] + f2 * D2[4] - (1/gamma) * (f1*f2) * (D1[2] - D2[2]) * (D1[1] - D2[1]))
    Dr = Dr.at[5].set(f1 * D1[5] + f2 * D2[5] - (1/gamma) * (f1*f2) * (D1[2] - D2[2])**2)
    return Dr

# ---------- Cost and Training Functions ----------

@jit
def cost(D_dns, D_out):
    norm_factor = jnp.linalg.norm(convert_to_matrix(D_dns))
    diff = D_dns - D_out
    diff = diff.at[1].set(diff[1] * 2)
    diff = diff.at[2].set(diff[2] * 2)
    diff = diff.at[4].set(diff[4] * 2)
    return jnp.sum(diff**2) / norm_factor**2

@jit
def error_relative(D_dns, D_out):
    return jnp.linalg.norm(convert_to_matrix(D_dns) - convert_to_matrix(D_out)) / jnp.linalg.norm(convert_to_matrix(D_dns))

@jit
def loss_fn(model, phase1, phase2, D_dns):
    D_out = model(phase1, phase2)
    return cost(D_dns, D_out)

batched_loss_fn = vmap(loss_fn, in_axes=(None, 0, 0, 0))

# ---------- Training Step with Batching ----------

def make_training_step(opt, opt_state, model, phases1, phases2, D_targets):
    def total_loss(model):
        return jnp.mean(batched_loss_fn(model, phases1, phases2, D_targets))

    loss, grads = jax.value_and_grad(total_loss)(model)
    updates, opt_state = opt.update(grads, opt_state, model)
    model = eqx.apply_updates(model, updates)
    return model, opt_state, loss

make_training_step = jit(make_training_step, static_argnums=0)

# ---------- Load Data from CSV ----------

data = pd.read_csv("data/optimised2_data.csv", header=None).values
phase1 = jnp.array(data[:200, 0:6])
phase2 = jnp.array(data[:200, 6:12])
D_dns = jnp.array(data[:200, 12:18])
num_samples = phase1.shape[0]
# ---------- Training Loop ----------
key = jax.random.PRNGKey(0)
dmn = DMN(
    theta=jax.random.uniform(key, (7,),float,-jnp.pi, jnp.pi),
    activation=jnp.array([0, 0, 0, 0.3, 0.22, 0.28, 0.56]),
    weight=jnp.ones((7,)),
    fractions=jnp.ones((7,)),
    left=jnp.array([1, 3, 5, -1, -1, -1, -1]),
    right=jnp.array([2, 4, 6, -1, -1, -1, -1]),
    layer=jnp.array([0, 1, 1, 2, 2, 2, 2])
)

batch_size = 20

opt = optax.adam(learning_rate=0.015)
opt_state = opt.init(dmn)

for epoch in range(1000):
    key,subkey = jax.random.split(key)
    indxs = jax.random.permutation(key,num_samples)
   
    phase1 = phase1[indxs]
    phase2 = phase2[indxs]
    D_dns = D_dns[indxs]

    for batch_index in range(0,num_samples, batch_size):
        phase_1_batch = phase1[batch_index:batch_index + batch_size]
        phase2_batch = phase2[batch_index:batch_index + batch_size]
        D_dns_batch = D_dns[batch_index:batch_index + batch_size]
        dmn, opt_state, loss = make_training_step(opt, opt_state, dmn, phase_1_batch, phase2_batch, D_dns_batch)
    
    if epoch % 10 == 0:
        example_out = dmn(phase1[0], phase2[0])
        rel_err = error_relative(D_dns[0], example_out)
        print(f"Epoch {epoch}, Loss: {loss:.6f}, Rel Error: {rel_err*100:.4f} %")

D_out = dmn(phase1[1], phase2[1])
print("Final Output:", D_out)
print("Target Output:", D_dns[1])