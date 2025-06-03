import jax
import jax.numpy as jnp
from jax import grad, jit, vmap, lax
import equinox as eqx
import optax
import pandas as pd
from matplotlib import pyplot as plt
import math

# ---------- DMN model as an Equinox Module ----------
class DMN(eqx.Module):
    theta: jax.Array
    activation: jax.Array
    left: jax.Array = eqx.field(static=True)
    right: jax.Array = eqx.field(static=True)

    def __init__(self,N,key=None):
        num_nodes = 2**(N+1) - 1
        
        if key is None:
            key = jax.random.PRNGKey(0)  
       
        self.theta = jax.random.uniform(key, (num_nodes,), float, -jnp.pi, jnp.pi)
        self.activation = jnp.zeros((num_nodes,)).at[2**N-1:].set(jax.random.uniform(key, (2**N,), float, 0.2, 0.8))
        left = -jnp.ones((num_nodes,), dtype=jnp.int32)
        right = -jnp.ones((num_nodes,), dtype=jnp.int32)

        for i in range(num_nodes - 2**N):
            l = 2 * i + 1
            r = 2 * i + 2
            left = left.at[i].set(l)
            right = right.at[i].set(r)
        self.left = left
        self.right = right
        


    def __call__(self, phase1, phase2):
        num_nodes = self.theta.shape[0]
        
        N = int(math.log2(num_nodes + 1)) - 1
        start = 2 ** N
    
        compliance = jnp.zeros((num_nodes, 6))
        ws = jnp.zeros((num_nodes,))
        fs = jnp.zeros((num_nodes,))
       

        
        def propagate_weights(i,ws):
            
            j = num_nodes - i - 1
            
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
            j = num_nodes-i -1
            l = self.left[j]
            r = self.right[j]
            
            def handle_leaf(_):
                return 0.0
            def handle_inner(_):
                return ws[l]/ws[j]
                
            is_not_leaf =  ~((l == -1) & (r == -1))
            
            updated = lax.cond(is_not_leaf, handle_inner,handle_leaf, operand=None)
            fs = fs.at[l].set(updated)
            fs = fs.at[r].set(1-updated)
        
            return  fs
        
        def body_fun(i, compliance):
            j = num_nodes - i -1 
            l = self.left[j]
            r = self.right[j]
            theta_i = self.theta[j]
            
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
            compliance = compliance.at[j].set(updated)
    
            return compliance.at[j].set(updated)
        
        ws = lax.fori_loop(0,num_nodes, propagate_weights, ws)
        fs = lax.fori_loop( start,num_nodes, propagate_fs, fs)
        compliance = lax.fori_loop(0,num_nodes, body_fun, compliance)

       
      
        return compliance[0]

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

dmn = DMN(N=5, key=key)

batch_size = 20
epoch_num = 5000
opt = optax.sgd(learning_rate=0.05)
opt_state = opt.init(dmn)

loss_ = []
for epoch in range(epoch_num):
    key,subkey = jax.random.split(key)
    # indxs = jax.random.permutation(key,num_samples)
   
    # phase1 = phase1[indxs]
    # phase2 = phase2[indxs]
    # D_dns = D_dns[indxs]

    for batch_index in range(0,num_samples, batch_size):
        phase_1_batch = phase1[batch_index:batch_index + batch_size]
        phase2_batch = phase2[batch_index:batch_index + batch_size]
        D_dns_batch = D_dns[batch_index:batch_index + batch_size]
        dmn, opt_state, loss = make_training_step(opt, opt_state, dmn, phase_1_batch, phase2_batch, D_dns_batch)
    loss_.append(loss)
    
    if epoch % 10 == 0:
        example_out = dmn(phase1[0], phase2[0])
        rel_err = error_relative(D_dns[0], example_out)
        print(f"Epoch {epoch}, Loss: {loss:.6f}, Rel Error: {rel_err*100:.4f} %")
for i in range(5):
    print(D_dns[i])
    print(dmn(phase1[i], phase2[i]))

epochs = jnp.arange(0,epoch_num)

# plt.plot(epochs,loss_)
# plt.savefig("DMN_loss.png")
plt.figure(figsize=(8, 5))
plt.loglog(epochs, loss_, marker='o', linestyle='-', color='tab:blue', label='Training Loss')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("DMN Training Loss Curve")
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()
plt.tight_layout()
plt.savefig("DMN_loss.png", dpi=150)