# Tutorial 0: Environment Setup and PyTorch Basics

**CSE 8803 — Language Models and Language Agents**
**Georgia Tech, Fall 2026**

---

This tutorial is designed to be completed during Week 1, before HW1 is released on Tue Sep 1. Week 1 has no graded deliverable for exactly this reason: get your environment working, get PACE-ICE access, and form your project team. It ensures you have a working development environment and basic PyTorch proficiency before we begin implementing language models from scratch in HW1.

**Estimated time:** 2-3 hours (Part A: 45 min, Part B: 60 min, Part C: 30 min, Part D: 15 min)

**Submission:** No formal submission. However, you should have a working environment by **Fri Aug
28** --- HW1 is released Tue Sep 1, so start Part A on day one.

**Do the laptop-side work first; it is not blocked on the cluster.** Parts B, C, and D are plain
PyTorch and run anywhere --- your own machine, Colab, or ICE. The ICE and laptop commands differ in
two places: where the conda environment is stored (A.2) and which PyTorch wheel you install (A.3).
Only **A.1 and A.4-A.6 genuinely require ICE**.

That ordering is deliberate. Course accounts are provisioned from Banner enrollment and may not all
appear at the same time during the opening week. If ICE login or Open OnDemand is not available yet,
that does not block the tutorial: complete the laptop-side work, then return to the ICE steps when
your course account appears.

---

## Part A: Environment Setup (Self-Guided)

### A.1 PACE-ICE Cluster Access

Georgia Tech's Partnership for an Advanced Computing Environment (PACE) provides GPU resources for this course through its Instructional Cluster Environment, **PACE-ICE**. All homework and project training should be done on PACE-ICE. ICE is shared across courses: treat queue time as part of your budget.

**How you get access: you do not request it.**

There is nothing to sign up for, and you must **not** fill out the College of Computing ICE access
form. That form says so itself: *"If you are a student interested in using ICE for course
assignments, please contact your instructor or TA... Please do not fill out this request for a course
in which you are a student."*

ICE accounts for this course are provisioned **automatically from registrar (Banner/CRN) enrollment**
once the course has been enabled for ICE. It keys off your official registration, not Canvas:

- If you are registered, your access appears on its own. There is no allocation to select and no
  charge account to bill --- if some form or older tutorial asks you for one, you are reading
  Phoenix (research cluster) instructions.
- If you **registered late, swapped sections, or are on a waitlist/permit**, your enrollment may not
  have propagated yet. Post in **Ed Discussion** from the Canvas course navigation so a TA can
  check.
- Access usually shows up within about a day of your enrollment being picked up --- but read the
  first-week caveat below before you conclude something is broken.

> **PACE will not answer you directly.** PACE does not provide support to students in courses that
> use ICE, and **students may not open PACE support tickets.** If you email `pace-support@oit.gatech.edu`
> you will be redirected back to us, a day later. The path is **Ed Discussion -> TA -> instructor
> -> PACE**; the course staff can and will open a ticket on your behalf when the
> problem is actually on the cluster.

> **Expect account provisioning to vary during Week 1, and do not read a missing account as a setup
> mistake.** First connect the GT VPN and try the SSH command below. If your login is rejected or
> Open OnDemand says `Unauthorized`, complete the laptop-side steps (A.2, A.3, Parts B-D) while you
> wait. If you still cannot log in after **Fri Sep 4**, post the exact result in **Ed Discussion**
> from the Canvas course navigation so the course staff can investigate.

**Step 0 --- connect the GT VPN. This is new for Fall 2026 and will catch returning students.**

Since **Aug 13, 2026**, ICE's firewall requires the **GT VPN** for connections from **eduroam and
from the dorms** --- both of which previously worked without it. Wired connections inside campus
buildings remain exempt.

1. Install the **GlobalProtect** client (`vpn.gatech.edu`) and connect *before* you `ssh`. The
   clientless / browser-based VPN is explicitly **not recommended** for PACE services.
2. **Learn this failure mode now:** with no VPN, `ssh` does not report a permission error --- the
   connection just **times out with no message at all**. That is indistinguishable from "my account
   was never created," and in Week 1 both causes are live simultaneously. Before reporting a login
   problem, confirm GlobalProtect is connected and retry; otherwise neither you nor we can tell the
   two apart.
3. If you used ICE in an earlier term, or you are following any guide written before August 2026,
   you will believe eduroam works without the VPN. It does not anymore.

**SSH Access:**

```bash
# Connect to the ICE login node
ssh <your-gt-username>@login-ice.pace.gatech.edu

# Your prompt will look like:  [<your-gt-username>@login-ice-1 ~]$
# This is a shared login node. Do NOT run compute here.
```

> **The host name matters.** `login-ice.pace.gatech.edu` is ICE. `login-pace.pace.gatech.edu` is
> **Phoenix**, a different cluster you probably do not have access to. Most PACE examples you will
> find by searching are Phoenix examples; see the two flags to avoid in A.6.

**File System Layout:**

| Path | Purpose | Quota (approximate) | Backed up? |
|------|---------|---------------------|------------|
| `~/` (home) | Scripts, configs, small files | ~30 GB | Yes --- daily snapshot |
| `~/scratch/` (a symlink PACE creates into the ICE scratch filesystem) | Datasets, tokenized caches, checkpoints, training output | ~300 GB **and** ~1 million files | **No** |
| `/storage/ice-shared/cse8803/` | Optional course-shared cache; use each assignment's checksummed downloader as the authoritative source | Staff announced | No |

> **Tip:** Keep your conda environments, datasets, and checkpoints under `~/scratch/`, not in your
> home directory. 30 GB of home goes fast --- one conda environment with PyTorch in it is several GB.

> **There are two ways to run out of scratch.** The **file-count** limit often bites before the byte
> limit: a few million small files (an unpacked token cache, a checkpoint per step) will start
> failing writes while `du` still looks fine. And scratch is **not backed up** --- untouched files
> are purged, and the purge is enforced at the end of the semester, so copy anything you want to
> keep off the cluster before finals. Treat both numbers above as approximate: PACE adjusts them, so
> check your own usage on the login node before a large download rather than trusting this table.

### A.2 Conda/Mamba Environment

Use the block for the machine you are working on. On a laptop, install
[Miniconda](https://docs.conda.io/projects/miniconda/) first. On ICE, keep the environment under
`~/scratch/` so it does not consume the smaller home-directory quota.

**Laptop:**

```bash
conda create -n cse8803 python=3.11 -y
conda activate cse8803
```

**ICE:**

```bash
module load anaconda3
mkdir -p ~/scratch/conda-envs
conda create --prefix ~/scratch/conda-envs/cse8803 python=3.11 -y
conda activate ~/scratch/conda-envs/cse8803
```

### A.3 PyTorch Installation with CUDA

On **ICE**, install the CUDA 12.6 build so the same environment can run on the older V100 cards:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126

python -c "import torch; print(f'PyTorch {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
```

On your **laptop**, use the command generated by the
[official PyTorch installer](https://pytorch.org/get-started/locally/) for your operating system and
GPU. In particular, current RTX 50-series GPUs may require a newer CUDA wheel (for example, CUDA
12.8) to include kernels for their architecture. A CPU-only laptop is also fine for Parts B-D;
expect `torch.cuda.is_available()` to print `False`.

> **Note:** Your wheel's CUDA version has to be supported by the driver *and* your wheel has to
> contain kernels for the GPU you were allocated. Check both with `nvidia-smi` from inside an
> allocated GPU job --- never from the login node, which has no GPU. We suggest the CUDA 12.6 build
> because it is the last one that still ships kernels for ICE's older V100 (Volta) cards; the
> current default wheels target CUDA 13.x, which **dropped Volta**. If you always pin a newer GPU
> (`-C HX00`, see A.6) the default wheel works too.

### A.4 VS Code Remote SSH (Recommended IDE)

1. Install [VS Code](https://code.visualstudio.com/)
2. Install the "Remote - SSH" extension
3. Add PACE to your SSH config (`~/.ssh/config` on your local machine):

```
Host ice
    HostName login-ice.pace.gatech.edu
    User <your-gt-username>
    ForwardAgent yes
```

4. In VS Code: `Cmd+Shift+P` → "Remote-SSH: Connect to Host" → select `ice`
5. Open your project folder on PACE

> **Tip:** Install the Python and Jupyter extensions on the remote host for full IDE support.

### A.5 Jupyter on ICE (Open OnDemand, or Port Forwarding)

The easy path is **Open OnDemand**, a web portal that requests the job and starts Jupyter or VS Code
for you: <https://ondemand-ice.pace.gatech.edu/>

Three things to know before you rely on it:

- It requires the **GT VPN**.
- It requires an active ICE course account. `Unauthorized` before your account is provisioned is
  expected; it is not a browser or Jupyter error.
- It **does not work well in Safari**. Use Chrome, Firefox, or Edge.
- **Closing the browser tab does not end your job.** The allocation keeps running (and keeps
  consuming the walltime you asked for) until you stop the session in the portal or `scancel` it.

If you prefer to do it from a terminal, forward the port yourself:

```bash
# On ICE (inside a compute node, NOT the login node):
hostname
# Copy the compute-node hostname printed above.
jupyter lab --no-browser --ip=0.0.0.0 --port=8888

# On your local machine (new terminal):
ssh -N -L 8888:<compute-node-hostname>:8888 \
  <your-gt-username>@login-ice.pace.gatech.edu
```

Then open `http://localhost:8888` in your browser and use the token printed by Jupyter. The
destination in `-L` must be the compute node running Jupyter; forwarding to `localhost` on the ICE
login node will not reach the job.

### A.6 Basic SLURM Commands

PACE uses SLURM for job scheduling. Key commands:

| Command | Purpose | Example |
|---------|---------|---------|
| `sbatch` | Submit a batch job | `sbatch train.sh` |
| `squeue` | Check your job queue | `squeue -u $USER` |
| `scancel` | Cancel a job | `scancel <job_id>` |
| `srun` | Run an interactive job | See below |
| `sinfo` | See which nodes and GPU types exist | `sinfo -o "%P %G %N"` |

**Two flags you should NOT pass on ICE:**

- **No `-A` / `--account`.** ICE has no charge accounts. SLURM attaches your college code (`coc`
  for us) by itself, and you never type it. Something like `-A CSE8803-FA26` matches nothing, and
  the job is rejected before it queues.
- **No `-p` / `--partition`.** PACE assigns the partition automatically from your course and from
  whether you asked for a GPU. There is no partition named `gpu`. As a College of Computing course
  we get routed to the higher-priority CoC GPU partition without asking for it.

Both flags *are* required on **Phoenix**, GT's research cluster --- as are QOS names like
`-q inferno` and `-q embers`, which do not exist here. If a command you copied has any of those in
it, it was written for Phoenix, and that is the most common reason a first ICE job fails.

**Interactive GPU session:**

```bash
srun --gres=gpu:1 -C HX00 --mem=32G --time=2:00:00 --pty bash
```

This gives you a shell on a compute node with one GPU, 32 GB RAM, for 2 hours. `--gres=gpu:1` asks
for *any* GPU; `-C HX00` narrows that to "the first available H100 or H200". You can pin one exact
model instead (`--gres=gpu:H100:1`, `--gres=gpu:H200:1`, `--gres=gpu:L40S:1`, `--gres=gpu:A100:1`),
or hand SLURM a list of everything you would accept, which usually queues faster than pinning one:

```bash
srun --gres=gpu:1 -C "H200|H100|L40S|A100-80GB|A100-40GB|A40" --mem=32G --time=2:00:00 --pty bash
```

#### WARNING: do not let SLURM hand you a V100

ICE still has several dozen **V100** cards (Volta, compute capability `sm_70`). Current PyTorch
wheels are built against CUDA 13.x, and **CUDA 13 dropped Volta**: those wheels contain no compiled
kernels for `sm_70`. The failure is the worst kind, because the GPU looks perfectly healthy:

```python
import torch
torch.cuda.is_available()                   # True  <- the GPU is there and visible
x = torch.randn(1024, 1024, device='cuda')
x @ x
# RuntimeError: CUDA error: no kernel image is available for execution on the device
```

`torch.cuda.is_available()` returning `True` while every kernel launch dies means **you landed on a
card your PyTorch build does not support** --- not that your code is wrong. Confirm which card you
got with `nvidia-smi --query-gpu=name --format=csv`. Two fixes:

1. **Ask for a newer GPU.** Add `-C HX00` to your `srun` / `salloc` / `#SBATCH` line, or use the
   OR-list `-C "H200|H100|L40S|A100-80GB|A100-40GB|A40"`. Steer clear of ICE's **MI210** cards as
   well --- those are AMD, and they need a ROCm build of PyTorch rather than a CUDA one.
2. **Or install a PyTorch build that still has Volta kernels** --- the CUDA 12.6 wheels (A.3).

This is why the job commands in this course pin a GPU generation even though pinning can add queue
time: ten extra minutes in the queue is cheaper than an afternoon spent debugging a
`no kernel image` error.

**Batch job script** (`train.sh`):

```bash
#!/bin/bash
#SBATCH --job-name=cse8803-hw1
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --gres=gpu:1
#SBATCH -C HX00
#SBATCH --mem=32G
#SBATCH --time=4:00:00
#SBATCH --cpus-per-task=8

module load anaconda3
conda activate ~/scratch/conda-envs/cse8803

python train.py --config configs/hw1.yaml
```

Submit with:

```bash
mkdir -p logs
sbatch train.sh
```

**Per-job limits, and why asking for more GPUs gives you less time.** A GPU job on ICE is capped
two ways: walltime (on the order of **16 hours**) and **GPU-hours** (also on the order of **16**;
CPU-only jobs get a bit longer). The GPU-hour cap is the one that catches people, because it counts
*GPUs times hours* --- so every extra GPU you request buys you **less** wall-clock time:

| `--gres=gpu:N` | Longest `--time` that fits a 16 GPU-hour cap |
|---|---|
| 1 GPU | 16 hours |
| 2 GPUs | 8 hours |
| 4 GPUs | 4 hours |
| 8 GPUs | 2 hours |

The reflex "request 8 GPUs to be safe" therefore hands you a **2-hour** ceiling, and a single-GPU
training run that would have finished comfortably gets killed instead. For this course: ask for
**one** GPU unless you have measured that you need more, and size `--time` to what the run actually
needs, since an over-long request only sits in the queue longer. Any workload longer than a single
job's cap has to checkpoint and resume across jobs --- which is exactly what HW1 3.4 asks you to
implement. These caps are PACE policy and can change; check the current numbers with
`scontrol show partition` (or the PACE docs) rather than trusting this table forever.

> **Important:** The login node is shared. Never run training or heavy computation there. Always use `srun` or `sbatch`.

---

## Part B: PyTorch Essentials

Work through these exercises in a Jupyter notebook or Python script on a GPU node.

### B.1 Tensor Operations

```python
import torch

# === Creation ===
# From data
x = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float32)

# Common constructors
zeros = torch.zeros(3, 4)
ones = torch.ones(3, 4)
rand = torch.randn(3, 4)          # Normal distribution
arange = torch.arange(0, 10, 2)   # [0, 2, 4, 6, 8]

# Like another tensor (same shape/device/dtype)
x_like = torch.zeros_like(x)

# === Indexing ===
print(x[0])        # First row
print(x[:, 1])    # Second column
print(x[0, 2])    # Element at (0, 2)

# Boolean indexing
mask = x > 3
print(x[mask])     # tensor([4., 5., 6.])

# === Reshaping ===
y = torch.randn(2, 3, 4)
print(y.shape)                    # torch.Size([2, 3, 4])
print(y.view(6, 4).shape)        # torch.Size([6, 4])
print(y.permute(0, 2, 1).shape)  # torch.Size([2, 4, 3])
print(y.reshape(-1).shape)       # torch.Size([24]) -- flatten

# === Broadcasting ===
a = torch.randn(3, 1)   # Shape: (3, 1)
b = torch.randn(1, 4)   # Shape: (1, 4)
c = a + b                # Shape: (3, 4) -- broadcast!

# === Device Placement ===
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
x_gpu = x.to(device)
print(x_gpu.device)  # cuda:0

# Tensors must be on the same device for operations
# x + x_gpu  # ERROR! Mixed devices
x_cpu = x_gpu.to('cpu')  # Move back
```

**Exercise B.1:** Create a (batch_size=8, seq_len=128, d_model=512) tensor of random values. Compute its mean and standard deviation along the last dimension. Verify the output shape is (8, 128).

### B.2 Autograd

```python
import torch

# Autograd tracks operations on tensors with requires_grad=True
x = torch.tensor([2.0, 3.0], requires_grad=True)
y = x ** 2 + 3 * x + 1  # y = x^2 + 3x + 1

# Compute gradients
loss = y.sum()
loss.backward()

# dy/dx = 2x + 3
print(x.grad)  # tensor([7., 9.])  (2*2+3=7, 2*3+3=9)

# === Computation Graph ===
# Each operation builds a graph. .backward() traverses it in reverse.
a = torch.randn(3, requires_grad=True)
b = a * 2
c = b.mean()
print(c.grad_fn)  # MeanBackward0 — shows the last operation

# === Detaching / No Grad ===
# Use torch.no_grad() for inference (saves memory, faster)
with torch.no_grad():
    y_no_grad = x ** 2  # No graph built
    print(y_no_grad.requires_grad)  # False

# .detach() removes a tensor from the graph
z = x.detach()  # z shares data but has no grad tracking

# === Gradient Accumulation Warning ===
# Gradients accumulate by default! Always zero them before .backward()
x = torch.tensor([1.0], requires_grad=True)
for _ in range(3):
    loss = x ** 2
    loss.backward()
print(x.grad)  # tensor([6.]) -- accumulated 2+2+2, not just 2!

# Correct pattern:
x = torch.tensor([1.0], requires_grad=True)
for _ in range(3):
    if x.grad is not None:
        x.grad.zero_()
    loss = x ** 2
    loss.backward()
print(x.grad)  # tensor([2.]) -- correct
```

**Exercise B.2:** Implement gradient descent manually (no optimizer) to minimize f(x) = (x - 3)^2 starting from x=0. Run 100 steps with lr=0.1. Print the final x value (should be close to 3.0).

### B.3 nn.Module Basics

```python
import torch
import torch.nn as nn

class SimpleNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        return x

# Instantiate
model = SimpleNet(input_dim=784, hidden_dim=256, output_dim=10)

# Inspect parameters
print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
for name, param in model.named_parameters():
    print(f"  {name}: {param.shape}")

# state_dict: ordered dict of all parameters
state = model.state_dict()
print(state.keys())

# Save and load
torch.save(state, 'model.pt')
model.load_state_dict(torch.load('model.pt'))
```

### B.4 Simple Training Loop

```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# Synthetic data: y = 2x + 1 + noise
torch.manual_seed(42)
X = torch.randn(1000, 1)
y = 2 * X + 1 + 0.1 * torch.randn(1000, 1)

# DataLoader
dataset = TensorDataset(X, y)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Model
model = nn.Linear(1, 1)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
loss_fn = nn.MSELoss()

# Training loop
for epoch in range(50):
    total_loss = 0.0
    for batch_x, batch_y in dataloader:
        # Forward pass
        pred = model(batch_x)
        loss = loss_fn(pred, batch_y)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    if (epoch + 1) % 10 == 0:
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1:3d} | Loss: {avg_loss:.4f}")

# Check learned parameters (should be ~2 and ~1)
print(f"Weight: {model.weight.item():.3f}, Bias: {model.bias.item():.3f}")
```

### B.5 GPU Training

```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# A self-contained 10-class problem with 784 input features.
# This deliberately does not reuse B.4's one-dimensional regression data.
torch.manual_seed(42)
X_cls = torch.randn(1024, 784)
y_cls = torch.randint(0, 10, (1024,))
classification_loader = DataLoader(
    TensorDataset(X_cls, y_cls), batch_size=32, shuffle=True
)

classification_model = SimpleNet(784, 256, 10).to(device)
classification_optimizer = torch.optim.Adam(
    classification_model.parameters(), lr=1e-3
)
classification_loss_fn = nn.CrossEntropyLoss()

for epoch in range(2):
    total_loss = 0.0
    for batch_x, batch_y in classification_loader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)

        pred = classification_model(batch_x)
        loss = classification_loss_fn(pred, batch_y)

        classification_optimizer.zero_grad()
        loss.backward()
        classification_optimizer.step()
        total_loss += loss.item()

    print(f"Epoch {epoch + 1} | Loss: {total_loss / len(classification_loader):.4f}")

# Common mistake: forgetting to move data to the same device as model
# RuntimeError: Expected all tensors to be on the same device
```

### B.6 Memory Management

Run B.5 first; this section intentionally reuses its `classification_model`,
`classification_loader`, optimizer, loss function, and `device`.

```python
import torch

if device.type == 'cuda':
    print(f"Allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
    print(f"Cached:    {torch.cuda.memory_reserved() / 1e9:.2f} GB")
    torch.cuda.empty_cache()  # Does not free tensors that are still in use.
else:
    print("CUDA is unavailable; GPU memory counters are skipped.")

# === Gradient Accumulation Pattern ===
# When batch size is too large for GPU memory, accumulate gradients
# over multiple smaller batches.

accumulation_steps = 4
classification_optimizer.zero_grad()

for i, (batch_x, batch_y) in enumerate(classification_loader):
    batch_x = batch_x.to(device)
    batch_y = batch_y.to(device)

    # Forward + backward (gradients accumulate)
    loss = classification_loss_fn(classification_model(batch_x), batch_y)
    loss = loss / accumulation_steps  # Normalize
    loss.backward()

    # Step every accumulation_steps batches, including a final partial group.
    if (i + 1) % accumulation_steps == 0 or i + 1 == len(classification_loader):
        classification_optimizer.step()
        classification_optimizer.zero_grad()

# === Memory-Saving Tips ===
# 1. Delete tensors you no longer need
large_tensor = torch.randn(1024, 1024, device=device)
del large_tensor
if device.type == 'cuda':
    torch.cuda.empty_cache()

# 2. Use torch.no_grad() during evaluation
test_data = torch.randn(8, 784, device=device)
with torch.no_grad():
    outputs = classification_model(test_data)
print(outputs.shape)

# 3. Use mixed precision for CUDA training
if device.type == 'cuda':
    scaler = torch.amp.GradScaler("cuda")
    for batch_x, batch_y in classification_loader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)
        classification_optimizer.zero_grad()
        with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
            pred = classification_model(batch_x)
            loss = classification_loss_fn(pred, batch_y)

        scaler.scale(loss).backward()
        scaler.step(classification_optimizer)
        scaler.update()
else:
    print("Mixed precision example skipped because CUDA is unavailable.")
```

**Exercise B.6:** Write a training loop that uses gradient accumulation with `accumulation_steps=4` to simulate a batch size of 128 using actual batches of 32.

---

## Part C: Transformers from the Library (30 min)

This section builds intuition for what HW1 will implement from scratch. Here we use HuggingFace `transformers` as a black box.

```bash
# Install (in your cse8803 environment)
pip install transformers accelerate
```

### C.1 Load a Pre-trained Model

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_name = "gpt2"  # 124M parameters, fits easily on one GPU
tokenizer = AutoTokenizer.from_pretrained(model_name)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = AutoModelForCausalLM.from_pretrained(model_name).to(device)

# Tokenize
text = "The capital of France is"
inputs = tokenizer(text, return_tensors="pt").to(device)
print(f"Token IDs: {inputs['input_ids']}")
print(f"Decoded back: {tokenizer.decode(inputs['input_ids'][0])}")
```

### C.2 Generate Text

```python
# Simple greedy generation
with torch.no_grad():
    outputs = model.generate(
        inputs['input_ids'],
        max_new_tokens=50,
        do_sample=False,  # Greedy
    )
print("Greedy:", tokenizer.decode(outputs[0], skip_special_tokens=True))
```

### C.3 Inspect Model Architecture

```python
# Print the full architecture
print(model)

# Count parameters
total_params = sum(p.numel() for p in model.parameters())
print(f"\nTotal parameters: {total_params:,}")  # ~124M for GPT-2

# Inspect specific layers
for name, param in model.named_parameters():
    if 'h.0' in name:  # First transformer block
        print(f"  {name}: {param.shape}")
```

**Questions to think about:**
- How many transformer blocks does GPT-2 have?
- What is the hidden dimension?
- What are the shapes of the attention weight matrices?
- How many parameters are in the embedding layer vs. the transformer blocks?

### C.4 Compare Decoding Strategies

```python
prompt = "Once upon a time in a land far away,"
inputs = tokenizer(prompt, return_tensors="pt").to(device)

strategies = {
    "Greedy": dict(do_sample=False),
    "Temperature=0.7": dict(do_sample=True, temperature=0.7),
    "Temperature=1.5": dict(do_sample=True, temperature=1.5),
    "Top-k=50": dict(do_sample=True, top_k=50),
    "Top-p=0.9": dict(do_sample=True, top_p=0.9),
    "Top-k=50 + Top-p=0.95": dict(do_sample=True, top_k=50, top_p=0.95),
}

for name, kwargs in strategies.items():
    torch.manual_seed(42)
    with torch.no_grad():
        output = model.generate(
            inputs['input_ids'],
            max_new_tokens=60,
            **kwargs,
        )
    generated = tokenizer.decode(output[0], skip_special_tokens=True)
    print(f"\n{'='*60}")
    print(f"Strategy: {name}")
    print(f"{'='*60}")
    print(generated)
```

**Observe:**
- Greedy decoding is deterministic but can be repetitive
- Higher temperature = more random/creative, but may be incoherent
- Top-k restricts to k most likely tokens at each step
- Top-p (nucleus sampling) restricts to the smallest set of tokens whose cumulative probability exceeds p
- In HW1, you will implement these decoding strategies yourself

---

## Part D: Experiment Tracking (W&B Example)

Homework and project experiments must retain reproducible configurations, metrics, and output
artifacts, but the course does **not** require a particular hosted tracking service. The assignment
scripts' JSON/JSONL logs are sufficient unless a handout says otherwise. TensorBoard and
[Weights & Biases](https://wandb.ai) (W&B) are optional interfaces for comparing runs.

This section demonstrates W&B for students who want a hosted dashboard. You may skip it and still
complete Tutorial 0 and the assignments.

### D.1 Installation and Login

```bash
pip install wandb
wandb login
# Paste your API key from https://wandb.ai/authorize
```

### D.2 Basic Logging

```python
import wandb
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# Initialize a run
wandb.init(
    project="cse8803-tutorial0",
    name="linear-regression-demo",
    config={
        "learning_rate": 0.01,
        "epochs": 50,
        "batch_size": 32,
        "model": "linear",
    }
)

# Synthetic data
torch.manual_seed(42)
X = torch.randn(1000, 1)
y = 2 * X + 1 + 0.1 * torch.randn(1000, 1)
dataloader = DataLoader(TensorDataset(X, y), batch_size=32, shuffle=True)

# Model + training
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
wandb.config.update({"device": str(device)})
model = nn.Linear(1, 1).to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=wandb.config.learning_rate)
loss_fn = nn.MSELoss()

for epoch in range(wandb.config.epochs):
    total_loss = 0.0
    for batch_x, batch_y in dataloader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)
        pred = model(batch_x)
        loss = loss_fn(pred, batch_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    avg_loss = total_loss / len(dataloader)
    
    # Log metrics
    wandb.log({
        "epoch": epoch,
        "train/loss": avg_loss,
        "train/weight": model.weight.detach().cpu().item(),
        "train/bias": model.bias.detach().cpu().item(),
    })

# Finish the run
wandb.finish()
```

### D.3 View Results

After running the above:
1. Go to https://wandb.ai and navigate to your `cse8803-tutorial0` project
2. You should see your run with live-updating charts
3. Explore: loss curve, parameter evolution, CPU/RAM metrics, and GPU utilization when the run uses CUDA

### D.4 Tips for the Course

- **Use `wandb.config`** to log all hyperparameters — makes comparing runs easy
- **Use meaningful run names** like `hw1-gpt2-lr1e4-bs64` instead of auto-generated names
- **Log validation metrics** separately: `wandb.log({"val/loss": ..., "val/perplexity": ...})`
- **Group related runs** using `wandb.init(group="experiment-name")`
- **Save model checkpoints** with `wandb.save("checkpoint.pt")` for reproducibility

---

## Checklist

Before moving on, verify you can:

- [ ] Install and connect the **GlobalProtect GT VPN** (required from eduroam and dorms since
      Aug 13, 2026 --- without it `ssh` silently times out)
- [ ] SSH into ICE (`login-ice.pace.gatech.edu`) and navigate the file system
- [ ] Activate your conda environment with PyTorch + CUDA
- [ ] Request an interactive GPU session with `srun` (no `-A`, no `-p`)
- [ ] Inside that session, check which GPU you were given (`nvidia-smi`) **and** run one real CUDA
      op (e.g. `x = torch.randn(1024, 1024, device='cuda'); x @ x`) --- `torch.cuda.is_available()`
      alone does not prove your PyTorch build supports that card
- [ ] Create tensors, move them to GPU, perform operations
- [ ] Write a training loop with autograd and an optimizer
- [ ] Load a HuggingFace model and generate text
- [ ] Retain experiment metrics using assignment JSON/JSONL logs, TensorBoard, or W&B

If you encounter issues, post in **Ed Discussion** from the Canvas course navigation and
include your error message, the ICE node name, the GPU model from `nvidia-smi`, and your
Python/PyTorch versions. Post there rather than emailing PACE --- students cannot open PACE tickets,
and the course staff can (A.1).

---

## Additional Resources

- [PyTorch Tutorials](https://docs.pytorch.org/tutorials/)
- [PACE Documentation](https://gatech.service-now.com/technology?id=kb_article_view&sysparm_article=KB0042503)
- [HuggingFace Transformers Docs](https://huggingface.co/docs/transformers/index)
- [W&B Quickstart](https://docs.wandb.ai/models/quickstart)
- [SLURM Documentation](https://slurm.schedmd.com/documentation.html)
