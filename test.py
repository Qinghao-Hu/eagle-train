import torch

# Load frequency map data
print("Loading frequency map data...")
mapping_data = torch.load("freq_map/qwen2.5/freq_32768.pt", map_location="cpu")
freq_d2t = mapping_data.get("d2t", None)
freq_t2d = mapping_data.get("t2d", None)

print(f"Frequency map d2t shape: {freq_d2t.shape}")
print(f"Frequency map t2d shape: {freq_t2d.shape}")

# Load EAGLE model
print("\nLoading EAGLE model...")
model_path = "/nobackup/model/eagle/EAGLE3-LLaMA3.1-Instruct-8B/pytorch_model.bin"
model_data = torch.load(model_path, map_location="cpu")

# Extract d2t and t2d from model if they exist
model_d2t = None
model_t2d = None

# Check if d2t and t2d exist in the model
for key in model_data.keys():
    if "d2t" in key.lower():
        model_d2t = model_data[key]
        print(f"Found d2t in model: key='{key}', shape={model_d2t.shape}")
    if "t2d" in key.lower():
        model_t2d = model_data[key]
        print(f"Found t2d in model: key='{key}', shape={model_t2d.shape}")

# If not found directly, list all keys to see what's available
if model_d2t is None or model_t2d is None:
    print("\nModel keys:")
    for key in sorted(model_data.keys()):
        print(f"  {key}: {model_data[key].shape if hasattr(model_data[key], 'shape') else type(model_data[key])}")

# Compare if both exist
if freq_d2t is not None and model_d2t is not None:
    print(f"\n=== D2T Comparison ===")
    print(f"Frequency map d2t shape: {freq_d2t.shape}")
    print(f"Model d2t shape: {model_d2t.shape}")

    if freq_d2t.shape == model_d2t.shape:
        are_equal = torch.equal(freq_d2t, model_d2t)
        print(f"Tensors are equal: {are_equal}")
        if not are_equal:
            diff = torch.abs(freq_d2t - model_d2t)
            print(f"Max difference: {diff.max().item()}")
            print(f"Mean difference: {diff.mean().item()}")
    else:
        print("Shapes don't match, cannot compare element-wise")

if freq_t2d is not None and model_t2d is not None:
    print(f"\n=== T2D Comparison ===")
    print(f"Frequency map t2d shape: {freq_t2d.shape}")
    print(f"Model t2d shape: {model_t2d.shape}")

    if freq_t2d.shape == model_t2d.shape:
        are_equal = torch.equal(freq_t2d, model_t2d)
        print(f"Tensors are equal: {are_equal}")
        if not are_equal:
            diff = torch.abs(freq_t2d - model_t2d)
            print(f"Max difference: {diff.max().item()}")
            print(f"Mean difference: {diff.mean().item()}")
    else:
        print("Shapes don't match, cannot compare element-wise")

print(f"\n=== Original Data ===")
print(f"Frequency map d2t:\n{freq_d2t}")
print(f"Frequency map t2d:\n{freq_t2d}")

if model_d2t is not None:
    print(f"\nModel d2t:\n{model_d2t}")
if model_t2d is not None:
    print(f"Model t2d:\n{model_t2d}")
