import os
import torch

paths = [
    r"D:\CMT_PROJECT\processed\supplementary_gait\best_gait_encoder.pth",
    r"D:\CMT_PROJECT\processed\pretrained_inception_encoder.pth",
]

for path in paths:
    print("\n" + "=" * 65)
    print("FILE:", path)

    if not os.path.isfile(path):
        print("STATUS: File not found")
        continue

    print("SIZE:", os.path.getsize(path), "bytes")

    with open(path, "rb") as f:
        print("FIRST 16 BYTES:", f.read(16))

    try:
        obj = torch.load(
            path,
            map_location="cpu",
            weights_only=True
        )

        print("STATUS: Loaded successfully")
        print("OBJECT TYPE:", type(obj).__name__)

        if isinstance(obj, dict):
            print("CHECKPOINT KEYS:", list(obj.keys())[:40])

            for key, value in obj.items():
                if isinstance(value, dict):
                    tensors = [
                        (name, tuple(tensor.shape))
                        for name, tensor in value.items()
                        if isinstance(tensor, torch.Tensor)
                    ]
                    if tensors:
                        print("TENSOR GROUP:", key)
                        print("EXAMPLE TENSORS:", tensors[:8])

                elif isinstance(value, torch.Tensor):
                    print("TENSOR:", key, tuple(value.shape))

        elif isinstance(obj, torch.Tensor):
            print("TENSOR SHAPE:", tuple(obj.shape))

    except Exception as exc:
        print("STATUS: LOAD FAILED")
        print("ERROR:", type(exc).__name__, str(exc)[:800])
