# Docker Image Comparison: Fat vs Slim

## Image Size Comparison

| Image | Base Image | Actual Size | Layers | Torch Variant |
| --- | --- | ---: | ---: | --- |
| `lesson3-fat` | `python:3.9` | 12.4 GB | 11 | `2.8.0+cu128` |
| `lesson3-slim` | `distroless/python3` | 1.02 GB | 46 | `2.11.0+cpu` |

Measured with:

```bash
docker images | grep lesson3
docker inspect lesson3-fat \
  | python3 -c "import sys,json; d=json.load(sys.stdin); print(len(d[0]['RootFS']['Layers']))"
docker inspect lesson3-slim \
  | python3 -c "import sys,json; d=json.load(sys.stdin); print(len(d[0]['RootFS']['Layers']))"
```

The slim image has more layers (46) than the fat image (11), but it is still much
smaller. Distroless base images pre-bake many fine-grained layers internally,
while `python:3.9` merges more content into fewer layers. Layer count does not
directly correlate with image size; the size depends on the uncompressed layer
content.

Representative `compare.sh` output after pinning the fat image dependencies and
the slim runtime digest:

```text
Export TorchScript model
  model.pt already exists; skipping export. Set FORCE_EXPORT=1 to regenerate it.

Compare image sizes
+ bash -lc docker images | grep lesson3
  lesson3-fat   latest  ...  12.4GB
  lesson3-slim  latest  ...  1.02GB

Count layers
  fat layers: 11
  slim layers: 46

Run inference with Docker
  + docker run --rm -v /home/nickolasz/Projects/GoIT/devops/lesson-3/test.jpg:/app/test.jpg lesson3-fat /app/test.jpg
  [info] Loaded 1000 ImageNet class labels from local file.

  Top-3 predictions:
    #1 beacon (437): 99.25%
    #2 breakwater (460): 0.16%
    #3 promontory (976): 0.03%

  + docker run --rm -v /home/nickolasz/Projects/GoIT/devops/lesson-3/test.jpg:/app/test.jpg lesson3-slim /app/test.jpg
  [info] Loaded 1000 ImageNet class labels from local file.

  Top-3 predictions:
    #1 beacon (437): 99.25%
    #2 breakwater (460): 0.16%
    #3 promontory (976): 0.03%

Inspect image metadata

  lesson3-fat:
    "RootFS": {
        "Type": "layers",
        "Layers": [
            "sha256:f2522c6ed78b338a9e272dd5038005d008f74729e036073e837f701f221b99ba",
            "sha256:fbde375eafc7442622b62f4756226bc327ab9e9a6acb70c5a3a685a2a036a669",
            "sha256:c041ff8d6c7338794cad076d734c7e4259395b9925ac352c50c816346a794f60",
            ...

  lesson3-slim:
    "RootFS": {
        "Type": "layers",
        "Layers": [
            "sha256:a33ba213ad26a605d904a741f67154f3904c74a43182c872c35f36b10fa005e0",
            "sha256:8fa10c0194df9b7c054c90dbe482585f768a54428fc90a5b78a0066a123b1bba",
            "sha256:4840c7c54023c867f19564429c89ddae4e9589c83dce82492183a7e9f7dab1fa",
            ...

Local inference commands
  source .venv/bin/activate
  python inference.py path/to/image.jpg
  python inference.py --image path/to/image.jpg
  python inference.py path/to/folder/
  .venv/bin/python inference.py path/to/image.jpg
```

## Problems with the Fat Image

1. Unnecessary system packages at runtime:
   - `build-essential`, `gcc`, `g++`, and `cmake` are only needed during build,
     not in production.
   - `wget`, `curl`, `vim`, `nano`, and `git` are development/debugging tools
     that increase image size and expand the attack surface.

2. Missing `--no-install-recommends`:
   - `apt` installs recommended dependencies, documentation, locales, and man
     pages, adding roughly 200-400 MB with no functional value for inference.

3. Full CUDA PyTorch distribution:
   - `pip install torch==2.8.0 torchvision==0.23.0` from PyPI pulls
     CUDA-enabled runtime packages. In this project, the resulting fat image is
     12.4 GB.
   - The slim image installs CPU-only wheels from the PyTorch CPU index and is
     1.02 GB, so the measured image-level saving is about 12x.

4. `pip` cache is not cleared between layers:
   - Each `RUN` instruction creates a separate layer. The `pip` cache remains in
     the builder layer and increases the final image size.

5. Large base image:
   - `python:3.9` is roughly 900 MB and includes a full Debian environment with
     compilers, development headers, and documentation.
   - `python:3.9-slim` is roughly 50 MB and excludes many unnecessary
     components.

## Slim Image Improvements

1. Multi-stage build:
   - The builder stage installs dependencies into `/root/.local`.
   - The runtime stage copies only `/root/.local` with `COPY --from=builder`.
   - No compilers or `apt` packages from the builder stage end up in the final
     image.
   - The runtime image is pinned by digest so the Python 3.11 user-site path
     stays stable.

2. Distroless runtime image:
   - Runtime uses `gcr.io/distroless/python3-debian12`.
   - `apt`, shell, `curl`, `wget`, and `gcc` are absent.
   - `/usr/bin/apt-get` is physically absent, unlike in `python:3.9-slim`.
   - The image is based on Debian 12 (bookworm), which has current security
     support.

3. `--no-install-recommends`:
   - Recommended `apt` packages that are not needed at runtime are excluded.

4. CPU-only PyTorch:
   - Dependencies are installed with
     `--index-url https://download.pytorch.org/whl/cpu`.
   - `--index-url` replaces PyPI as the primary index and prevents accidental
     CUDA wheel selection.
   - Avoiding CUDA runtime packages is the main reason the slim image is
     1.02 GB instead of 12.4 GB.

5. `--no-cache-dir` for `pip`:
   - `pip` does not store cache inside the builder layer, reducing intermediate
     image size.

6. Minimal runtime files:
   - The final image contains only `model.pt`, `inference.py`, and
     `imagenet_classes.txt`.
   - Development files, documentation, and tests are not copied into the runtime
     image.

## Further Optimization Suggestions

1. Use CPU-only PyTorch, already applied in the slim image:

   ```bash
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
   ```

   This reduces PyTorch from roughly 800 MB to roughly 300 MB by excluding CUDA.

2. Use ONNX Runtime instead of PyTorch:

   Convert `model.pt` to `model.onnx` and run inference with `onnxruntime`.

   ```bash
   python3 -c "
   import torch, torchvision
   m = torch.jit.load('model.pt'); m.eval()
   dummy = torch.zeros(1, 3, 224, 224)
   torch.onnx.export(m, dummy, 'model.onnx', opset_version=13)
   "

   pip install onnxruntime
   ```

   This can reduce the image from roughly 2 GB to roughly 200-300 MB.

3. Use a pinned distroless base image:

   ```dockerfile
   FROM gcr.io/distroless/python3-debian12@sha256:2fdb05402a2cf21cf78fdb3ba4c5db167241e9e498140f5bf689d7efb773731f
   ```

   This removes the shell, package manager, and unnecessary binaries from the
   runtime image.

4. Use the TorchScript Lite interpreter:

   ```python
   torch._C._jit_get_operation()
   torch.jit._export_operator_list()
   ```

   This helps minimize the included operators in `model.pt`.

5. Improve Docker layer caching:

   ```dockerfile
   COPY requirements.txt .
   RUN pip install -r requirements.txt
   COPY . .
   ```

   Changing application code would not invalidate the dependency installation
   layer.

6. Squash layers, if Docker experimental mode is enabled:

   ```bash
   docker build --squash -f Dockerfile.slim -t lesson3-slim .
   ```

   Squashing merges all layers into one and can eliminate files left behind
   between layers.

7. Quantize the model to INT8:

   ```python
   torch.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
   ```

   This can reduce `model.pt` from roughly 14 MB to roughly 4 MB and speed up
   CPU inference.

## Commands to Build and Test

See the build and test instructions in [README.md](README.md#quick-start).
