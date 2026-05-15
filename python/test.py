import onnxruntime as ort
import numpy as np
import time

sess = ort.InferenceSession(
    "../go/models/temporal_upscaler_raft.onnx",
    providers=["CUDAExecutionProvider"],
)

binding = sess.io_binding()

current_lr = ort.OrtValue.ortvalue_from_numpy(
    np.random.randn(1, 3, 720, 1280).astype(np.float32), "cuda", 0
)
prev_sr = ort.OrtValue.ortvalue_from_numpy(
    np.random.randn(1, 3, 1440, 2560).astype(np.float32), "cuda", 0
)
flow_lr = ort.OrtValue.ortvalue_from_numpy(
    np.random.randn(1, 2, 720, 1280).astype(np.float32), "cuda", 0
)
output = ort.OrtValue.ortvalue_from_shape_and_type(
    [1, 3, 1440, 2560], np.float32, "cuda", 0
)

binding.bind_ortvalue_input("current_lr", current_lr)
binding.bind_ortvalue_input("prev_sr", prev_sr)
binding.bind_ortvalue_input("flow_lr", flow_lr)
binding.bind_ortvalue_output("sr_output", output)

for _ in range(10):
    sess.run_with_iobinding(binding)

times = []
for _ in range(100):
    t = time.perf_counter()
    sess.run_with_iobinding(binding)
    times.append((time.perf_counter() - t) * 1000)

print(f"IO Binding Average: {np.mean(times):.1f} ms")
print(f"IO Binding FPS: {1000/np.mean(times):.1f}")

# import onnxruntime as ort
# import numpy as np
# import time

# sess_options = ort.SessionOptions()
# sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL

# sess = ort.InferenceSession(
#     "../go/models/temporal_upscaler.onnx",
#     sess_options,
#     providers=["DmlExecutionProvider", "CPUExecutionProvider"],
# )
# print("Provider:", sess.get_providers())

# inputs = {
#     "current_lr": np.random.randn(1, 3, 270, 480).astype(np.float32),
#     "prev_sr": np.random.randn(1, 3, 1080, 1920).astype(np.float32),
#     "flow_lr": np.random.randn(1, 2, 270, 480).astype(np.float32),
# }

# for _ in range(5):
#     sess.run(None, inputs)

# times = []
# for _ in range(50):
#     t = time.perf_counter()
#     sess.run(None, inputs)
#     times.append((time.perf_counter() - t) * 1000)

# print(f"Average: {np.mean(times):.1f} ms")
# print(f"FPS: {1000/np.mean(times):.1f}")