# TensorRT Examples on Jetson

I merged the following two repos to this one for my Jetson Nano.

1. [AastaNV/TRT_object_detection](https://github.com/AastaNV/TRT_object_detection.git)
2. [NVIDIA/object-detection-tensorrt-example](https://github.com/NVIDIA/object-detection-tensorrt-example)

## Background

`Jetpack 4.2` with `1.12.2` self-build tensorflow installed on my Jetson Nano.

## Install

Run all steps on your target platform, ex. Jetson Nano.

1. Install `tensorflow-gpu` with tensorRT enabled on your Jetson platform
2. Install pycuda (ref [this](https://devtalk.nvidia.com/default/topic/1013387/jetson-tx2/is-the-memory-management-method-of-tx1-and-tx2-different-/post/5352551/#5352551) for Jetson Nano)
3. Patch your `graphsurgeon converter`, please refer to the next section.
4. Put your `frozen_inference_graph.pb` to the repo root.
5. Run `convery.py` with one picture, ex. `python3 convert.py 1.jpg`
     - this step will generate `uff` and `bin` files on the repo root
6. Run `camera.py`, ex. `python3 camera`, enjoy!

## Update graphsurgeon converter

Edit /usr/lib/python3.6/dist-packages/graphsurgeon/node_manipulation.py

```C
diff --git a/node_manipulation.py b/node_manipulation.py
index d2d012a..1ef30a0 100644
--- a/node_manipulation.py
+++ b/node_manipulation.py
@@ -30,6 +30,7 @@ def create_node(name, op=None, _do_suffix=False, **kwargs):
     node = NodeDef()
     node.name = name
     node.op = op if op else name
+    node.attr["dtype"].type = 1
     for key, val in kwargs.items():
         if key == "dtype":
             node.attr["dtype"].type = val.as_datatype_enum
```
