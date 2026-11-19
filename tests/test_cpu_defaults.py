"""Thread count and precision defaults (src/blue_onnx/cpu.py). No model weights needed."""

from __future__ import annotations

import os
import tempfile
import unittest
from unittest import mock

import numpy as np

from src.blue_onnx import cpu


def _cgroup(files: dict[str, str]) -> str:
    root = tempfile.mkdtemp()
    for rel, text in files.items():
        path = os.path.join(root, rel)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            f.write(text)
    return root


class TestCgroupLimit(unittest.TestCase):
    def test_v2_quota_rounds_up(self):
        self.assertEqual(cpu.cgroup_cpu_limit(_cgroup({"cpu.max": "200000 100000\n"})), 2)
        self.assertEqual(cpu.cgroup_cpu_limit(_cgroup({"cpu.max": "150000 100000\n"})), 2)
        self.assertEqual(cpu.cgroup_cpu_limit(_cgroup({"cpu.max": "50000 100000\n"})), 1)

    def test_v2_unlimited(self):
        self.assertIsNone(cpu.cgroup_cpu_limit(_cgroup({"cpu.max": "max 100000\n"})))

    def test_v1(self):
        root = _cgroup({"cpu/cpu.cfs_quota_us": "400000\n", "cpu/cpu.cfs_period_us": "100000\n"})
        self.assertEqual(cpu.cgroup_cpu_limit(root), 4)
        root = _cgroup({"cpu/cpu.cfs_quota_us": "-1\n", "cpu/cpu.cfs_period_us": "100000\n"})
        self.assertIsNone(cpu.cgroup_cpu_limit(root))

    def test_no_cgroup_fs(self):
        self.assertIsNone(cpu.cgroup_cpu_limit(tempfile.mkdtemp()))


class TestThreads(unittest.TestCase):
    def test_env_wins(self):
        with mock.patch.dict(os.environ, {"ORT_NUM_THREADS": "3"}):
            self.assertEqual(cpu.default_num_threads(), 3)

    def test_quota_caps_threads(self):
        # the measured failure: a 2-CPU container on a 16-thread host used to get 8 threads
        with mock.patch.dict(os.environ, {"ORT_NUM_THREADS": ""}), \
                mock.patch.object(cpu, "cgroup_cpu_limit", return_value=2), \
                mock.patch.object(cpu.os, "cpu_count", return_value=16):
            self.assertEqual(cpu.default_num_threads(), 2)

    def test_capped_at_eight(self):
        with mock.patch.dict(os.environ, {"ORT_NUM_THREADS": ""}), \
                mock.patch.object(cpu, "available_cpus", return_value=64):
            self.assertEqual(cpu.default_num_threads(), 8)


class TestPrecision(unittest.TestCase):
    def setUp(self):
        self._env = mock.patch.dict(os.environ, {"BLUE_ONNX_PRECISION": ""})
        self._env.start()

    def tearDown(self):
        self._env.stop()

    def test_auto(self):
        r = cpu.resolve_precision
        self.assertEqual(r(None, None, machine="x86_64", system="Linux"), "int8")
        self.assertEqual(r("auto", ["CPUExecutionProvider"], machine="AMD64", system="Windows"), "int8")
        self.assertEqual(r(None, None, machine="aarch64", system="Linux"), "int8")
        self.assertEqual(r(None, None, machine="arm64", system="Darwin"), "fp32")
        self.assertEqual(r(None, ["CUDAExecutionProvider", "CPUExecutionProvider"], machine="x86_64", system="Linux"), "fp32")
        self.assertEqual(r(None, [("OpenVINOExecutionProvider", {})], machine="x86_64", system="Linux"), "fp32")

    def test_explicit_and_env(self):
        self.assertEqual(cpu.resolve_precision("fp32", None, machine="x86_64", system="Linux"), "fp32")
        self.assertEqual(cpu.resolve_precision("int8", None, machine="arm64", system="Darwin"), "int8")
        with mock.patch.dict(os.environ, {"BLUE_ONNX_PRECISION": "fp32"}):
            self.assertEqual(cpu.resolve_precision(None, None, machine="x86_64", system="Linux"), "fp32")
        with self.assertRaises(ValueError):
            cpu.resolve_precision("fp16", None)


class TestProviderThreads(unittest.TestCase):
    def test_openvino_gets_thread_count(self):
        self.assertEqual(cpu.with_thread_options(["OpenVINOExecutionProvider", "CPUExecutionProvider"], 2),
                         [("OpenVINOExecutionProvider", {"num_of_threads": "2"}), "CPUExecutionProvider"])

    def test_caller_options_win(self):
        out = cpu.with_thread_options([("OpenVINOExecutionProvider", {"num_of_threads": "4", "device_type": "CPU"})], 2)
        self.assertEqual(out, [("OpenVINOExecutionProvider", {"num_of_threads": "4", "device_type": "CPU"})])

    def test_others_untouched(self):
        self.assertEqual(cpu.with_thread_options(["CUDAExecutionProvider"], 2), ["CUDAExecutionProvider"])


class TestQuantize(unittest.TestCase):
    def test_int8_graph_matches_fp32(self):
        import onnx
        import onnxruntime as ort
        from onnx import TensorProto, helper, numpy_helper

        rng = np.random.default_rng(0)
        w = rng.standard_normal((64, 32)).astype(np.float32) / 8
        graph = helper.make_graph(
            [helper.make_node("MatMul", ["x", "w"], ["y"])], "g",
            [helper.make_tensor_value_info("x", TensorProto.FLOAT, ["B", 64])],
            [helper.make_tensor_value_info("y", TensorProto.FLOAT, ["B", 32])],
            [numpy_helper.from_array(w, "w")],
        )
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
        d = tempfile.mkdtemp()
        src = os.path.join(d, "vector_estimator.onnx")
        onnx.save(model, src)
        dst = cpu.int8_vector_estimator_path(d, src)
        self.assertEqual(dst, os.path.join(d, cpu.INT8_VECTOR_ESTIMATOR))
        self.assertIn("MatMulInteger", {n.op_type for n in onnx.load(dst).graph.node})
        x = rng.standard_normal((4, 64)).astype(np.float32)
        y = ort.InferenceSession(src, providers=["CPUExecutionProvider"]).run(None, {"x": x})[0]
        q = ort.InferenceSession(dst, providers=["CPUExecutionProvider"]).run(None, {"x": x})[0]
        self.assertLess(np.linalg.norm(q - y) / np.linalg.norm(y), 0.05)
        # a second call finds the file instead of quantizing again
        self.assertEqual(cpu.int8_vector_estimator_path(d, src, create=False), dst)


if __name__ == "__main__":
    unittest.main()
