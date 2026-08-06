import hashlib
import os

import numpy as np
import tensorrt as trt
from cuda.bindings import runtime as cudart

from utils.logger import Logger


def _check(err) -> None:
    if isinstance(err, tuple):
        err = err[0]
    if err != cudart.cudaError_t.cudaSuccess:
        raise RuntimeError(f"CUDA error: {err}")


def _hash_onnx(onnx_path: str) -> str:
    digest = hashlib.sha256()
    with open(onnx_path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    digest.update(trt.__version__.encode())
    return digest.hexdigest()


def build_engine(
    onnx_path: str, engine_path: str, logger: Logger | None = None
) -> bytes:
    hash_path = f"{engine_path}.sha256"
    onnx_hash = _hash_onnx(onnx_path)

    if os.path.exists(engine_path) and os.path.exists(hash_path):
        with open(hash_path) as f:
            cached_hash = f.read().strip()
        if cached_hash == onnx_hash:
            if logger is not None:
                logger.info(f"Reusing cached TensorRT engine {engine_path}")
            with open(engine_path, "rb") as f:
                return f.read()

    trt_logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(trt_logger)
    network = builder.create_network()
    parser = trt.OnnxParser(network, trt_logger)

    if not parser.parse_from_file(onnx_path):
        for i in range(parser.num_errors):
            print(parser.get_error(i))
        raise RuntimeError("Failed to parse ONNX model")

    builder_config = builder.create_builder_config()
    builder_config.set_memory_pool_limit(
        trt.MemoryPoolType.WORKSPACE, 1 << 30
    )  # 1GB memory limit

    serialized_engine = builder.build_serialized_network(network, builder_config)

    if serialized_engine is None:
        raise RuntimeError("Engine not built")

    with open(engine_path, "wb") as f:
        f.write(serialized_engine)
    with open(hash_path, "w") as f:
        f.write(onnx_hash)

    return bytes(serialized_engine)


class TensorRtRuntime:
    def __init__(self, engine_bytes: bytes):
        trt_logger = trt.Logger(trt.Logger.WARNING)
        self._runtime = trt.Runtime(trt_logger)
        self._engine = self._runtime.deserialize_cuda_engine(engine_bytes)
        if self._engine is None:
            raise RuntimeError(
                f"failed to deserialize TensorRT engine: incompatible with "
                f"installed TensorRT {trt.__version__}. Delete the cached "
                f"engine and .sha256 sidecar and rebuild."
            )
        self._context = self._engine.create_execution_context()

        err, self._stream = cudart.cudaStreamCreate()
        _check(err)

        output_names = [
            self._engine.get_tensor_name(i)
            for i in range(self._engine.num_io_tensors)
            if self._engine.get_tensor_mode(self._engine.get_tensor_name(i))
            == trt.TensorIOMode.OUTPUT
        ]
        if len(output_names) != 1:
            raise RuntimeError(
                f"expected exactly one output tensor, got {output_names}"
            )
        self._output_name = output_names[0]

        self._device_ptrs: dict[str, int] = {}
        self._host_output: np.ndarray | None = None

    def infer(self, inputs: dict[str, np.ndarray]) -> np.ndarray:
        for name, array in inputs.items():
            host_array = np.ascontiguousarray(array, dtype=np.float32)
            self._context.set_input_shape(name, host_array.shape)

            if name not in self._device_ptrs:
                err, d_ptr = cudart.cudaMalloc(host_array.nbytes)
                _check(err)
                self._device_ptrs[name] = d_ptr
                self._context.set_tensor_address(name, int(d_ptr))

            _check(
                cudart.cudaMemcpyAsync(
                    self._device_ptrs[name],
                    host_array.ctypes.data,
                    host_array.nbytes,
                    cudart.cudaMemcpyKind.cudaMemcpyHostToDevice,
                    self._stream,
                )
            )

        if self._host_output is None:
            output_shape = tuple(self._context.get_tensor_shape(self._output_name))
            self._host_output = np.empty(output_shape, dtype=np.float32)
            err, d_output = cudart.cudaMalloc(self._host_output.nbytes)
            _check(err)
            self._device_ptrs[self._output_name] = d_output
            self._context.set_tensor_address(self._output_name, int(d_output))

        if not self._context.execute_async_v3(self._stream):
            raise RuntimeError(
                f"execute_async_v3 failed for inputs {list(inputs)} -> "
                f"output {self._output_name!r}"
            )
        _check(
            cudart.cudaMemcpyAsync(
                self._host_output.ctypes.data,
                self._device_ptrs[self._output_name],
                self._host_output.nbytes,
                cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost,
                self._stream,
            )
        )
        _check(cudart.cudaStreamSynchronize(self._stream))

        return self._host_output.copy()

    def close(self) -> None:
        for d_ptr in self._device_ptrs.values():
            _check(cudart.cudaFree(d_ptr))
        self._device_ptrs.clear()
        _check(cudart.cudaStreamDestroy(self._stream))

    def __enter__(self) -> "TensorRtRuntime":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()
