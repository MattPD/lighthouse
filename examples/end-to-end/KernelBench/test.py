#!/usr/bin/env python

# RUN: python %s 2>&1 | FileCheck %s
# CHECK-NOT: Execution failed

import subprocess
from pathlib import Path

if __name__ == "__main__":
    # This is a simple test to run the KernelBench example end-to-end.
    # It imports the PyTorch model, converts it to MLIR, runs the optimization pipeline, and executes the module.
    # The test passes if the output of the module matches the expected output from the PyTorch model.

    kb_program = Path(__file__).parent / "kernel_bench"
    project_root = Path(__file__).parent.parent.parent.parent
    kb_path = project_root / "third_party" / "KernelBench" / "KernelBench"
    kb_kernels = [
        kb_path / kb_script
        for kb_script in [
            "level1/1_Square_matrix_multiplication_.py",
            "level2/2_Standard_matrix_multiplication_.py",
        ]
    ]

    for kb_kernel in kb_kernels:
        result = subprocess.run(
            [
                kb_program,
                str(kb_kernel),
                "--input-shape",
                "4096x4096xf32xid,4096x4096xf32xrnd,4096x4096xf32x0",
                "--print-tensor=3",
            ],
            capture_output=True,
            text=True,
        )

        print("STDOUT:")
        print(result.stdout)
        print("STDERR:")
        print(result.stderr)

        assert result.returncode == 0, "Execution failed"
