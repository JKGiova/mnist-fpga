from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRAINING_DIR = PROJECT_ROOT / "training"
QUANTIZATION_DIR = PROJECT_ROOT / "quantization"

sys.path.insert(0, str(TRAINING_DIR))

from model import MnistMLP  # noqa: E402


DEFAULT_CHECKPOINT = (
    TRAINING_DIR / "artifacts" / "mnist-mlp-fp32.pt"
)
DEFAULT_DATA_DIR = TRAINING_DIR / "data"
DEFAULT_OUTPUT_DIR = QUANTIZATION_DIR / "artifacts"

INT8_MAX = 127
INT32_MIN = -(1 << 31)
INT32_MAX = (1 << 31) - 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Quantize the 784-16-10 MNIST MLP for FPGA inference."
    )

    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT,
    )

    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=512,
    )

    return parser.parse_args()


def load_checkpoint(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    try:
        checkpoint = torch.load(
            path,
            map_location="cpu",
            weights_only=True,
        )
    except TypeError:
        checkpoint = torch.load(
            path,
            map_location="cpu",
        )

    if "state_dict" not in checkpoint:
        raise ValueError("The checkpoint does not contain state_dict")

    return checkpoint


def create_loader(
    data_dir: Path,
    train: bool,
    batch_size: int,
) -> DataLoader:
    dataset = datasets.MNIST(
        root=data_dir,
        train=train,
        download=True,
        transform=transforms.ToTensor(),
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )


def quantize_symmetric_int8(
    tensor: torch.Tensor,
) -> tuple[torch.Tensor, float]:
    """
    Symmetric INT8 quantization:

        scale = max(abs(tensor)) / 127
        quantized = round(tensor / scale)

    The range -127..127 is used so zero remains exactly zero.
    """

    maximum = float(tensor.detach().abs().max().item())

    if maximum == 0.0:
        scale = 1.0
    else:
        scale = maximum / INT8_MAX

    quantized = torch.round(tensor / scale)
    quantized = quantized.clamp(-INT8_MAX, INT8_MAX)

    return quantized.to(torch.int8), scale


def quantize_bias_int32(
    bias: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    quantized = torch.round(
        bias.detach() / scale
    ).to(torch.int64)

    check_int32("bias", quantized)

    return quantized.to(torch.int32)


def quantize_inputs(images: torch.Tensor) -> torch.Tensor:
    """
    MNIST pixels arrive from ToTensor() in the range 0.0..1.0.

    They are converted to signed INT8 values in the positive range
    0..127.
    """

    flattened = images.reshape(images.size(0), 784)

    quantized = torch.round(flattened * INT8_MAX)
    quantized = quantized.clamp(0, INT8_MAX)

    return quantized.to(torch.int8)


def check_int32(
    name: str,
    tensor: torch.Tensor,
) -> None:
    minimum = int(tensor.min().item())
    maximum = int(tensor.max().item())

    if minimum < INT32_MIN or maximum > INT32_MAX:
        raise OverflowError(
            f"{name} does not fit in INT32: "
            f"minimum={minimum}, maximum={maximum}"
        )


def rounded_right_shift(
    tensor: torch.Tensor,
    shift: int,
) -> torch.Tensor:
    """
    Rounded right shift for non-negative values.

    Equivalent hardware operation:

        result = (value + 2^(shift - 1)) >> shift
    """

    if shift == 0:
        return tensor

    rounding_value = 1 << (shift - 1)

    return (tensor + rounding_value) >> shift


def rounded_right_shift_scalar(
    value: int,
    shift: int,
) -> int:
    if shift == 0:
        return value

    return (value + (1 << (shift - 1))) >> shift


@torch.no_grad()
def calibrate_hidden_shift(
    loader: DataLoader,
    weight: torch.Tensor,
    bias: torch.Tensor,
) -> tuple[int, int]:
    """
    Find the smallest power-of-two shift that makes every calibrated
    hidden activation fit in 0..127.
    """

    weight_int64 = weight.to(torch.int64)
    bias_int64 = bias.to(torch.int64)

    maximum_relu_accumulator = 0

    for images, _ in loader:
        inputs = quantize_inputs(images).to(torch.int64)

        accumulator = (
            inputs @ weight_int64.T
            + bias_int64
        )

        check_int32(
            "first-layer calibration accumulator",
            accumulator,
        )

        accumulator = accumulator.clamp_min(0)

        batch_maximum = int(accumulator.max().item())

        maximum_relu_accumulator = max(
            maximum_relu_accumulator,
            batch_maximum,
        )

    shift = 0

    while (
        rounded_right_shift_scalar(
            maximum_relu_accumulator,
            shift,
        )
        > INT8_MAX
    ):
        shift += 1

    return shift, maximum_relu_accumulator


def integer_forward(
    images: torch.Tensor,
    fc1_weight: torch.Tensor,
    fc1_bias: torch.Tensor,
    hidden_shift: int,
    fc2_weight: torch.Tensor,
    fc2_bias: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    inputs = quantize_inputs(images).to(torch.int64)

    # First fully connected layer.
    fc1_accumulator = (
        inputs @ fc1_weight.to(torch.int64).T
        + fc1_bias.to(torch.int64)
    )

    check_int32(
        "first-layer accumulator",
        fc1_accumulator,
    )

    # ReLU.
    hidden = fc1_accumulator.clamp_min(0)

    # Requantize to 0..127 using a power-of-two division.
    hidden = rounded_right_shift(
        hidden,
        hidden_shift,
    )

    hidden = hidden.clamp(0, INT8_MAX)
    hidden = hidden.to(torch.int8)

    # Second fully connected layer.
    fc2_accumulator = (
        hidden.to(torch.int64)
        @ fc2_weight.to(torch.int64).T
        + fc2_bias.to(torch.int64)
    )

    check_int32(
        "second-layer accumulator",
        fc2_accumulator,
    )

    return fc2_accumulator, fc1_accumulator


@torch.no_grad()
def evaluate(
    model: MnistMLP,
    loader: DataLoader,
    fc1_weight: torch.Tensor,
    fc1_bias: torch.Tensor,
    hidden_shift: int,
    fc2_weight: torch.Tensor,
    fc2_bias: torch.Tensor,
) -> dict:
    model.eval()

    total = 0
    fp32_correct = 0
    int8_correct = 0
    agreement = 0

    fc1_minimum = INT32_MAX
    fc1_maximum = INT32_MIN
    fc2_minimum = INT32_MAX
    fc2_maximum = INT32_MIN

    for images, labels in loader:
        fp32_logits = model(images)
        fp32_prediction = fp32_logits.argmax(dim=1)

        int8_logits, fc1_accumulator = integer_forward(
            images=images,
            fc1_weight=fc1_weight,
            fc1_bias=fc1_bias,
            hidden_shift=hidden_shift,
            fc2_weight=fc2_weight,
            fc2_bias=fc2_bias,
        )

        int8_prediction = int8_logits.argmax(dim=1)

        total += labels.size(0)

        fp32_correct += (
            fp32_prediction == labels
        ).sum().item()

        int8_correct += (
            int8_prediction == labels
        ).sum().item()

        agreement += (
            fp32_prediction == int8_prediction
        ).sum().item()

        fc1_minimum = min(
            fc1_minimum,
            int(fc1_accumulator.min().item()),
        )

        fc1_maximum = max(
            fc1_maximum,
            int(fc1_accumulator.max().item()),
        )

        fc2_minimum = min(
            fc2_minimum,
            int(int8_logits.min().item()),
        )

        fc2_maximum = max(
            fc2_maximum,
            int(int8_logits.max().item()),
        )

    return {
        "samples": total,
        "fp32_accuracy": 100.0 * fp32_correct / total,
        "int8_accuracy": 100.0 * int8_correct / total,
        "prediction_agreement": 100.0 * agreement / total,
        "fc1_accumulator_min": fc1_minimum,
        "fc1_accumulator_max": fc1_maximum,
        "fc2_accumulator_min": fc2_minimum,
        "fc2_accumulator_max": fc2_maximum,
    }


def write_lines(
    path: Path,
    lines: list[str],
) -> None:
    path.write_text(
        "\n".join(lines) + "\n",
        encoding="ascii",
    )


def signed_byte(value: int) -> int:
    """
    Convert a signed INT8 value to its two's-complement byte.
    """

    return value & 0xFF


def signed_int32_hex(value: int) -> str:
    """
    Convert a signed INT32 value to eight hexadecimal characters.
    """

    return f"{value & 0xFFFFFFFF:08x}"


def export_mem_files(
    output_dir: Path,
    fc1_weight: torch.Tensor,
    fc1_bias: torch.Tensor,
    fc2_weight: torch.Tensor,
    fc2_bias: torch.Tensor,
) -> None:
    # FC1:
    # 784 memory words.
    # Each word contains the 16 weights for one input pixel.
    # Neuron 0 is stored in bits [7:0].
    fc1_weight_lines = []

    for input_index in range(784):
        word = 0

        for neuron in range(16):
            value = int(
                fc1_weight[neuron, input_index].item()
            )

            word |= (
                signed_byte(value)
                << (8 * neuron)
            )

        fc1_weight_lines.append(f"{word:032x}")

    # FC2:
    # 16 memory words.
    # Each word contains the 10 weights for one hidden activation.
    # Class 0 is stored in bits [7:0].
    fc2_weight_lines = []

    for hidden_index in range(16):
        word = 0

        for output_class in range(10):
            value = int(
                fc2_weight[output_class, hidden_index].item()
            )

            word |= (
                signed_byte(value)
                << (8 * output_class)
            )

        fc2_weight_lines.append(f"{word:020x}")

    fc1_bias_lines = [
        signed_int32_hex(int(value.item()))
        for value in fc1_bias
    ]

    fc2_bias_lines = [
        signed_int32_hex(int(value.item()))
        for value in fc2_bias
    ]

    write_lines(
        output_dir / "fc1-weights.mem",
        fc1_weight_lines,
    )

    write_lines(
        output_dir / "fc1-bias.mem",
        fc1_bias_lines,
    )

    write_lines(
        output_dir / "fc2-weights.mem",
        fc2_weight_lines,
    )

    write_lines(
        output_dir / "fc2-bias.mem",
        fc2_bias_lines,
    )


def main() -> None:
    args = parse_args()

    checkpoint = load_checkpoint(args.checkpoint)

    model = MnistMLP()
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    # Quantize both weight matrices independently.
    fc1_weight, fc1_weight_scale = (
        quantize_symmetric_int8(model.fc1.weight)
    )

    fc2_weight, fc2_weight_scale = (
        quantize_symmetric_int8(model.fc2.weight)
    )

    # x_real = x_int8 * input_scale
    input_scale = 1.0 / INT8_MAX

    # FC1 bias uses the same scale as the FC1 multiplication result.
    fc1_accumulator_scale = (
        input_scale * fc1_weight_scale
    )

    fc1_bias = quantize_bias_int32(
        model.fc1.bias,
        fc1_accumulator_scale,
    )

    print("Calibrating hidden activation shift...")

    calibration_loader = create_loader(
        data_dir=args.data_dir,
        train=True,
        batch_size=args.batch_size,
    )

    hidden_shift, calibration_maximum = (
        calibrate_hidden_shift(
            loader=calibration_loader,
            weight=fc1_weight,
            bias=fc1_bias,
        )
    )

    # The right shift multiplies the scale by 2^shift.
    hidden_scale = (
        fc1_accumulator_scale
        * (1 << hidden_shift)
    )

    fc2_accumulator_scale = (
        hidden_scale * fc2_weight_scale
    )

    fc2_bias = quantize_bias_int32(
        model.fc2.bias,
        fc2_accumulator_scale,
    )

    print("Testing integer-only inference...")

    test_loader = create_loader(
        data_dir=args.data_dir,
        train=False,
        batch_size=args.batch_size,
    )

    metrics = evaluate(
        model=model,
        loader=test_loader,
        fc1_weight=fc1_weight,
        fc1_bias=fc1_bias,
        hidden_shift=hidden_shift,
        fc2_weight=fc2_weight,
        fc2_bias=fc2_bias,
    )

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    metadata = {
        "architecture": {
            "inputs": 784,
            "hidden": 16,
            "outputs": 10,
            "activation": "ReLU",
        },
        "input": {
            "dtype": "int8",
            "range": [0, 127],
            "scale": input_scale,
            "zero_point": 0,
        },
        "fc1": {
            "weight_dtype": "int8",
            "weight_scale": fc1_weight_scale,
            "bias_dtype": "int32",
            "bias_scale": fc1_accumulator_scale,
            "accumulator_dtype": "int32",
        },
        "hidden": {
            "dtype": "int8",
            "range": [0, 127],
            "right_shift": hidden_shift,
            "scale": hidden_scale,
            "calibration_maximum": calibration_maximum,
        },
        "fc2": {
            "weight_dtype": "int8",
            "weight_scale": fc2_weight_scale,
            "bias_dtype": "int32",
            "bias_scale": fc2_accumulator_scale,
            "accumulator_dtype": "int32",
        },
        "metrics": metrics,
    }

    quantized_checkpoint = {
        "fc1_weight": fc1_weight,
        "fc1_bias": fc1_bias,
        "hidden_shift": hidden_shift,
        "fc2_weight": fc2_weight,
        "fc2_bias": fc2_bias,
        "metadata": metadata,
    }

    torch.save(
        quantized_checkpoint,
        args.output_dir / "mnist-mlp-int8.pt",
    )

    export_mem_files(
        output_dir=args.output_dir,
        fc1_weight=fc1_weight,
        fc1_bias=fc1_bias,
        fc2_weight=fc2_weight,
        fc2_bias=fc2_bias,
    )

    with (
        args.output_dir / "quantization.json"
    ).open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2)
        file.write("\n")

    print()
    print(
        f"FP32 test accuracy:   "
        f"{metrics['fp32_accuracy']:.2f}%"
    )
    print(
        f"INT8 test accuracy:   "
        f"{metrics['int8_accuracy']:.2f}%"
    )
    print(
        f"Prediction agreement: "
        f"{metrics['prediction_agreement']:.2f}%"
    )
    print(f"Hidden right shift:   {hidden_shift}")
    print(
        f"FC1 accumulator:      "
        f"{metrics['fc1_accumulator_min']} to "
        f"{metrics['fc1_accumulator_max']}"
    )
    print(
        f"FC2 accumulator:      "
        f"{metrics['fc2_accumulator_min']} to "
        f"{metrics['fc2_accumulator_max']}"
    )
    print(f"Files saved to:       {args.output_dir}")


if __name__ == "__main__":
    main()