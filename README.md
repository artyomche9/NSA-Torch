# 🐋NSA-Torch🐋

NSA-Torch is a pure PyTorch implementation of [Native Sparse Attention: Hardware-Aligned and Natively Trainable Sparse Attention](https://arxiv.org/abs/2502.11089).


---

## Installation

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/not-heavychevy/NSA-Torch.git
   cd NSA-Torch
   ```
2. **Install Dependencies:**

   It is recommended to use a virtual environment. Then install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

You can import and use the NSA model in your PyTorch projects. Below is a sample code snippet:

```python
import torch
from nsa import NSA

# Initialize the NSA model with desired parameters
model = NSA(block_size=16, window_size=8, topk_blocks=4)

# Create random input tensors
B, T, D = 2, 128, 64  # Batch size, sequence length, feature dimension
q = torch.randn(B, T, D)
k = torch.randn(B, T, D)
v = torch.randn(B, T, D)

# Define gate values for each branch (can be learned or fixed)
gate_cmp = torch.ones(B, T)
gate_slc = torch.ones(B, T)
gate_swa = torch.ones(B, T)

# Run the NSA model
output = model(q, k, v, gate_cmp, gate_slc, gate_swa)
print("Output shape:", output.shape)

```

## Running Tests

To run the unit tests and verify the correctness of the NSA implementation:

1. Ensure you are in the repository root.
2. Run the following command:
	```bash  
	pytest tests/test_nsa.py > test_results.txt 2>&1
	```

This command will execute all tests in the `tests/test_nsa.py` file and save the output in `test_results.txt`.

## License

This project is licensed under the [MIT License](LICENSE).

## Contributing

Contributions, issues, and feature requests are welcome! Feel free to check the [issues page](https://github.com/artyomche9/NSA-Torch/issues).

## Acknowledgments

This repository is inspired by recent research in sparse attention mechanisms for efficient long-context modeling in language models.

---
Feel free to reach out if you have any questions or suggestions. Happy coding!