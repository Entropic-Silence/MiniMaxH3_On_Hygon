import infinicore.nn.functional as F
import torch

import infinicore


def assert_output(actual, source, min_val, max_val):
    infinicore.sync_stream()
    torch.testing.assert_close(actual, torch.clamp(source, min_val, max_val))


def test_hardtanh_graph_replay():
    torch_device = torch.device("cuda", 0)
    device = infinicore.device("cuda", 0)
    min_val, max_val = -1.0, 1.0

    base = torch.linspace(-3.0, 3.0, 32, device=torch_device)
    torch_input = base.reshape(2, 2, 2, 2, 2).permute(0, 2, 4, 1, 3)
    torch_output = torch.empty(tuple(torch_input.shape), device=torch_device)
    assert not torch_input.is_contiguous()

    input_tensor = infinicore.strided_from_blob(
        torch_input.data_ptr(),
        list(torch_input.shape),
        list(torch_input.stride()),
        dtype=infinicore.float32,
        device=device,
    )
    output_tensor = infinicore.from_torch(torch_output)
    infinicore.set_device(device)

    F.hardtanh(
        input_tensor,
        min_val=min_val,
        max_val=max_val,
        out=output_tensor,
    )
    infinicore.sync_stream()

    infinicore.start_graph_recording(device)
    F.hardtanh(
        input_tensor,
        min_val=min_val,
        max_val=max_val,
        out=output_tensor,
    )
    graph = infinicore.stop_graph_recording()

    graph.run()
    assert_output(torch_output, torch_input, min_val, max_val)

    replacement = torch.linspace(3.0, -3.0, 32, device=torch_device).reshape(
        torch_input.shape
    )
    torch_input.copy_(replacement)
    torch.cuda.synchronize()
    graph.run()
    assert_output(torch_output, torch_input, min_val, max_val)


if __name__ == "__main__":
    test_hardtanh_graph_replay()
    print("HardTanh CUDA graph replay passed")
