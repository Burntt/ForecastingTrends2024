import torch
import os
import shutil
import torch

def prepare_traced_model(model, args):
    """
    This function converts a model to TorchScript via tracing, preparing it for export.

    Parameters:
    model: The original PyTorch model.
    args: Arguments, including batch_size, seq_len, enc_in, pred_len, dec_in, and use_gpu.

    Returns:
    traced_model: The model converted to TorchScript.
    """

    # Determine the device to use
    device = torch.device("cuda" if args.use_gpu else "cpu")
    model = model.to(device)

    # Prepare inputs
    example_x_enc = torch.rand(args.batch_size, args.seq_len, args.enc_in).to(device).float()
    example_x_mark_enc = torch.rand(args.batch_size, args.seq_len, 1).to(device).float()
    example_x_dec = torch.rand(args.batch_size, args.pred_len, args.dec_in).to(device).float()
    example_x_mark_dec = torch.rand(args.batch_size, args.pred_len, 1).to(device).float()

    inputs_trace = (example_x_enc, example_x_mark_enc, example_x_dec, example_x_mark_dec)
    traced_model = torch.jit.trace(model, inputs_trace)
    inputs_examples = [example_x_enc, example_x_mark_enc, example_x_dec, example_x_mark_dec]
    
    # Get the shapes of the inputs
    input_shapes = [input.shape for input in inputs_trace]

    return traced_model, inputs_examples, input_shapes


def save_to_triton_flavor(traced_model, inputs_trace, preds, tmp_folder_triton, setting, version):
    """
    This function saves the TorchScript model to a Triton flavor.

    Parameters:
    traced_model: The TorchScript model.
    inputs_trace: The inputs used for model tracing.
    preds: The predictions from the model.
    tmp_folder_triton: The name of the temporary directory for Triton.
    setting: The settings for the model.
    version: The version of the model.

    Returns:
    None
    """

    # Create the directories
    os.makedirs(f"{tmp_folder_triton}/{version}", exist_ok=True)

    # Save the traced model
    torch.jit.save(traced_model, f"{tmp_folder_triton}/{version}/model.pt")

    # Get the shapes of the inputs
    input_shapes = [input.shape for input in inputs_trace]

    # Create the config.pbtxt file
    config = """
    name: "{model_name}"
    platform: "pytorch_libtorch"
    """.format(model_name=tmp_folder_triton)

    for i, shape in enumerate(input_shapes):
        config += """
        input [{{
            name: "input{0}"
            data_type: TYPE_FP32
            format: FORMAT_NONE
            dims: {1}
        }}]
        """.format(i, list(shape))

    # Assuming the output has the same shape as the preds
    output_shape = list(preds.shape)

    config += """
    output [{{
        name: "output0"
        data_type: TYPE_FP32
        dims: {0}
    }}]
    """.format(output_shape)

    # Write the config to a file
    with open(f'{tmp_folder_triton}/config.pbtxt', 'w') as f:
        f.write(config)
    # Save the settings string
    with open(f'{tmp_folder_triton}/setting.txt', 'w') as f:
        f.write(setting)

    # Save the arguments as an artifact
    source_args_path = os.path.join('./checkpoints/' + setting, 'args.pth')
    destination_args_path = f'{tmp_folder_triton}/args.pth'
    shutil.copyfile(source_args_path, destination_args_path)
