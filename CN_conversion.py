from pathlib import Path
import openvino as ov
from functools import partial
import torch
from typing import Tuple

CONTROLNET_OV_PATH = Path('./cn_openvino_edge/controlnet.xml')
UNET_OV_PATH = Path('./cn_openvino_edge/unet_controlnet.xml')
TEXT_ENCODER_OV_PATH = Path('./cn_openvino_edge/text_encoder.xml')
VAE_DECODER_OV_PATH = Path('./cn_openvino_edge/vae_decoder.xml')
OPENPOSE_OV_PATH = Path("./cn_openvino_edge/openpose.xml")


inputs = {
    "sample": torch.randn((2, 4, 64, 64)),
    "timestep": torch.tensor(50),
    "encoder_hidden_states": torch.randn((2,77,768)),
    "controlnet_cond": torch.randn((2,3,512,512))
}

dtype_mapping = {
    torch.float32: ov.Type.f32,
    torch.float64: ov.Type.f64,
    torch.int32: ov.Type.i32,
    torch.int64: ov.Type.i64
}


def controlnet_conversion(controlnet):
    if CONTROLNET_OV_PATH.exists() and UNET_OV_PATH.exists():
        return None, None
    
    input_info = [(name, ov.PartialShape(inp.shape)) for name, inp in inputs.items()] 

    controlnet.eval()
    with torch.no_grad():
        down_block_res_samples, mid_block_res_sample = controlnet(**inputs, return_dict=False)
        
    if not CONTROLNET_OV_PATH.exists():
        with torch.no_grad():
            controlnet.forward = partial(controlnet.forward, return_dict=False)
            ov_model = ov.convert_model(controlnet, example_input=inputs, input=input_info)
            ov.save_model(ov_model, CONTROLNET_OV_PATH)
            del ov_model
        print('ControlNet successfully converted to IR')
    else:
        print(f"ControlNet will be loaded from {CONTROLNET_OV_PATH}")
    
    return down_block_res_samples, mid_block_res_sample

class UnetWrapper(torch.nn.Module):
    def __init__(
        self, 
        unet, 
        sample_dtype=torch.float32, 
        timestep_dtype=torch.int64, 
        encoder_hidden_states=torch.float32, 
        down_block_additional_residuals=torch.float32, 
        mid_block_additional_residual=torch.float32
    ):
        super().__init__()
        self.unet = unet
        self.sample_dtype = sample_dtype
        self.timestep_dtype = timestep_dtype
        self.encoder_hidden_states_dtype = encoder_hidden_states
        self.down_block_additional_residuals_dtype = down_block_additional_residuals
        self.mid_block_additional_residual_dtype = mid_block_additional_residual

    def forward(
        self, 
        sample:torch.Tensor, 
        timestep:torch.Tensor, 
        encoder_hidden_states:torch.Tensor, 
        down_block_additional_residuals:Tuple[torch.Tensor],  
        mid_block_additional_residual:torch.Tensor
    ):
        sample.to(self.sample_dtype)
        timestep.to(self.timestep_dtype)
        encoder_hidden_states.to(self.encoder_hidden_states_dtype)
        down_block_additional_residuals = [res.to(self.down_block_additional_residuals_dtype) for res in down_block_additional_residuals]
        mid_block_additional_residual.to(self.mid_block_additional_residual_dtype)
        return self.unet(
            sample, 
            timestep, 
            encoder_hidden_states, 
            down_block_additional_residuals=down_block_additional_residuals, 
            mid_block_additional_residual=mid_block_additional_residual
        )

def flattenize_inputs(inputs):
    flatten_inputs = []
    for input_data in inputs:
        if input_data is None:
            continue
        if isinstance(input_data, (list, tuple)):
            flatten_inputs.extend(flattenize_inputs(input_data))
        else:
            flatten_inputs.append(input_data)
    return flatten_inputs

def unet_conversion(pipe, down_block_res_samples, mid_block_res_sample):

    if UNET_OV_PATH.exists():
        return

    inputs.pop("controlnet_cond", None)
    inputs["down_block_additional_residuals"] = down_block_res_samples
    inputs["mid_block_additional_residual"] = mid_block_res_sample

    unet = UnetWrapper(pipe.unet)
    unet.eval()

    with torch.no_grad():
        ov_model = ov.convert_model(unet, example_input=inputs)
        
    flatten_inputs = flattenize_inputs(inputs.values())
    for input_data, input_tensor in zip(flatten_inputs, ov_model.inputs):
        input_tensor.get_node().set_partial_shape(ov.PartialShape(input_data.shape))
        input_tensor.get_node().set_element_type(dtype_mapping[input_data.dtype])
    ov_model.validate_nodes_and_infer_types()
    ov.save_model(ov_model, UNET_OV_PATH)
    del ov_model
    print('Unet successfully converted to IR')

def text_encoder_conversion(text_encoder:torch.nn.Module):

    if TEXT_ENCODER_OV_PATH.exists():
        return

    input_ids = torch.ones((1, 77), dtype=torch.long)
    # switch model to inference mode
    text_encoder.eval()

    # disable gradients calculation for reducing memory consumption
    with torch.no_grad():
        ov_model = ov.convert_model(
            text_encoder,  # model instance
            example_input=input_ids,  # inputs for model tracing
            input=([1, 77],)
        )
        ov.save_model(ov_model, TEXT_ENCODER_OV_PATH)
        del ov_model
    print('Text Encoder successfully converted to IR')


def vae_decoder_conversion(vae: torch.nn.Module):

    if VAE_DECODER_OV_PATH.exists():
        return
    
    class VAEDecoderWrapper(torch.nn.Module):
        def __init__(self, vae):
            super().__init__()
            self.vae = vae

        def forward(self, latents):
            return self.vae.decode(latents)

    vae_decoder = VAEDecoderWrapper(vae)
    latents = torch.zeros((1, 4, 64, 64))

    vae_decoder.eval()
    with torch.no_grad():
        ov_model = ov.convert_model(vae_decoder, example_input=latents, input=[(1,4,64,64),])
        ov.save_model(ov_model, VAE_DECODER_OV_PATH)
    del ov_model
    print('VAE decoder successfully converted to IR')
