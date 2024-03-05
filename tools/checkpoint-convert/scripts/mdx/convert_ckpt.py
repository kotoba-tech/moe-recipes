import argparse

import torch
from transformers import MixtralForCausalLM, MixtralConfig
from transformers import PretrainedConfig

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=str, required=True, help="HuggingFace transformers config path"
    )
    parser.add_argument("--ckpt", type=str, required=True, help="Path to checkpoint (`model.pth`)")
    parser.add_argument("--out", type=str, required=True, help="Path to output directory")
    parser.add_argument("--sequence-length", type=int, required=True)
    args = parser.parse_args()

    print(f"Loading HF config: {args.config}", flush=True)
    config = PretrainedConfig.from_json_file(args.config)
    print(f"HF config: {config}", flush=True)
    model = MixtralForCausalLM(config=config)

    print(f"Loading CKPT: {args.ckpt}", flush=True)
    state_dict = torch.load(args.ckpt, map_location="cpu")

    print("Loading state dict into HF model", flush=True)
    model.load_state_dict(state_dict)

    print("Saving HF model", flush=True)
    model.save_pretrained(args.out, safe_serialization=True)

if __name__ == "__main__":
    main()
