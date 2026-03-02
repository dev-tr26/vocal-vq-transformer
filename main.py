# import os
# os.environ["WANDB_SYMLINK"] = "false"   # Windows fix

# import torch
# import torchaudio

# from config import ModelConfig
# from model import VocalSoundTransformer
# from generation import generate_vocal_sound
# from audio_utils import AudioProcessor



# def main():
#     device = torch.device("cpu")
#     print("Using device:", device)

#     # ---- LOAD CONFIG ----
#     config = ModelConfig()

#     # ---- INIT MODEL ----
#     model = VocalSoundTransformer(config).to(device)

#     # ---- LOAD TRAINED CHECKPOINT ----
#     ckpt_path = "checkpoints/best_model.pt"
#     checkpoint = torch.load(ckpt_path, map_location=device)

#     # IMPORTANT: your checkpoint stores a dict
#     model.load_state_dict(checkpoint["model"])
#     print("Model weights loaded")

#     # ---- GENERATE AUDIO ----
#     label = 0  # change if you have multiple classes

#     processor = AudioProcessor(config)

#     mel = generate_vocal_sound(model, label, config, device)
#     audio = processor.reconstruct_audio(mel.squeeze(0))


#     torchaudio.save(
#         "generated_debug.wav",
#         audio.unsqueeze(0),
#         sample_rate=config.sample_rate
#     )

#     print("Saved generated_debug.wav")


# if __name__ == "__main__":
#     main()
