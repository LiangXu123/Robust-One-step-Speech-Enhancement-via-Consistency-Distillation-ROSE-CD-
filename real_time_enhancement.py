from sgmse.util.other import pad_spec
from sgmse.model import ScoreModel
import glob
import torch
import numpy as np
from tqdm import tqdm
from os import makedirs
from soundfile import write
from torchaudio import load
from os.path import join, dirname
from argparse import ArgumentParser
from librosa import resample
import math
from collections import deque

# Set CUDA architecture list
from sgmse.util.other import set_torch_cuda_arch_list
set_torch_cuda_arch_list()

class CausalRealTimeAudioEnhancer:
    def __init__(self, model_ckpt, device="cuda", chunk_length_ms=20, 
                 sampler_type="pc", corrector="ald", corrector_steps=1, 
                 snr=0.5, N=30, t_eps=0.03, buffer_chunks=3):
        """
        Initialize causal real-time audio enhancer (no future data access)
        
        Args:
            model_ckpt: Path to model checkpoint
            device: Device to use for inference
            chunk_length_ms: Length of each chunk in milliseconds (20ms for real-time)
                             if -1, process entire audio at once
            sampler_type: Sampler type for the PC sampler
            corrector: Corrector class for the PC sampler
            corrector_steps: Number of corrector steps
            snr: SNR value for Langevin dynamics
            N: Number of reverse steps
            t_eps: Minimum process time
            buffer_chunks: Number of past chunks to keep for context (affects latency)
        """
        self.device = device
        self.chunk_length_ms = chunk_length_ms
        self.sampler_type = sampler_type
        self.corrector = corrector
        self.corrector_steps = corrector_steps
        self.snr = snr
        self.N = N
        self.t_eps = t_eps
        self.buffer_chunks = buffer_chunks
        
        # Load model
        self.model = ScoreModel.load_from_checkpoint(
            model_ckpt, map_location=device, strict=False)
        self.model.t_eps = t_eps
        self.model.eval()
        
        # Set target sample rate and padding mode based on model
        if self.model.backbone == 'ncsnpp_48k':
            self.target_sr = 48000
            self.pad_mode = "reflection"
        elif self.model.backbone == 'ncsnpp_v2':
            self.target_sr = 16000
            self.pad_mode = "reflection"
        else:
            self.target_sr = 16000
            self.pad_mode = "zero_pad"
            
        # Calculate chunk parameters for causal real-time processing if chunk_length_ms != -1
        if chunk_length_ms == -1:
            self.chunk_samples = None  # Will process full audio
        else:
            self.chunk_samples = int((chunk_length_ms / 1000.0) * self.target_sr)
        
        # Initialize causal processing buffers
        if buffer_chunks > 0:
            self.input_buffer = deque(maxlen=buffer_chunks)  # Past input chunks
        else:
            self.input_buffer = None
        self.is_first_chunk = True
        
        self.context_samples = (self.chunk_samples * buffer_chunks) if self.chunk_samples else 0
        
        # print(f"Model backbone: {self.model.backbone}")
        # print(f"Target sample rate: {self.target_sr}")
        # if self.chunk_length_ms == -1:
        #     print(f"Chunk length: full audio processing")
        # else:
        #     print(f"Chunk length: {chunk_length_ms}ms ({self.chunk_samples} samples)")
        # print(f"Buffer chunks: {buffer_chunks} (context: {buffer_chunks * chunk_length_ms}ms)" if buffer_chunks > 0 else "Buffer chunks: 0 (no context buffering)")
        # if self.chunk_length_ms == -1:
        #     print("Processing mode: full audio processing (no chunking)")
        # else:
        #     print(f"Total latency: {chunk_length_ms}ms processing" + (f" + {chunk_length_ms}ms buffering" if buffer_chunks > 0 else ""))
        # print("CAUSAL MODE: No future data access")

    def _enhance_context_chunk(self, context_audio):
        """
        Enhance audio using context from past chunks only (causal)
        
        Args:
            context_audio: Tensor of shape [1, samples] containing current + past chunks
            
        Returns:
            Enhanced audio for the CURRENT chunk only
        """
        if context_audio.size(1) == 0:
            return torch.zeros(1, self.chunk_samples if self.chunk_samples else context_audio.size(1))
            
        # Normalize
        norm_factor = context_audio.abs().max()
        if norm_factor > 0:
            context_audio = context_audio / norm_factor
        else:
            return torch.zeros(1, self.chunk_samples if self.chunk_samples else context_audio.size(1))
        
        # Prepare DNN input
        Y = torch.unsqueeze(self.model._forward_transform(
            self.model._stft(context_audio.to(self.device))), 0)
        try:
            Y = pad_spec(Y, mode=self.pad_mode)
        except RuntimeError as e:
            # print(f"[Warning] pad_spec failed with mode={self.pad_mode}, trying zero_pad instead.")
            Y = pad_spec(Y, mode="zero_pad")

        
        # Reverse sampling
        if self.model.sde.__class__.__name__ == 'OUVESDE':
            if self.sampler_type == 'pc':
                sampler = self.model.get_pc_sampler(
                    'reverse_diffusion', self.corrector, Y.to(self.device), 
                    N=self.N, corrector_steps=self.corrector_steps, snr=self.snr)
            elif self.sampler_type == 'stochastic':
                sampler = self.model.get_stochastic_sampler(
                    self.model, Y.to(self.device), N=self.N, snr=self.snr)
            elif self.sampler_type == 'ode':
                sampler = self.model.get_ode_sampler(Y.to(self.device), N=self.N)
            else:
                raise ValueError(f"Sampler type {self.sampler_type} not supported")
        elif self.model.sde.__class__.__name__ == 'SBVESDE':
            sampler_type = 'ode' if self.sampler_type == 'pc' else self.sampler_type
            sampler = self.model.get_sb_sampler(
                sde=self.model.sde, y=Y.cuda(), sampler_type=sampler_type)
        else:
            raise ValueError(f"SDE {self.model.sde.__class__.__name__} not supported")
        
        sample, _ = sampler()
        
        # Backward transform to time domain
        enhanced_context = self.model.to_audio(sample.squeeze(), context_audio.size(1))
        
        if self.chunk_samples is None:
            # Full audio processing, return entire enhanced audio
            current_chunk = enhanced_context
        else:
            # Extract only the CURRENT chunk (last chunk_samples)
            if enhanced_context.size(0) >= self.chunk_samples:
                current_chunk = enhanced_context[-self.chunk_samples:]
            else:
                current_chunk = torch.nn.functional.pad(
                    enhanced_context, (0, self.chunk_samples - enhanced_context.size(0)))
        
        # Renormalize
        if norm_factor > 0:
            current_chunk = current_chunk * norm_factor
            
        return current_chunk.unsqueeze(0)  # Return [1, chunk_samples] or [1, full_audio_length]

    def process_chunk(self, audio_chunk):
        """
        Process a single audio chunk in real-time (causal)
        
        Args:
            audio_chunk: Tensor of shape [1, chunk_samples] - NEW incoming audio
            
        Returns:
            Enhanced audio chunk of the same size
        """
        if self.chunk_length_ms == -1:
            # Process entire audio at once (no chunking)
            return self._enhance_context_chunk(audio_chunk)
        
        if self.buffer_chunks == 0:
            # No buffering - process each chunk independently
            context_audio = audio_chunk
        else:
            # Add current chunk to input buffer
            self.input_buffer.append(audio_chunk.clone())
            
            # Create context using only past + current chunks (no future)
            if len(self.input_buffer) == 1:
                # First chunk - process as is
                context_audio = audio_chunk
            else:
                # Concatenate past chunks + current chunk for context
                context_chunks = list(self.input_buffer)
                context_audio = torch.cat(context_chunks, dim=1)
        
        # Enhance using causal context
        enhanced_chunk = self._enhance_context_chunk(context_audio)
        
        self.is_first_chunk = False
        
        return enhanced_chunk

    def reset_buffers(self):
        """Reset all buffers for new audio stream"""
        if self.input_buffer is not None:
            self.input_buffer.clear()
        self.is_first_chunk = True

    def enhance_file_causal(self, input_file, output_file):
        """
        Enhance an entire audio file using causal chunk-based processing
        
        Args:
            input_file: Path to input audio file
            output_file: Path to output enhanced audio file
        """
        # print(f"Enhancing {input_file} (CAUSAL MODE)...")
        
        # Reset buffers for new file
        self.reset_buffers()
        
        # Load audio
        y, sr = load(input_file)
        
        # Resample if necessary
        if sr != self.target_sr:
            y = torch.tensor(resample(y.numpy(), orig_sr=sr, target_sr=self.target_sr))
        
        T_orig = y.size(1)
        # print(f"Original audio shape: {y.shape}")
        
        if self.chunk_length_ms == -1:
            # Process entire audio at once
            enhanced_audio = self.process_chunk(y)
            enhanced_audio = enhanced_audio.squeeze(0)
            # Trim if longer than original
            if enhanced_audio.size(0) > T_orig:
                enhanced_audio = enhanced_audio[:T_orig]
        else:
            # Calculate number of chunks (non-overlapping for causal processing)
            num_chunks = math.ceil(T_orig / self.chunk_samples)
            # print(f"Processing {num_chunks} chunks causally...")
            
            enhanced_chunks = []
            
            # Process each chunk causally (one by one, in order)
            # for i in tqdm(range(num_chunks), desc="Processing chunks causally"):
            for i in range(num_chunks):
                start_idx = i * self.chunk_samples
                end_idx = min(start_idx + self.chunk_samples, T_orig)
                
                # Extract current chunk
                chunk = y[:, start_idx:end_idx]
                
                # Pad last chunk if necessary
                if chunk.size(1) < self.chunk_samples:
                    padding = self.chunk_samples - chunk.size(1)
                    chunk = torch.nn.functional.pad(chunk, (0, padding), mode='constant', value=0)
                
                # Process chunk causally
                enhanced_chunk = self.process_chunk(chunk)
                
                # Remove padding from last chunk
                if end_idx < start_idx + self.chunk_samples:
                    actual_length = end_idx - start_idx
                    enhanced_chunk = enhanced_chunk[:, :actual_length]
                
                enhanced_chunks.append(enhanced_chunk.cpu())
            
            # Concatenate all enhanced chunks (no overlap-add needed for causal)
            # print("Concatenating causal results...")
            enhanced_audio = torch.cat(enhanced_chunks, dim=1).squeeze(0)
            
            # Trim to original length
            if enhanced_audio.size(0) > T_orig:
                enhanced_audio = enhanced_audio[:T_orig]
        
        # Write enhanced audio
        makedirs(dirname(output_file), exist_ok=True)
        write(output_file, enhanced_audio.cpu().numpy(), self.target_sr)
        # print(f"Enhanced audio saved to {output_file}")
        
        return enhanced_audio

def main():
    parser = ArgumentParser()
    parser.add_argument("--test_dir", type=str, 
                        default='/workspace/exp_code/Eval_waspaa2025/RTF-test/10audio_each1s',
                        help='Directory containing the test data')
    parser.add_argument("--enhanced_dir", type=str, 
                        default='/workspace/exp_code/Eval_waspaa2025/RTF-test/tmp',
                        help='Directory containing the enhanced data')
    parser.add_argument("--ckpt", 
                        default="/workspace/exp_code/sgmse/logs/M3_gpu2_bs8_precond/tsfzrd6w/last.ckpt", 
                        type=str, help='Path to model checkpoint')
    parser.add_argument("--chunk_length_ms", type=int, default=260,
                        help="Length of each chunk in milliseconds (20ms for real-time), -1 for full audio")
    parser.add_argument("--buffer_chunks", type=int, default=0,
                        help="Number of past chunks to keep for context (affects latency)")
    parser.add_argument("--sampler_type", type=str, default="pc",
                        help="Sampler type for the PC sampler")
    parser.add_argument("--corrector", type=str, choices=("ald", "langevin", "none"), 
                        default="ald", help="Corrector class for the PC sampler")
    parser.add_argument("--corrector_steps", type=int, default=1, 
                        help="Number of corrector steps")
    parser.add_argument("--snr", type=float, default=0.5,
                        help="SNR value for (annealed) Langevin dynamics")
    parser.add_argument("--N", type=int, default=30,
                        help="Number of reverse steps")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to use for inference")
    parser.add_argument("--t_eps", type=float, default=0.03,
                        help="The minimum process time")
    
    args = parser.parse_args()
    
    # Initialize causal enhancer
    enhancer = CausalRealTimeAudioEnhancer(
        model_ckpt=args.ckpt,
        device=args.device,
        chunk_length_ms=args.chunk_length_ms,
        buffer_chunks=args.buffer_chunks,
        sampler_type=args.sampler_type,
        corrector=args.corrector,
        corrector_steps=args.corrector_steps,
        snr=args.snr,
        N=args.N,
        t_eps=args.t_eps
    )
    
    # Get list of audio files
    audio_files = []
    audio_files += sorted(glob.glob(join(args.test_dir, '*.wav')))
    audio_files += sorted(glob.glob(join(args.test_dir, '**', '*.wav')))
    audio_files += sorted(glob.glob(join(args.test_dir, '*.flac')))
    audio_files += sorted(glob.glob(join(args.test_dir, '**', '*.flac')))
    
    # audio_files = audio_files[:1]
    # print(f"Found {len(audio_files)} audio files to process")
    
    # Process each file
    for audio_file in tqdm(audio_files, desc="Processing files"):
        filename = audio_file.replace(args.test_dir, "")
        filename = filename[1:] if filename.startswith("/") else filename
        
        output_file = join(args.enhanced_dir, filename)
        enhancer.enhance_file_causal(audio_file, output_file)

if __name__ == '__main__':
    main()
