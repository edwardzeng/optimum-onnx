#!/usr/bin/env python3
"""
Standalone Whisper ONNX inference script.
This script provides a minimal implementation for running Whisper models exported to ONNX format.
No dependency on transformers library for inference, only uses numpy and onnxruntime.
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import onnxruntime as ort


class WhisperONNXEncoder:
    """Whisper encoder for ONNX Runtime inference."""
    
    def __init__(self, session: ort.InferenceSession):
        self.session = session
        self.input_names = [input.name for input in session.get_inputs()]
        self.output_names = [output.name for output in session.get_outputs()]
        
    def __call__(
        self,
        input_features: np.ndarray,
        attention_mask: Optional[np.ndarray] = None
    ) -> Dict[str, np.ndarray]:
        """
        Forward pass through the encoder.
        
        Args:
            input_features: Mel spectrogram features of shape (batch_size, n_mels, sequence_length)
            attention_mask: Optional attention mask
            
        Returns:
            Dictionary with 'last_hidden_state' key containing encoder outputs
        """
        inputs = {"input_features": input_features}
        
        # Add attention mask if it's in the model inputs
        if attention_mask is not None and "attention_mask" in self.input_names:
            inputs["attention_mask"] = attention_mask
            
        outputs = self.session.run(None, inputs)
        
        # Map outputs to dictionary
        output_dict = {}
        for name, output in zip(self.output_names, outputs):
            output_dict[name] = output
            
        return output_dict


class WhisperONNXDecoder:
    """Whisper decoder for ONNX Runtime inference."""
    
    def __init__(
        self,
        session: ort.InferenceSession,
        use_cache: bool = True,
        use_merged: bool = False
    ):
        self.session = session
        self.use_cache = use_cache
        self.use_merged = use_merged
        
        self.input_names = [input.name for input in session.get_inputs()]
        self.output_names = [output.name for output in session.get_outputs()]
        
        # Identify key-value cache inputs/outputs
        self.key_value_input_names = [
            name for name in self.input_names 
            if ("key" in name or "value" in name) and "past" in name
        ]
        self.key_value_output_names = [
            name for name in self.output_names 
            if ("key" in name or "value" in name) and "present" in name
        ]
        
        # For merged models
        self.has_use_cache_branch = "use_cache_branch" in self.input_names
        
    def __call__(
        self,
        input_ids: np.ndarray,
        encoder_hidden_states: np.ndarray,
        encoder_attention_mask: Optional[np.ndarray] = None,
        decoder_attention_mask: Optional[np.ndarray] = None,
        past_key_values: Optional[Tuple[np.ndarray, ...]] = None,
        use_cache_branch: Optional[np.ndarray] = None,
        cache_position: Optional[np.ndarray] = None,
    ) -> Dict[str, Union[np.ndarray, Tuple]]:
        """
        Forward pass through the decoder.
        
        Args:
            input_ids: Decoder input token IDs
            encoder_hidden_states: Encoder output hidden states
            encoder_attention_mask: Optional encoder attention mask
            decoder_attention_mask: Optional decoder attention mask
            past_key_values: Optional past key values for caching
            use_cache_branch: For merged models, indicates which branch to use
            cache_position: Cache position for merged models
            
        Returns:
            Dictionary with 'logits' and optionally 'past_key_values'
        """
        inputs = {
            "input_ids": input_ids,
            "encoder_hidden_states": encoder_hidden_states,
        }
        
        # Add optional inputs if they exist in the model
        if encoder_attention_mask is not None and "encoder_attention_mask" in self.input_names:
            inputs["encoder_attention_mask"] = encoder_attention_mask
            
        if decoder_attention_mask is not None and "decoder_attention_mask" in self.input_names:
            inputs["decoder_attention_mask"] = decoder_attention_mask
            
        if use_cache_branch is not None and "use_cache_branch" in self.input_names:
            inputs["use_cache_branch"] = use_cache_branch
            
        if cache_position is not None and "cache_position" in self.input_names:
            inputs["cache_position"] = cache_position
            
        # Add past key values if provided
        if past_key_values is not None and len(self.key_value_input_names) > 0:
            for name, value in zip(self.key_value_input_names, past_key_values):
                inputs[name] = value
                
        outputs = self.session.run(None, inputs)
        
        # Process outputs
        output_dict = {}
        for name, output in zip(self.output_names, outputs):
            output_dict[name] = output
            
        # Extract past key values if present
        if self.use_cache and len(self.key_value_output_names) > 0:
            past_key_values = tuple(
                output_dict[name] for name in self.key_value_output_names
            )
            output_dict["past_key_values"] = past_key_values
            
        return output_dict


class WhisperONNX:
    """
    Whisper model for ONNX Runtime inference.
    
    This class provides a minimal implementation for running Whisper models
    exported to ONNX format without dependency on the transformers library.
    """
    
    def __init__(
        self,
        encoder_path: Union[str, Path],
        decoder_path: Union[str, Path],
        decoder_with_past_path: Optional[Union[str, Path]] = None,
        config_path: Optional[Union[str, Path]] = None,
        providers: Optional[List[str]] = None,
        provider_options: Optional[List[Dict]] = None,
        use_cache: bool = True,
    ):
        """
        Initialize Whisper ONNX model.
        
        Args:
            encoder_path: Path to encoder ONNX model
            decoder_path: Path to decoder ONNX model
            decoder_with_past_path: Optional path to decoder with past ONNX model
            config_path: Optional path to model config JSON
            providers: ONNX Runtime execution providers
            provider_options: Provider-specific options
            use_cache: Whether to use key-value caching for faster generation
        """
        self.use_cache = use_cache and decoder_with_past_path is not None
        
        # Set default providers
        if providers is None:
            providers = ['CPUExecutionProvider']
            
        # Load ONNX sessions
        self.encoder_session = ort.InferenceSession(
            str(encoder_path),
            providers=providers,
            provider_options=provider_options
        )
        
        self.decoder_session = ort.InferenceSession(
            str(decoder_path),
            providers=providers,
            provider_options=provider_options
        )
        
        self.decoder_with_past_session = None
        if decoder_with_past_path and Path(decoder_with_past_path).exists():
            self.decoder_with_past_session = ort.InferenceSession(
                str(decoder_with_past_path),
                providers=providers,
                provider_options=provider_options
            )
            
        # Check if decoder is merged (has use_cache_branch input)
        decoder_inputs = [input.name for input in self.decoder_session.get_inputs()]
        self.use_merged = "use_cache_branch" in decoder_inputs
        
        # Initialize encoder and decoder wrappers
        self.encoder = WhisperONNXEncoder(self.encoder_session)
        self.decoder = WhisperONNXDecoder(
            self.decoder_session,
            use_cache=self.use_cache and not self.use_merged,
            use_merged=self.use_merged
        )
        
        if self.decoder_with_past_session:
            self.decoder_with_past = WhisperONNXDecoder(
                self.decoder_with_past_session,
                use_cache=True,
                use_merged=False
            )
        else:
            self.decoder_with_past = None
            
        # Load config if provided
        self.config = None
        if config_path and Path(config_path).exists():
            with open(config_path, 'r') as f:
                self.config = json.load(f)
                
        # Set default generation parameters
        self.eos_token_id = 50257  # Default Whisper EOS token
        self.pad_token_id = 50257
        self.decoder_start_token_id = 50258  # Default Whisper decoder start token
        self.max_length = 448  # Default max length for Whisper
        
        # Override with config values if available
        if self.config:
            self.eos_token_id = self.config.get('eos_token_id', self.eos_token_id)
            self.pad_token_id = self.config.get('pad_token_id', self.pad_token_id)
            self.decoder_start_token_id = self.config.get('decoder_start_token_id', self.decoder_start_token_id)
            self.max_length = self.config.get('max_length', self.max_length)
            
    def encode(
        self,
        input_features: np.ndarray,
        attention_mask: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Encode audio features.
        
        Args:
            input_features: Mel spectrogram features
            attention_mask: Optional attention mask
            
        Returns:
            Encoder hidden states
        """
        encoder_outputs = self.encoder(input_features, attention_mask)
        return encoder_outputs["last_hidden_state"]
        
    def decode(
        self,
        input_ids: np.ndarray,
        encoder_hidden_states: np.ndarray,
        encoder_attention_mask: Optional[np.ndarray] = None,
        past_key_values: Optional[Tuple[np.ndarray, ...]] = None,
    ) -> Dict[str, Union[np.ndarray, Tuple]]:
        """
        Decode tokens given encoder hidden states.
        
        Args:
            input_ids: Decoder input token IDs
            encoder_hidden_states: Encoder output hidden states
            encoder_attention_mask: Optional encoder attention mask
            past_key_values: Optional past key values for caching
            
        Returns:
            Dictionary with logits and optionally past_key_values
        """
        # Select appropriate decoder based on cache usage
        if self.use_merged:
            # For merged decoder, handle use_cache_branch
            use_cache_branch = np.array([past_key_values is not None], dtype=bool)
            cache_position = None
            
            if past_key_values is None and self.use_cache:
                # Generate dummy past key values for first step
                batch_size = input_ids.shape[0]
                # This is a simplified version - actual shape depends on model architecture
                dummy_shape = (batch_size, 12, 1, 64)  # batch, heads, seq_len, head_dim
                past_key_values = tuple(
                    np.zeros(dummy_shape, dtype=np.float32)
                    for _ in range(len(self.decoder.key_value_input_names))
                )
                cache_position = np.zeros((1,), dtype=np.int64)
                
            decoder_outputs = self.decoder(
                input_ids=input_ids,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                past_key_values=past_key_values,
                use_cache_branch=use_cache_branch,
                cache_position=cache_position,
            )
        else:
            # Use separate decoder models
            if past_key_values is None or not self.use_cache:
                decoder = self.decoder
            else:
                decoder = self.decoder_with_past if self.decoder_with_past else self.decoder
                
            decoder_outputs = decoder(
                input_ids=input_ids,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                past_key_values=past_key_values,
            )
            
        return decoder_outputs
        
    def generate(
        self,
        input_features: np.ndarray,
        max_new_tokens: Optional[int] = None,
        min_length: int = 0,
        num_beams: int = 1,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        repetition_penalty: float = 1.0,
        do_sample: bool = False,
        attention_mask: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Generate text from audio features.
        
        Args:
            input_features: Mel spectrogram features of shape (batch_size, n_mels, sequence_length)
            max_new_tokens: Maximum number of tokens to generate
            min_length: Minimum length of generated sequence
            num_beams: Number of beams for beam search (1 = greedy)
            temperature: Temperature for sampling
            top_k: Top-k filtering
            top_p: Top-p (nucleus) filtering
            repetition_penalty: Repetition penalty
            do_sample: Whether to use sampling instead of greedy/beam search
            attention_mask: Optional attention mask for encoder
            
        Returns:
            Generated token IDs of shape (batch_size, sequence_length)
        """
        batch_size = input_features.shape[0]
        
        # Set max length
        if max_new_tokens is None:
            max_new_tokens = self.max_length
            
        # Encode input features
        encoder_hidden_states = self.encode(input_features, attention_mask)
        
        # Initialize decoder input with start token
        decoder_input_ids = np.full(
            (batch_size, 1),
            self.decoder_start_token_id,
            dtype=np.int64
        )
        
        # Initialize past key values
        past_key_values = None
        
        # Generate tokens one by one
        generated_tokens = []
        finished = np.zeros(batch_size, dtype=bool)
        
        for step in range(max_new_tokens):
            # Get decoder outputs
            decoder_outputs = self.decode(
                input_ids=decoder_input_ids,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=attention_mask,
                past_key_values=past_key_values,
            )
            
            logits = decoder_outputs["logits"]
            
            # Get next token logits (last position)
            next_token_logits = logits[:, -1, :]
            
            # Apply repetition penalty
            if repetition_penalty != 1.0 and len(generated_tokens) > 0:
                for prev_token in generated_tokens:
                    for batch_idx in range(batch_size):
                        next_token_logits[batch_idx, prev_token[batch_idx]] /= repetition_penalty
                        
            # Apply temperature
            if temperature != 1.0:
                next_token_logits = next_token_logits / temperature
                
            # Apply top-k filtering
            if top_k is not None and top_k > 0:
                indices_to_remove = next_token_logits < np.sort(next_token_logits, axis=-1)[:, -top_k:].min(axis=-1, keepdims=True)
                next_token_logits[indices_to_remove] = -float('inf')
                
            # Apply top-p (nucleus) filtering
            if top_p is not None and top_p < 1.0:
                sorted_logits = np.sort(next_token_logits, axis=-1)[:, ::-1]
                sorted_indices = np.argsort(next_token_logits, axis=-1)[:, ::-1]
                cumulative_probs = np.cumsum(self._softmax(sorted_logits, axis=-1), axis=-1)
                
                # Remove tokens with cumulative probability above the threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                # Keep at least one token
                sorted_indices_to_remove[:, 0] = False
                
                for batch_idx in range(batch_size):
                    indices_to_remove = sorted_indices[batch_idx, sorted_indices_to_remove[batch_idx]]
                    next_token_logits[batch_idx, indices_to_remove] = -float('inf')
                    
            # Sample or take argmax
            if do_sample and temperature > 0:
                probs = self._softmax(next_token_logits, axis=-1)
                next_tokens = np.array([
                    np.random.choice(len(p), p=p) for p in probs
                ], dtype=np.int64)
            else:
                next_tokens = np.argmax(next_token_logits, axis=-1)
                
            # Update finished sequences
            finished |= (next_tokens == self.eos_token_id)
            
            # Store generated tokens
            generated_tokens.append(next_tokens)
            
            # Break if all sequences are finished
            if finished.all() and step >= min_length:
                break
                
            # Prepare next decoder input
            if self.use_cache:
                # Only use the new token as input when using cache
                decoder_input_ids = next_tokens.reshape(batch_size, 1)
                past_key_values = decoder_outputs.get("past_key_values")
            else:
                # Concatenate all tokens
                decoder_input_ids = np.concatenate([
                    decoder_input_ids,
                    next_tokens.reshape(batch_size, 1)
                ], axis=1)
                
        # Stack generated tokens
        if generated_tokens:
            generated = np.stack(generated_tokens, axis=1)
            
            # Add decoder start token at the beginning
            generated = np.concatenate([
                np.full((batch_size, 1), self.decoder_start_token_id, dtype=np.int64),
                generated
            ], axis=1)
        else:
            generated = np.full((batch_size, 1), self.decoder_start_token_id, dtype=np.int64)
            
        return generated
        
    @staticmethod
    def _softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
        """Compute softmax values for array x along specified axis."""
        exp_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
        return exp_x / np.sum(exp_x, axis=axis, keepdims=True)
        
    @classmethod
    def from_pretrained(
        cls,
        model_path: Union[str, Path],
        providers: Optional[List[str]] = None,
        provider_options: Optional[List[Dict]] = None,
        use_cache: bool = True,
    ) -> "WhisperONNX":
        """
        Load a Whisper ONNX model from a directory.
        
        Args:
            model_path: Path to directory containing ONNX files
            providers: ONNX Runtime execution providers
            provider_options: Provider-specific options
            use_cache: Whether to use key-value caching
            
        Returns:
            WhisperONNX model instance
        """
        model_path = Path(model_path)
        
        # Find ONNX files
        encoder_path = model_path / "encoder_model.onnx"
        decoder_path = model_path / "decoder_model.onnx"
        decoder_with_past_path = model_path / "decoder_with_past_model.onnx"
        
        # Check for merged decoder
        decoder_merged_path = model_path / "decoder_model_merged.onnx"
        if decoder_merged_path.exists():
            decoder_path = decoder_merged_path
            decoder_with_past_path = None
            
        # Find config file
        config_path = model_path / "config.json"
        
        if not encoder_path.exists():
            raise FileNotFoundError(f"Encoder model not found at {encoder_path}")
        if not decoder_path.exists():
            raise FileNotFoundError(f"Decoder model not found at {decoder_path}")
            
        return cls(
            encoder_path=encoder_path,
            decoder_path=decoder_path,
            decoder_with_past_path=decoder_with_past_path if decoder_with_past_path and decoder_with_past_path.exists() else None,
            config_path=config_path if config_path.exists() else None,
            providers=providers,
            provider_options=provider_options,
            use_cache=use_cache,
        )


# Audio processing utilities
def load_audio(audio_path: Union[str, Path], sample_rate: int = 16000) -> np.ndarray:
    """
    Load audio file and resample to target sample rate.
    
    Args:
        audio_path: Path to audio file
        sample_rate: Target sample rate (Whisper uses 16kHz)
        
    Returns:
        Audio waveform as numpy array
    """
    try:
        import librosa
        audio, sr = librosa.load(audio_path, sr=sample_rate)
        return audio
    except ImportError:
        raise ImportError("librosa is required for audio loading. Install it with: pip install librosa")


def log_mel_spectrogram(
    audio: np.ndarray,
    n_mels: int = 80,
    n_fft: int = 400,
    hop_length: int = 160,
    sample_rate: int = 16000,
    padding: int = 0,
) -> np.ndarray:
    """
    Compute log-Mel spectrogram from audio.
    
    Args:
        audio: Audio waveform
        n_mels: Number of Mel bands
        n_fft: FFT window size
        hop_length: Hop length for STFT
        sample_rate: Sample rate of audio
        padding: Amount of padding to add
        
    Returns:
        Log-Mel spectrogram
    """
    try:
        import librosa
        
        if padding > 0:
            audio = np.pad(audio, (0, padding), mode='constant')
            
        # Compute STFT
        stft = librosa.stft(
            audio,
            n_fft=n_fft,
            hop_length=hop_length,
            window='hann',
            center=False
        )
        
        # Compute magnitude
        magnitude = np.abs(stft) ** 2
        
        # Create Mel filterbank
        mel_filters = librosa.filters.mel(
            sr=sample_rate,
            n_fft=n_fft,
            n_mels=n_mels
        )
        
        # Apply Mel filterbank
        mel_spec = mel_filters @ magnitude
        
        # Convert to log scale
        log_mel_spec = np.log10(np.maximum(mel_spec, 1e-10))
        
        # Normalize (Whisper uses specific normalization)
        log_mel_spec = np.maximum(log_mel_spec, log_mel_spec.max() - 8.0)
        log_mel_spec = (log_mel_spec + 4.0) / 4.0
        
        return log_mel_spec
        
    except ImportError:
        raise ImportError("librosa is required for audio processing. Install it with: pip install librosa")


def prepare_input_features(
    audio: Union[np.ndarray, str, Path],
    sample_rate: int = 16000,
    chunk_length: int = 30,
    n_mels: int = 80,
) -> np.ndarray:
    """
    Prepare audio input features for Whisper model.
    
    Args:
        audio: Audio waveform or path to audio file
        sample_rate: Sample rate of audio
        chunk_length: Length of audio chunks in seconds (Whisper uses 30s)
        n_mels: Number of Mel bands
        
    Returns:
        Input features of shape (1, n_mels, n_frames)
    """
    # Load audio if path is provided
    if isinstance(audio, (str, Path)):
        audio = load_audio(audio, sample_rate)
        
    # Pad or trim to chunk length
    n_samples = chunk_length * sample_rate
    if len(audio) > n_samples:
        audio = audio[:n_samples]
    else:
        audio = np.pad(audio, (0, n_samples - len(audio)), mode='constant')
        
    # Compute log-Mel spectrogram
    mel_spec = log_mel_spectrogram(
        audio,
        n_mels=n_mels,
        sample_rate=sample_rate
    )
    
    # Add batch dimension
    mel_spec = mel_spec[np.newaxis, :, :]
    
    return mel_spec.astype(np.float32)


# Simple tokenizer utilities (for demonstration)
class SimpleTokenizer:
    """
    Simple tokenizer for demonstration purposes.
    In production, use the actual Whisper tokenizer.
    """
    
    def __init__(self, vocab_size: int = 51865):
        self.vocab_size = vocab_size
        self.pad_token_id = 50257
        self.eos_token_id = 50257
        self.decoder_start_token_id = 50258
        
        # Create a simple vocabulary (for demo only)
        self.vocab = {}
        self.inverse_vocab = {}
        
        # Special tokens
        special_tokens = {
            50257: "<|endoftext|>",
            50258: "<|startoftranscript|>",
            50259: "<|en|>",
            50260: "<|zh|>",
            50261: "<|de|>",
            50262: "<|es|>",
            50263: "<|ru|>",
            50264: "<|ko|>",
            50265: "<|fr|>",
            50266: "<|ja|>",
            50267: "<|pt|>",
            50268: "<|tr|>",
            50269: "<|pl|>",
        }
        
        for token_id, token in special_tokens.items():
            self.vocab[token] = token_id
            self.inverse_vocab[token_id] = token
            
    def decode(self, token_ids: np.ndarray, skip_special_tokens: bool = True) -> List[str]:
        """
        Decode token IDs to text.
        
        Args:
            token_ids: Token IDs to decode
            skip_special_tokens: Whether to skip special tokens
            
        Returns:
            List of decoded strings
        """
        if len(token_ids.shape) == 1:
            token_ids = token_ids[np.newaxis, :]
            
        texts = []
        for tokens in token_ids:
            text_tokens = []
            for token_id in tokens:
                if skip_special_tokens and token_id in [self.pad_token_id, self.eos_token_id, self.decoder_start_token_id]:
                    continue
                    
                if token_id in self.inverse_vocab:
                    text_tokens.append(self.inverse_vocab[token_id])
                else:
                    # For demo, just return token ID as string
                    text_tokens.append(f"<token_{token_id}>")
                    
            texts.append(" ".join(text_tokens))
            
        return texts


# Example usage
def main():
    """Example usage of WhisperONNX model."""
    
    print("Whisper ONNX Inference Example")
    print("=" * 50)
    
    # Example paths (update these to your actual model paths)
    model_path = "./whisper_onnx_model"
    
    # Check if model exists
    if not Path(model_path).exists():
        print(f"Model not found at {model_path}")
        print("Please export a Whisper model to ONNX format first.")
        print("\nYou can export a model using optimum-cli:")
        print("  optimum-cli export onnx --model openai/whisper-base whisper_onnx_model/")
        return
        
    try:
        # Load model
        print(f"Loading model from {model_path}...")
        model = WhisperONNX.from_pretrained(
            model_path,
            providers=['CPUExecutionProvider'],  # Use CUDA if available
            use_cache=True
        )
        print("Model loaded successfully!")
        
        # Create dummy audio input for demonstration
        print("\nCreating dummy audio input...")
        sample_rate = 16000
        duration = 5  # seconds
        n_samples = sample_rate * duration
        
        # Generate a simple sine wave as dummy audio
        t = np.linspace(0, duration, n_samples)
        frequency = 440  # A4 note
        audio = np.sin(2 * np.pi * frequency * t).astype(np.float32)
        
        # Prepare input features
        print("Preparing input features...")
        input_features = prepare_input_features(audio, sample_rate=sample_rate)
        print(f"Input features shape: {input_features.shape}")
        
        # Generate transcription
        print("\nGenerating transcription...")
        generated_ids = model.generate(
            input_features,
            max_new_tokens=50,
            temperature=0.0,  # Use greedy decoding
            do_sample=False
        )
        print(f"Generated token IDs shape: {generated_ids.shape}")
        
        # Decode tokens (using simple tokenizer for demo)
        tokenizer = SimpleTokenizer()
        transcription = tokenizer.decode(generated_ids, skip_special_tokens=True)
        print(f"\nTranscription: {transcription[0]}")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()