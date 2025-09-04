# Whisper ONNX 推理代码

这是一个独立的 Whisper ONNX 推理实现，参考了 `optimum/onnxruntime/modeling_seq2seq.py` 的设计，但最小化了对外部库的依赖。

## 特点

- ✅ 不需要 `use_io_binding`（按要求）
- ✅ 最小化依赖，不需要 transformers 库进行推理
- ✅ 所有代码在一个文件中
- ✅ 支持编码器-解码器架构
- ✅ 支持 past key values 缓存以加速生成
- ✅ 支持 merged decoder（合并的解码器）
- ✅ 支持多种生成策略（贪婪、采样、top-k、top-p）

## 依赖

推理时只需要：
```bash
pip install numpy onnxruntime
```

如果需要音频处理功能：
```bash
pip install librosa  # 可选，用于音频加载和处理
```

## 使用方法

### 1. 导出 Whisper 模型到 ONNX

首先需要将 Whisper 模型导出为 ONNX 格式。可以使用 optimum-cli：

```bash
# 安装 optimum
pip install optimum[onnxruntime]

# 导出模型（以 whisper-base 为例）
optimum-cli export onnx --model openai/whisper-base whisper_onnx_model/

# 或者导出带缓存的版本（推荐，生成速度更快）
optimum-cli export onnx --model openai/whisper-base --task automatic-speech-recognition-with-past whisper_onnx_model/
```

### 2. 基本使用

```python
from whisper_onnx_inference import WhisperONNX, prepare_input_features
import numpy as np

# 加载模型
model = WhisperONNX.from_pretrained(
    "whisper_onnx_model",
    providers=['CPUExecutionProvider'],  # 或 'CUDAExecutionProvider' for GPU
    use_cache=True  # 使用缓存加速生成
)

# 准备音频输入（这里使用示例音频）
# 实际使用时，可以从文件加载：audio = load_audio("path/to/audio.wav")
audio = np.random.randn(16000 * 5).astype(np.float32)  # 5秒的音频

# 准备输入特征
input_features = prepare_input_features(audio, sample_rate=16000)

# 生成转录
generated_ids = model.generate(
    input_features,
    max_new_tokens=100,
    temperature=0.0,  # 使用贪婪解码
    do_sample=False
)

print(f"Generated token IDs: {generated_ids}")
```

### 3. 高级使用

```python
# 使用采样和 top-k/top-p
generated_ids = model.generate(
    input_features,
    max_new_tokens=100,
    temperature=0.8,
    do_sample=True,
    top_k=50,
    top_p=0.95,
    repetition_penalty=1.2
)

# 使用 CUDA 提供程序
model = WhisperONNX.from_pretrained(
    "whisper_onnx_model",
    providers=['CUDAExecutionProvider', 'CPUExecutionProvider'],
    provider_options=[
        {'device_id': 0},  # CUDA 设备 ID
        {}  # CPU 不需要选项
    ],
    use_cache=True
)
```

### 4. 从音频文件转录

```python
from whisper_onnx_inference import WhisperONNX, load_audio, prepare_input_features

# 加载模型
model = WhisperONNX.from_pretrained("whisper_onnx_model")

# 从文件加载音频（需要 librosa）
audio = load_audio("speech.wav", sample_rate=16000)

# 准备输入
input_features = prepare_input_features(audio)

# 生成转录
generated_ids = model.generate(input_features)

# 解码（需要实际的 tokenizer，这里只是示例）
# 在实际使用中，应该使用 Whisper 的 tokenizer
# from transformers import WhisperTokenizer
# tokenizer = WhisperTokenizer.from_pretrained("openai/whisper-base")
# transcription = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
```

## 主要类和函数

### WhisperONNX
主要的模型类，处理编码器-解码器架构的 ONNX 推理。

**主要方法：**
- `from_pretrained()`: 从目录加载 ONNX 模型
- `encode()`: 编码音频特征
- `decode()`: 解码生成 token
- `generate()`: 自回归生成文本

### WhisperONNXEncoder
编码器包装类，处理音频特征的编码。

### WhisperONNXDecoder
解码器包装类，支持带缓存和不带缓存的解码。

### 音频处理函数
- `load_audio()`: 加载音频文件
- `log_mel_spectrogram()`: 计算对数梅尔频谱图
- `prepare_input_features()`: 准备模型输入特征

## 架构说明

该实现支持三种解码器配置：

1. **标准解码器**：不使用缓存，每次生成都重新计算所有注意力
2. **带缓存的解码器**：使用 past key values 缓存，加速生成
3. **合并解码器**：将带缓存和不带缓存的逻辑合并到一个模型中

代码会自动检测导出的 ONNX 模型类型并选择合适的配置。

## 性能优化

1. **使用缓存**：设置 `use_cache=True` 以使用 past key values 缓存
2. **使用 GPU**：使用 `CUDAExecutionProvider` 进行 GPU 加速
3. **批处理**：代码支持批量推理，可以同时处理多个音频

## 限制

1. **Tokenizer**：示例中的 tokenizer 是简化版本，实际使用需要完整的 Whisper tokenizer
2. **音频处理**：音频处理功能需要额外安装 librosa
3. **语言检测**：未实现自动语言检测功能

## 与原始 modeling_seq2seq.py 的主要区别

1. **无 transformers 依赖**：推理时不需要 transformers 库
2. **无 IO Binding**：按要求移除了 IO Binding 支持
3. **简化的配置**：直接从 config.json 读取必要参数
4. **独立文件**：所有代码在一个文件中，便于部署

## 测试

运行示例代码：
```bash
python3 whisper_onnx_inference.py
```

这将运行一个简单的测试，使用虚拟音频数据测试模型加载和推理流程。

## License

此代码参考了 HuggingFace Optimum 的实现，遵循 Apache 2.0 许可证。