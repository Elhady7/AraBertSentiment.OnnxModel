# AraBertSentiment.OnnxModel

**Full Arabic Sentiment Analyzer powered by BERT & ONNX, usable across all .NET platforms.**

![NuGet] https://www.nuget.org/packages/AraBertSentiment.OnnxModel/

---

## 🧠 What is it?
A lightweight, cross-platform Arabic Sentiment Analyzer built using:
- Pretrained [AraBERT model]([https://arxiv.org/abs/2003.00104](https://huggingface.co/CAMeL-Lab/bert-base-arabic-camelbert-da-sentiment))
- ONNX runtime
- .NET 6.0+ & netstandard2.0 compatibility

---

## 🚀 How to Install

```bash
dotnet add package AraBertSentiment.OnnxModel
```

---

## 📦 Download the Model Assets
You must manually download the model assets once and place them in a directory of your choice.

📥 [Download Model Assets (ZIP)](https://github.com/Elhady7/AraBertSentimentModel/releases/download/v1.0.0/ModelAraBertOnnxSentimentAssets.zip)
📝 Then extract it and pass its directory to the constructor:

```csharp
var analyzer = new OnnxSentimentAnalyzer("C:\\Path\\To\\ExtractedAssets");
await analyzer.InitializeAsync();
```

---

## 💡 Usage

```csharp
var analyzer = new OnnxSentimentAnalyzer("C:\\SentimentModel");
await analyzer.InitializeAsync();

var result = await analyzer.AnalyzeAsync("هذا المنتج رائع للغاية!");
Console.WriteLine($"Sentiment: {result.Label}, Confidence: {result.Confidence:P2}");
```

**Labels:**
- Positive
- Negative
- Neutral
with Percentage gives you the control to detect at
 which level you need the Sentiment (Confidence)

---

## 🔧 Frameworks Supported
- .NET Standard 2.0
- .NET 6.0
- .NET 8.0
- .NET 9.0

With support for `async/await` in modern targets and sync fallback in older ones.

---

## 📁 File Structure
```
Model Directory
│
├── model.onnx
├── vocab.txt
├── tokenizer_config.json
├── special_tokens_map.json
└── config.json
```

---

## 📃 License
MIT License

---

## 🙋‍♂️ Maintainer
**Hady Salah** – [GitHub](https://github.com/Elhady7)

---

Feel free to open issues or submit PRs if you'd like to contribute or report bugs!
