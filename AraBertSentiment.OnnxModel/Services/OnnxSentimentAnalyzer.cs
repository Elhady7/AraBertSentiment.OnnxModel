using System;
using System.IO;
using System.Linq;
using System.Reflection;
using System.Collections.Generic;
using System.Threading.Tasks;
using AraBertSentiment.OnnxModel.Models;
using AraBertSentiment.OnnxModel.Interfaces;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.Tokenizers;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace AraBertSentiment.OnnxModel.Services
{
    public class OnnxSentimentAnalyzer : IOnnxSentimentAnalyzer, IDisposable
    {
        private InferenceSession _session = null!;
        private BertTokenizer _tokenizer = null!;
        private readonly string[] _labels = { "Positive", "Negative", "Neutral" };
        private readonly string _customDirectory;

        public OnnxSentimentAnalyzer() : this(Path.Combine(Path.GetDirectoryName(Assembly.GetExecutingAssembly().Location)!, "Resources"))
        {
        }

        public OnnxSentimentAnalyzer(string modelDirectory)
        {
            _customDirectory = modelDirectory;
        }

#if NET6_0_OR_GREATER
        public async Task InitializeAsync()
        {
            string modelPath = Path.Combine(_customDirectory, "model.onnx");
            string vocabPath = Path.Combine(_customDirectory, "vocab.txt");

            if (!File.Exists(modelPath))
                throw new FileNotFoundException("model.onnx not found", modelPath);
            if (!File.Exists(vocabPath))
                throw new FileNotFoundException("vocab.txt not found", vocabPath);

            await Task.Run(() =>
            {
                _session = new InferenceSession(modelPath);
                _tokenizer = BertTokenizer.Create(vocabPath);
            }).ConfigureAwait(false);
        }

        public Task<SentimentResultModel> AnalyzeAsync(string text)
        {
            int[] tokenIds = _tokenizer.EncodeToIds(text).ToArray();
            long[] inputIds = tokenIds.Select(id => (long)id).ToArray();
            long[] attentionMask = inputIds.Select(_ => 1L).ToArray();
            long[] tokenTypeIds = inputIds.Select(_ => 0L).ToArray();

            var inputs = new List<NamedOnnxValue>
            {
                NamedOnnxValue.CreateFromTensor("input_ids", new DenseTensor<long>(inputIds, new[] { 1, inputIds.Length })),
                NamedOnnxValue.CreateFromTensor("attention_mask", new DenseTensor<long>(attentionMask, new[] { 1, inputIds.Length })),
                NamedOnnxValue.CreateFromTensor("token_type_ids", new DenseTensor<long>(tokenTypeIds, new[] { 1, inputIds.Length }))
            };

            return Task.Run(() =>
            {
                using var results = _session.Run(inputs);
                float[] rawLogits = results.First().AsTensor<float>().ToArray();
                float[] probabilities = Softmax(rawLogits);
                int classIndex = Array.IndexOf(probabilities, probabilities.Max());

                return new SentimentResultModel
                {
                    Label = _labels[classIndex],
                    Confidence = probabilities[classIndex]
                };
            });
        }

        public void Initialize() => InitializeAsync().GetAwaiter().GetResult();
        public SentimentResultModel Analyze(string text) => AnalyzeAsync(text).GetAwaiter().GetResult();
#else
        public void Initialize()
        {
            string modelPath = Path.Combine(_customDirectory, "model.onnx");
            string vocabPath = Path.Combine(_customDirectory, "vocab.txt");

            if (!File.Exists(modelPath))
                throw new FileNotFoundException("model.onnx not found", modelPath);
            if (!File.Exists(vocabPath))
                throw new FileNotFoundException("vocab.txt not found", vocabPath);

            _session = new InferenceSession(modelPath);
            _tokenizer = BertTokenizer.Create(vocabPath);
        }

        public SentimentResultModel Analyze(string text)
        {
            int[] tokenIds = _tokenizer.EncodeToIds(text).ToArray();
            long[] inputIds = tokenIds.Select(id => (long)id).ToArray();
            long[] attentionMask = inputIds.Select(_ => 1L).ToArray();
            long[] tokenTypeIds = inputIds.Select(_ => 0L).ToArray();

            var inputs = new List<NamedOnnxValue>
            {
                NamedOnnxValue.CreateFromTensor("input_ids", new DenseTensor<long>(inputIds, new[] { 1, inputIds.Length })),
                NamedOnnxValue.CreateFromTensor("attention_mask", new DenseTensor<long>(attentionMask, new[] { 1, inputIds.Length })),
                NamedOnnxValue.CreateFromTensor("token_type_ids", new DenseTensor<long>(tokenTypeIds, new[] { 1, inputIds.Length }))
            };

            using var results = _session.Run(inputs);
            float[] rawLogits = results.First().AsTensor<float>().ToArray();
            float[] probabilities = Softmax(rawLogits);
            int classIndex = Array.IndexOf(probabilities, probabilities.Max());

            return new SentimentResultModel
            {
                Label = _labels[classIndex],
                Confidence = probabilities[classIndex]
            };
        }

        public Task InitializeAsync()
        {
            Initialize();
            return Task.CompletedTask;
        }

        public Task<SentimentResultModel> AnalyzeAsync(string text)
        {
            return Task.FromResult(Analyze(text));
        }
#endif

        private float[] Softmax(float[] logits)
        {
            float max = logits.Max();
            var exp = logits.Select(v => (float)Math.Exp(v - max)).ToArray();
            float sum = exp.Sum();
            return exp.Select(v => v / sum).ToArray();
        }

        public void Dispose() => _session.Dispose();
    }
}
