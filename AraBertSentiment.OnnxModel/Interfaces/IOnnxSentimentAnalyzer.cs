using AraBertSentiment.OnnxModel.Models;
using System;
using System.Collections.Generic;
using System.Text;
using System.Threading.Tasks;

namespace AraBertSentiment.OnnxModel.Interfaces
{
    public interface IOnnxSentimentAnalyzer
    {
        public SentimentResultModel Analyze(string text);
        public Task<SentimentResultModel> AnalyzeAsync(string text);

        public Task InitializeAsync();
        public void Initialize();
    }
}
