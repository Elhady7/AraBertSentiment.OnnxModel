using System;
using System.Collections.Generic;
using System.Text;

namespace AraBertSentiment.OnnxModel.Models
{
    public class SentimentResultModel

    {
        public string Label { get; set; } = string.Empty;
        public float Confidence { get; set; }
    }
}
