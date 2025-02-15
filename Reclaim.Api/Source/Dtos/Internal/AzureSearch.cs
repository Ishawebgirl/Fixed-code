using Azure.Search.Documents.Indexes;
using Azure.Search.Documents.Indexes.Models;
using Microsoft.Extensions.Diagnostics.HealthChecks;
using Newtonsoft.Json;
using System.Collections.Generic;
using System.Linq;

namespace Reclaim.Api.Dtos;

public class AzureSearch
{
    public class QueryResult
    {
        public string Answer { get; set; }
        public List<VectorDocumentReference> References { get; set; }
    }

    public class AnswerExtractionResult
    {
        [JsonProperty("answer")]
        public string Answer { get; set; }

        [JsonProperty("lineNumbers")]
        public List<int> LineNumbers { get; set; } = new List<int>();
    }

    public class VectorDocument
    {
        [SimpleField(IsKey = true, IsFilterable = true)]
        [JsonProperty("id")]
        public string ID { get; set; }

        [SimpleField(IsFilterable = true, IsSortable = true)]
        [JsonProperty("DocumentID")]
        public int DocumentID { get; set; }

        [SimpleField(IsFilterable = true, IsSortable = true)]
        [JsonProperty("ClaimID")]
        public int ClaimID { get; set; }

        [SimpleField(IsFilterable = true, IsSortable = true, IsFacetable = true)]
        [JsonProperty("InvestigatorID")]
        public int? InvestigatorID { get; set; }

        [SearchableField]
        [JsonProperty("FileName")]
        public string FileName { get; set; }

        [SimpleField(IsFilterable = true, IsSortable = true)]
        [JsonProperty("PageNumber")]
        public int PageNumber { get; set; }

        [SimpleField(IsFilterable = false, IsSortable = false, IsFacetable = false)]
        [JsonProperty("Embedding")]
        public float[] Embedding { get; set; }

        [SearchableField]
        [JsonProperty("Content")]
        public string Content { get; set; }

        [SearchableField]
        [JsonProperty("BoundingBoxes")]
        public List<string> BoundingBoxes { get; set; } = new List<string>();
    }

    public class VectorDocumentReference
    {
        public int DocumentID { get; set; }
        public string FileName { get; set; }
        public int PageNumber { get; set; }
        public List<string> BoundingBoxes { get; set; }
    }

    public class VectorDocumentWithNumberedLines
    {
        public VectorDocument VectorDocument { get; set; }
        public List<NumberedLine> NumberedLines { get; set; }
    }

    public class NumberedLine
    {
        public int Number { get; set; }
        public string Line { get; set; }
    }
}