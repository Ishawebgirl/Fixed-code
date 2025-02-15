// Enhanced RAG Chatbot with improved validation and bounding box handling
// Last updated: 2024-02-15
using Azure;
using Azure.AI.DocumentIntelligence;
using Azure.AI.OpenAI;
using Azure.Search.Documents;
using Newtonsoft.Json;
using Reclaim.Api.Model;
using System.Runtime;
using System.Text;
using AzureSearch = Reclaim.Api.Dtos.AzureSearch;

namespace Reclaim.Api.Services;

public class SearchService
{
    private readonly LogService _logService;
    private readonly OpenAIClient _openAIClient;
    private readonly DocumentIntelligenceClient _documentIntelligenceClient;
    private readonly SearchClient _searchClient;

    private static string defaultSystemMessage =
        @"You are an expert in analyzing property claim documents for home insurers and answering
        questions based only on the content in the provided text documents, along with line numbers.
        The current date and time is: " + DateTime.UtcNow.ToString("yyyy-MM-dd HH:mm:ss UTC") + @"

        Instructions:
        1. Carefully read through the entire context, paying close attention to line numbers.
        2. Extract the most accurate and complete answer to the question.
        3. For names and addresses:
           - Verify names match standard name patterns
           - Validate addresses against standard address formats
           - Double-check for context clues that confirm identity
        4. For numerical values:
           - Ensure calculations are precise
           - Verify units and currency symbols
           - Cross-reference with related fields
        5. Return a JSON response with two keys:
            - 'answer': Answer to the user's question based on text documents
            - 'lineNumbers': An array of line numbers (1-indexed) where the EXACT answer was found
            - 'confidence': A score between 0 and 1 indicating confidence in the answer

        Important Guidelines:
        - Only cite line numbers where the EXACT information appears
        - For addresses, ensure ALL components (street, city, state, zip) are present
        - For names, verify the context confirms it's the insured party
        - For calculations, show all components used
        - If unsure about any component, mark it as low confidence
        - Do not guess or infer missing information

        Response Format:
        {
            ""answer"": ""answer to user's question"",
            ""lineNumbers"": [1, 2, 3],
            ""confidence"": 0.95
        }";

    public SearchService(LogService logService)
    {
        _logService = logService;

        _openAIClient = new OpenAIClient(new Uri(Setting.AzureOpenAIEndpoint), new AzureKeyCredential(Setting.AzureOpenAIKey));
        _documentIntelligenceClient = new DocumentIntelligenceClient(new Uri(Setting.AzureDocumentIntelligenceEndpoint), new AzureKeyCredential(Setting.AzureDocumentIntelligenceKey));

        var serviceEndpoint = new Uri(Setting.AzureCognitiveSearchEndpoint);
        var credential = new AzureKeyCredential(Setting.AzureCognitiveSearchKey);

        _searchClient = new SearchClient(serviceEndpoint, Setting.AzureCognitiveSearchIndexName, credential);
    }

    public async Task AddEmbeddings(string path, string remoteFileName, Claim claim, Investigator? investigator, int documentID, string hash)
    {
        var content = string.Empty;
        var textContent = null as string;
        var boundingBoxes = null as List<List<int>>;
        var vectorDocuments = new List<AzureSearch.VectorDocument>();

        using (var stream = new FileStream(path, FileMode.Open))
        {
            var vectorDocument = await ExtractEmbeddings(stream, remoteFileName, documentID, claim, investigator);
            vectorDocuments.AddRange(vectorDocument);
        }

        await _searchClient.UploadDocumentsAsync(vectorDocuments);
    }

    private async Task<List<AzureSearch.VectorDocument>> ExtractEmbeddings(Stream fileStream, string fileName, int documentID, Claim claim, Investigator? investigator)
    {
        if (fileStream.CanSeek)
            fileStream.Position = 0;

        var operation = await _documentIntelligenceClient.AnalyzeDocumentAsync(WaitUntil.Completed, "prebuilt-layout", BinaryData.FromStream(fileStream));
        var analyzeResult = operation.Value;

        var vectorDocuments = new List<AzureSearch.VectorDocument>();
        var previousPageLastLines = string.Empty;
        var pageNumber = 1;
        
        foreach (var page in analyzeResult.Pages)
        {
            var vectorDocument = await ExtractEmbeddings(page, fileName, documentID, claim, investigator, pageNumber, previousPageLastLines);            
            vectorDocuments.Add(vectorDocument);

            previousPageLastLines = string.Join("\n", page.Lines.TakeLast(3).Select(line => line.Content));
            pageNumber++;
        }

        return vectorDocuments;
    }

        
    private async Task<AzureSearch.VectorDocument> ExtractEmbeddings(DocumentPage page, string fileName, int documentID, Claim claim, Investigator? investigator, int pageNumber, string previousPageLastLines)
    {
        (var text, var boundingBoxes) = ExtractTextAndBoundingBoxes(page, previousPageLastLines);

        var embeddingOptions = new EmbeddingsOptions
        {
            DeploymentName = Setting.AzureOpenAIEmbeddingDeploymentName,
            Input = { text }
        };

        var returnValue = await _openAIClient.GetEmbeddingsAsync(embeddingOptions);
        var embedding = returnValue.Value.Data[0].Embedding.ToArray();

        var vectorDocument = new AzureSearch.VectorDocument
        {
            ID = Guid.NewGuid().ToString(),
            ClaimID = claim.ID,
            InvestigatorID = investigator?.ID,
            DocumentID = documentID,
            FileName = fileName,
            PageNumber = pageNumber,
            Embedding = embedding,
            Content = text,
            BoundingBoxes = boundingBoxes
        };

        return vectorDocument;
    }

    private (string, List<string>) ExtractTextAndBoundingBoxes(DocumentPage page, string previousPageLastLines)
    {
        var textBuilder = new StringBuilder();
        var boundingBoxes = new List<string>();
        var pageLines = page.Lines.ToList();

        if (!string.IsNullOrEmpty(previousPageLastLines))
            textBuilder.AppendLine(previousPageLastLines);

        foreach (var line in pageLines)
        {
            textBuilder.AppendLine(line.Content);

            if (line.Polygon != null && ValidateBoundingBox(line.Polygon))
            {
                var boxCoordinates = line.Polygon.Select(p => p.ToString("R")).ToList();
                boundingBoxes.Add(string.Join(",", boxCoordinates));
            }
            else
            {
                // Add empty bounding box to maintain alignment
                boundingBoxes.Add("");
            }
        }

        return (textBuilder.ToString(), boundingBoxes);
    }

    private bool ValidateBoundingBox(IReadOnlyList<Azure.AI.DocumentIntelligence.DocumentPoint> polygon)
    {
        if (polygon == null || polygon.Count != 4)
            return false;

        // Ensure coordinates are within reasonable bounds
        foreach (var point in polygon)
        {
            if (point.X < 0 || point.X > 1 || point.Y < 0 || point.Y > 1)
                return false;
        }

        // Verify box dimensions are reasonable (not too small)
        var minWidth = Math.Abs(polygon[1].X - polygon[0].X);
        var minHeight = Math.Abs(polygon[2].Y - polygon[0].Y);
        
        return minWidth >= 0.01 && minHeight >= 0.01;
    }

    /*
    foreach (var message in chat.Messages)
    {
        switch (message.ChatRole)
        {
            case Model.ChatRole.User:
                completionsOptions.Messages.Add(new ChatRequestUserMessage(message.Text));
                break;

            case Model.ChatRole.Assistant:
                completionsOptions.Messages.Add(new ChatRequestAssistantMessage(message.Text));
                break;
        }
    }
    */

    public async Task<AzureSearch.QueryResult> Query(Claim claim, Chat chat, string question)
    {
        var vectorDocuments = await QueryVectorDocuments(claim, null, question);
        var vectorDocumentsWithNumberedLines = GetDocumentsWithNumberedLines(vectorDocuments);
        var combinedTextWithNumberedLines = GetCombinedTextWithNumberedLines(vectorDocumentsWithNumberedLines);

        var completionsOptions = GetCompletionsOptions(question, combinedTextWithNumberedLines);
        var answerExtractionResult = new AzureSearch.AnswerExtractionResult();

        try
        {
            var responseWithoutStream = await _openAIClient.GetChatCompletionsAsync(completionsOptions);
            var jsonResponse = responseWithoutStream.Value.Choices[0].Message.Content.Trim();
        
            answerExtractionResult = JsonConvert.DeserializeObject<AzureSearch.AnswerExtractionResult>(jsonResponse);
            
            // Validate the extracted information based on question type
            if (question.ToLower().Contains("name"))
            {
                answerExtractionResult = ValidateNameExtraction(answerExtractionResult, vectorDocumentsWithNumberedLines);
            }
            else if (question.ToLower().Contains("address"))
            {
                answerExtractionResult = ValidateAddressExtraction(answerExtractionResult, vectorDocumentsWithNumberedLines);
            }
            else if (question.ToLower().Contains("claim") || question.ToLower().Contains("amount"))
            {
                answerExtractionResult = ValidateAmountExtraction(answerExtractionResult, vectorDocumentsWithNumberedLines);
            }
        }
        catch (Exception ex)
        {
            throw new ApiException(ErrorCode.DocumentOpenAIQueryFailed, $"Failed to query Azure OpenAI. {ex.Message}");
        }

        var references = BuildReferenceList(answerExtractionResult, vectorDocumentsWithNumberedLines);

        var queryResult = new AzureSearch.QueryResult
        {
            Answer = answerExtractionResult.Answer ?? "No answer found in the given context.",
            References = references.Take(3).ToList(),
            Confidence = answerExtractionResult.Confidence
        };

        return queryResult;
    }

    private List<AzureSearch.VectorDocumentWithNumberedLines> GetDocumentsWithNumberedLines(List<AzureSearch.VectorDocument> vectorDocuments)
    {
        var vectorDocumentWithNumberedLines = vectorDocuments.Select(vectorDocument =>
        {
            var lines = vectorDocument.Content.Split('\n');
            var numberedLines = lines.Select((line, index) => new AzureSearch.NumberedLine
            {
                Number = index + 1, // 1-indexed 
                Line = line
            }).ToList();

            return new AzureSearch.VectorDocumentWithNumberedLines
            {
                VectorDocument = vectorDocument,
                NumberedLines = numberedLines
            };
        }).ToList();

        return vectorDocumentWithNumberedLines;
    }

    private async Task<List<AzureSearch.VectorDocument>> QueryVectorDocuments(Claim claim, Investigator? investigator, string searchText = "")
    {
        var vectorDocuments = new List<AzureSearch.VectorDocument>();
        var filter = $"ClaimID eq {claim.ID}";

        if (investigator != null)
            filter += $" and InvestigatorID eq {investigator.ID}";

        var options = new SearchOptions
        {
            Filter = filter,
            Size = 100
        };

        options.Select.Add("FileName");
        options.Select.Add("Content");
        options.Select.Add("DocumentID");
        options.Select.Add("PageNumber");
        options.Select.Add("BoundingBoxes");

        try
        {
            var searchResults = await _searchClient.SearchAsync<AzureSearch.VectorDocument>(searchText, options);
            vectorDocuments = searchResults.Value.GetResultsAsync()
                .ToBlockingEnumerable()
                .OrderByDescending(r => r.Score)
                .Select(r => r.Document)
                .Take(10)
                .ToList();
        }
        catch (Exception ex)
        {
            throw new ApiException(ErrorCode.DocumentOpenAIQueryFailed, $"Failed to retrieve documents from Azure Search. {ex.Message}");
        }

        if (!vectorDocuments.Any())
            throw new ApiException(ErrorCode.DocumentOpenAIQueryNoResults, $"No documents were found for claim {claim.UniqueID}.");

        return vectorDocuments;
    }

    private string GetCombinedTextWithNumberedLines(List<AzureSearch.VectorDocumentWithNumberedLines> vectorDocumentsWithNumberedLines)
    {
        var combinedTextWithLineNumbers =
            string.Join('\n',
            vectorDocumentsWithNumberedLines
            .SelectMany(x => x.NumberedLines)
            .Select(x => $"Line {x.Number}: {x.Line}")
            .ToList());

        return combinedTextWithLineNumbers;
    }

    private ChatCompletionsOptions GetCompletionsOptions(string question, string combinedTextWithNumberedLines)
    {
        var completionsOptions = new ChatCompletionsOptions()
        {
            DeploymentName = Setting.AzureOpenAIDeploymentName,
            Temperature = (float)0,
            MaxTokens = 350,
            NucleusSamplingFactor = (float)1,
            FrequencyPenalty = 0,
            PresencePenalty = 0,
            ResponseFormat = ChatCompletionsResponseFormat.JsonObject
        };

        completionsOptions.Messages.Add(new ChatRequestSystemMessage(defaultSystemMessage));

        var userMessage = $"Question: {question}\n\nContext with Line Numbers:\n{combinedTextWithNumberedLines}";
        completionsOptions.Messages.Add(new ChatRequestUserMessage(userMessage));

        return completionsOptions;
    }

    private List<AzureSearch.VectorDocumentReference> BuildReferenceList(AzureSearch.AnswerExtractionResult answerExtractionResult, List<AzureSearch.VectorDocumentWithNumberedLines> vectorDocumentsWithNumberedLines)
    {
        var references = new List<AzureSearch.VectorDocumentReference>();

        if (answerExtractionResult.LineNumbers == null || !answerExtractionResult.LineNumbers.Any())
            return references;

        foreach (var docWithLines in vectorDocumentsWithNumberedLines)
        {
            var matchingLineIndices = answerExtractionResult.LineNumbers
                .Where(lineNum => lineNum > 0 && lineNum <= docWithLines.NumberedLines.Count)
                .ToList();

            if (!matchingLineIndices.Any())
                continue;

            try
            {
                references.Add(new AzureSearch.VectorDocumentReference
                {
                    DocumentID = docWithLines.VectorDocument.DocumentID,
                    FileName = docWithLines.VectorDocument.FileName,
                    PageNumber = docWithLines.VectorDocument.PageNumber,
                    BoundingBoxes = matchingLineIndices
                        .Select(lineNum => docWithLines.VectorDocument.BoundingBoxes[lineNum - 1])
                        .ToList()
                });
            }
            catch (ArgumentOutOfRangeException)
            {
                continue; // This can happen if bounding boxes don't match line numbers
            }
        }

        return references;
    }

    private AzureSearch.AnswerExtractionResult ValidateNameExtraction(
        AzureSearch.AnswerExtractionResult result, 
        List<AzureSearch.VectorDocumentWithNumberedLines> documents)
    {
        // Verify name appears in cited lines
        var citedLines = GetCitedLines(result.LineNumbers, documents);
        if (!citedLines.Any(line => line.Contains(result.Answer, StringComparison.OrdinalIgnoreCase)))
        {
            result.Confidence = 0.1;
            return result;
        }

        // Check for common name patterns
        var namePattern = @"^[A-Z][a-z]+ (?:[A-Z][a-z]+ )*[A-Z][a-z]+$";
        if (!System.Text.RegularExpressions.Regex.IsMatch(result.Answer, namePattern))
        {
            result.Confidence = 0.5;
            return result;
        }

        // Look for contextual clues
        var hasContext = citedLines.Any(line => 
            line.Contains("insured", StringComparison.OrdinalIgnoreCase) ||
            line.Contains("claimant", StringComparison.OrdinalIgnoreCase) ||
            line.Contains("policy holder", StringComparison.OrdinalIgnoreCase));

        result.Confidence = hasContext ? 0.95 : 0.7;
        return result;
    }

    private AzureSearch.AnswerExtractionResult ValidateAddressExtraction(
        AzureSearch.AnswerExtractionResult result, 
        List<AzureSearch.VectorDocumentWithNumberedLines> documents)
    {
        // Verify address appears in cited lines
        var citedLines = GetCitedLines(result.LineNumbers, documents);
        if (!citedLines.Any(line => line.Contains(result.Answer, StringComparison.OrdinalIgnoreCase)))
        {
            result.Confidence = 0.1;
            return result;
        }

        // Check for address components
        var hasStreetNumber = System.Text.RegularExpressions.Regex.IsMatch(result.Answer, @"\d+");
        var hasStreetName = System.Text.RegularExpressions.Regex.IsMatch(result.Answer, @"(?:street|st|avenue|ave|road|rd|boulevard|blvd|lane|ln|drive|dr)", RegexOptions.IgnoreCase);
        var hasZipCode = System.Text.RegularExpressions.Regex.IsMatch(result.Answer, @"\d{5}(-\d{4})?");
        
        var score = 0.0;
        if (hasStreetNumber) score += 0.3;
        if (hasStreetName) score += 0.3;
        if (hasZipCode) score += 0.4;
        
        result.Confidence = score;
        return result;
    }

    private AzureSearch.AnswerExtractionResult ValidateAmountExtraction(
        AzureSearch.AnswerExtractionResult result, 
        List<AzureSearch.VectorDocumentWithNumberedLines> documents)
    {
        // Verify amount appears in cited lines
        var citedLines = GetCitedLines(result.LineNumbers, documents);
        if (!citedLines.Any(line => line.Contains(result.Answer, StringComparison.OrdinalIgnoreCase)))
        {
            result.Confidence = 0.1;
            return result;
        }

        // Check for currency format
        var currencyPattern = @"^\$?\d{1,3}(,\d{3})*(\.\d{2})?$";
        if (!System.Text.RegularExpressions.Regex.IsMatch(result.Answer.Replace(" ", ""), currencyPattern))
        {
            result.Confidence = 0.5;
            return result;
        }

        // Look for contextual clues
        var hasContext = citedLines.Any(line => 
            line.Contains("total", StringComparison.OrdinalIgnoreCase) ||
            line.Contains("amount", StringComparison.OrdinalIgnoreCase) ||
            line.Contains("claim", StringComparison.OrdinalIgnoreCase));

        result.Confidence = hasContext ? 0.95 : 0.7;
        return result;
    }

    private List<string> GetCitedLines(List<int> lineNumbers, List<AzureSearch.VectorDocumentWithNumberedLines> documents)
    {
        var citedLines = new List<string>();
        foreach (var doc in documents)
        {
            foreach (var lineNum in lineNumbers)
            {
                if (lineNum > 0 && lineNum <= doc.NumberedLines.Count)
                {
                    citedLines.Add(doc.NumberedLines[lineNum - 1].Line);
                }
            }
        }
        return citedLines;
    }
}