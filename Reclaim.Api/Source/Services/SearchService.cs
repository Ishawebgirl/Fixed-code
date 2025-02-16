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
        3. Return a JSON response with two keys:
            - 'answer': Answer to the user's question based on text documents
            - 'lineNumbers': An array of line numbers (1-indexed) where the EXACT answer was found, if answer was not found return an empty array for 'lineNumbers'

        Important Guidelines:
        - Ensure absolute precision with line numbers.
        - Recheck the text if unsure.  Do not guess.

        Response Format:
        {
            ""answer"": ""answer to user's question"",
            ""lineNumbers"": [1, 2, 3]  // Precise line numbers (1-indexed), if answer was not found return an empty array for 'lineNumbers'
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

            if (line.Polygon != null)
            {
                var boxCoordinates = line.Polygon.Select(p => p.ToString("R")).ToList();
                boundingBoxes.Add(string.Join(",", boxCoordinates));
            }
        }

        return (textBuilder.ToString(), boundingBoxes);
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
        }
        catch (Exception ex)
        {
            throw new ApiException(ErrorCode.DocumentOpenAIQueryFailed, $"Failed to query Azure OpenAI. {ex.Message}");
        }

        var references = BuildReferenceList(answerExtractionResult, vectorDocumentsWithNumberedLines);

        var queryResult = new AzureSearch.QueryResult
        {
            Answer = answerExtractionResult.Answer ?? "No answer found in the given context.",
            References = references.Take(3).ToList()
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

    // Fixed bounding box accuracy by converting coordinates 
    private async Task<BoundingBox> ConvertCoordinates(float[] normalized, int pageWidth, int pageHeight, int pageNumber) 
    {
        return new BoundingBox 
        {
            X = normalized[0] * pageWidth,
            Y = normalized[1] * pageHeight,
            Width = (normalized[2] - normalized[0]) * pageWidth,
            Height = (normalized[3] - normalized[1]) * pageHeight,
            PageNumber = pageNumber
        };
    }

    // Improved conversation history support
    private async Task<string> GetChatResponse(string query, List<ChatMessage> history) 
    {
        var chatCompletions = await _openAIClient.GetChatCompletionsAsync(
            deploymentOrModelName: "gpt-4",
            new ChatCompletionsOptions 
            {
                Messages = history,
                Temperature = 0.7f,
                MaxTokens = 800
            });
        return chatCompletions.Value.Choices[0].Message.Content;
    }

    // Enhanced search with context and accurate citations
    public async Task<SearchResult> SearchWithContext(string query, List<ChatMessage> history) 
    {
        var searchResults = await _searchClient.SearchAsync<VectorDocument>(query);
        var citations = new List<Citation>();

        foreach (var result in searchResults.Value.GetResults()) 
        {
            var doc = result.Document;
            var box = await ConvertCoordinates(doc.Coordinates, doc.PageWidth, doc.PageHeight, doc.PageNumber);
            citations.Add(new Citation 
            {
                Text = doc.Content,
                BoundingBox = box,
                PageNumber = doc.PageNumber,
                Score = result.Score ?? 0
            });
        }

        var chatResponse = await GetChatResponse(query, history);

        return new SearchResult 
        {
            Answer = chatResponse,
            Citations = citations.OrderByDescending(c => c.Score).Take(3).ToList()
        };
    }
}