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

    private const int COORDINATE_PRECISION = 6;
    private const int MAX_CONTEXT_WINDOW = 5;
    private const double MIN_CONFIDENCE_THRESHOLD = 0.90;
    private const int ADDRESS_CONTEXT_WINDOW = 8;
    private const double BOUNDING_BOX_TOLERANCE = 0.05;

    private static string defaultSystemMessage =
        @"You are an expert in analyzing property claim documents for home insurers and answering
        questions based only on the content in the provided text documents, along with line numbers.
        The current date and time is: " + DateTime.UtcNow.ToString("yyyy-MM-dd HH:mm:ss UTC") + @"

        Instructions:
        1. For EVERY answer, you MUST:
           - Only use information explicitly stated in the document
           - Include exact line numbers where the information is found
           - Verify information across multiple mentions if available
           - Return a confidence score of 0.0 to 1.0

        2. For NAMES (e.g., 'What is the name of the insured party?'):
           - Look for formal name fields or headers
           - Include any titles (Mr., Mrs., etc.) and suffixes
           - Verify against signature lines or repeated mentions
           - Only cite lines that contain the EXACT full name

        3. For ADDRESSES:
           - Must include complete address (street, city, state, zip)
           - Look for address fields or letterhead sections
           - Verify against any repeated mentions
           - Only cite lines containing the EXACT address components
           - Check for address format indicators (e.g., 'Address:', 'Location:')

        4. For NUMERICAL VALUES and CALCULATIONS:
           - Must show exact numbers from the document
           - Include line numbers for EACH number used
           - Show calculation steps if multiple numbers are involved
           - Verify totals against component values
           - Only cite lines containing the EXACT numbers used

        5. Response Format Requirements:
           - 'answer': Must be exact text from document
           - 'lineNumbers': Array of line numbers containing EXACT text
           - 'confidence': Only high confidence (>0.90) for critical fields
           - 'type': Must be one of ['name', 'address', 'calculation', 'general']

        6. Citation Rules:
           - Only cite lines that contain the EXACT text used in the answer
           - For addresses, cite ALL lines containing address components
           - For calculations, cite ALL lines containing relevant numbers
           - Never cite contextual or surrounding lines unless they contain answer components

        If you cannot find information meeting these strict requirements, respond with:
        {
            ""answer"": ""I cannot find a reliable answer in the provided documents"",
            ""lineNumbers"": [],
            ""confidence"": 0.0,
            ""type"": ""general""
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
        {
            textBuilder.AppendLine("--- Previous Page Context ---");
            textBuilder.AppendLine(previousPageLastLines);
            textBuilder.AppendLine("--- Current Page Content ---");
        }

        foreach (var line in pageLines)
        {
            textBuilder.AppendLine(line.Content);

            if (line.Polygon != null)
            {
                if (ValidateBoundingBox(line.Polygon))
                {
                    var boxCoordinates = line.Polygon
                        .Select(p => Math.Round(p, COORDINATE_PRECISION).ToString("F" + COORDINATE_PRECISION))
                        .ToList();
                    boundingBoxes.Add(string.Join(",", boxCoordinates));
                }
                else
                {
                    _logService.Warning($"Invalid bounding box detected for line: {line.Content}");
                    boundingBoxes.Add(string.Empty);
                }
            }
        }

        return (textBuilder.ToString(), boundingBoxes);
    }

    private bool ValidateBoundingBox(IReadOnlyList<float> polygon)
    {
        if (polygon == null || polygon.Count != 8) return false;

        if (!polygon.All(coord => !float.IsNaN(coord) && !float.IsInfinity(coord) && coord >= 0))
            return false;

        try
        {
            var points = new List<(float X, float Y)>
            {
                (polygon[0], polygon[1]),
                (polygon[2], polygon[3]),
                (polygon[4], polygon[5]),
                (polygon[6], polygon[7])
            };

            var width1 = Math.Abs(points[1].X - points[0].X);
            var width2 = Math.Abs(points[2].X - points[3].X);
            var height1 = Math.Abs(points[2].Y - points[1].Y);
            var height2 = Math.Abs(points[3].Y - points[0].Y);

            var tolerance = BOUNDING_BOX_TOLERANCE;
            
            if (Math.Abs(width1 - width2) > width1 * tolerance ||
                Math.Abs(height1 - height2) > height1 * tolerance)
                return false;

            if (width1 < 5 || height1 < 5)
                return false;

            var aspectRatio = width1 / height1;
            if (aspectRatio < 0.1 || aspectRatio > 20)
                return false;

        }
        catch (Exception ex)
        {
            _logService.Error($"Error validating bounding box: {ex.Message}");
            return false;
        }

        return true;
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
        // First, build conversation history
        var conversationContext = BuildConversationContext(chat);
        
        // Use conversation context for better semantic search
        var vectorDocuments = await QueryVectorDocuments(claim, null, $"{conversationContext}\n{question}");
        var vectorDocumentsWithNumberedLines = GetDocumentsWithNumberedLines(vectorDocuments);
        var combinedTextWithNumberedLines = GetCombinedTextWithNumberedLines(vectorDocumentsWithNumberedLines);

        var completionsOptions = GetCompletionsOptions(question, combinedTextWithNumberedLines, conversationContext);
        var answerExtractionResult = new AzureSearch.AnswerExtractionResult();

        try
        {
            var responseWithoutStream = await _openAIClient.GetChatCompletionsAsync(completionsOptions);
            var jsonResponse = responseWithoutStream.Value.Choices[0].Message.Content.Trim();
        
            answerExtractionResult = JsonConvert.DeserializeObject<AzureSearch.AnswerExtractionResult>(jsonResponse);
            
            // Enhanced validation of answer and references
            if (answerExtractionResult.LineNumbers != null && answerExtractionResult.LineNumbers.Any())
            {
                var exactTextMatches = ValidateExactTextMatches(
                    answerExtractionResult.Answer,
                    answerExtractionResult.LineNumbers,
                    vectorDocumentsWithNumberedLines
                );
                
                if (!exactTextMatches)
                {
                    // If exact matches weren't found, try to find better matches
                    var (correctedAnswer, correctedLines) = FindBestTextMatches(
                        answerExtractionResult.Answer,
                        vectorDocumentsWithNumberedLines
                    );
                    
                    answerExtractionResult.Answer = correctedAnswer;
                    answerExtractionResult.LineNumbers = correctedLines;
                }
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
            References = references.Take(3).ToList()
        };

        return queryResult;
    }

    private string BuildConversationContext(Chat chat)
    {
        if (chat?.Messages == null || !chat.Messages.Any())
            return string.Empty;

        var contextBuilder = new StringBuilder();
        contextBuilder.AppendLine("Previous conversation context:");
        
        foreach (var message in chat.Messages.TakeLast(6)) // Keep last 3 turns (6 messages)
        {
            var role = message.ChatRole == Model.ChatRole.User ? "Human" : "Assistant";
            contextBuilder.AppendLine($"{role}: {message.Text}");
        }
        
        return contextBuilder.ToString();
    }

    private ChatCompletionsOptions GetCompletionsOptions(string question, string combinedTextWithNumberedLines, string conversationContext)
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

        // Add conversation history if exists
        if (!string.IsNullOrEmpty(conversationContext))
        {
            completionsOptions.Messages.Add(new ChatRequestSystemMessage(
                "Previous conversation context to maintain continuity:\n" + conversationContext));
        }

        var userMessage = $"Question: {question}\n\nContext with Line Numbers:\n{combinedTextWithNumberedLines}";
        completionsOptions.Messages.Add(new ChatRequestUserMessage(userMessage));

        return completionsOptions;
    }

    private bool ValidateExactTextMatches(string answer, List<int> lineNumbers, List<AzureSearch.VectorDocumentWithNumberedLines> documents)
    {
        var normalizedAnswer = NormalizeText(answer);
        
        foreach (var doc in documents)
        {
            var relevantLines = lineNumbers
                .Where(ln => ln > 0 && ln <= doc.NumberedLines.Count)
                .Select(ln => doc.NumberedLines[ln - 1].Line);
                
            var combinedText = NormalizeText(string.Join(" ", relevantLines));
            
            // Check if all answer components are found in the cited lines
            var answerParts = normalizedAnswer.Split(new[] { ' ', ',', '.', ';' }, StringSplitOptions.RemoveEmptyEntries);
            if (answerParts.All(part => combinedText.Contains(part)))
                return true;
        }
        
        return false;
    }

    private (string answer, List<int> lineNumbers) FindBestTextMatches(
        string answer, 
        List<AzureSearch.VectorDocumentWithNumberedLines> documents)
    {
        var bestMatchLines = new List<int>();
        var normalizedAnswer = NormalizeText(answer);
        var answerParts = normalizedAnswer.Split(new[] { ' ', ',', '.', ';' }, StringSplitOptions.RemoveEmptyEntries);
        
        foreach (var doc in documents)
        {
            for (int i = 0; i < doc.NumberedLines.Count; i++)
            {
                var line = NormalizeText(doc.NumberedLines[i].Line);
                var matchScore = answerParts.Count(part => line.Contains(part)) / (double)answerParts.Length;
                
                if (matchScore > 0.5) // If more than 50% of answer parts found
                {
                    bestMatchLines.Add(i + 1);
                }
            }
        }
        
        // If we found better matches, reconstruct the answer from the matching lines
        if (bestMatchLines.Any())
        {
            var matchingText = string.Join(" ", bestMatchLines
                .Select(ln => documents
                    .SelectMany(d => d.NumberedLines)
                    .FirstOrDefault(nl => nl.Number == ln)?.Line)
                .Where(l => l != null));
                
            return (matchingText, bestMatchLines);
        }
        
        return (answer, bestMatchLines);
    }

    private string NormalizeText(string text)
    {
        return text.ToLower()
            .Replace("  ", " ")
            .Trim();
    }

    private List<AzureSearch.VectorDocumentWithNumberedLines> GetDocumentsWithNumberedLines(List<AzureSearch.VectorDocument> vectorDocuments)
    {
        var vectorDocumentWithNumberedLines = vectorDocuments.Select(vectorDocument =>
        {
            var lines = vectorDocument.Content.Split('\n');
            var numberedLines = lines.Select((line, index) => new AzureSearch.NumberedLine
            {
                Number = index + 1,
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

    private List<AzureSearch.VectorDocumentReference> BuildReferenceList(
        AzureSearch.AnswerExtractionResult answerExtractionResult,
        List<AzureSearch.VectorDocumentWithNumberedLines> vectorDocumentsWithNumberedLines)
    {
        var references = new List<AzureSearch.VectorDocumentReference>();

        if (answerExtractionResult.LineNumbers == null ||
            !answerExtractionResult.LineNumbers.Any() ||
            answerExtractionResult.Confidence < MIN_CONFIDENCE_THRESHOLD)
        {
            return references;
        }

        var contextWindow = answerExtractionResult.Type?.ToLower() switch
        {
            "address" => ADDRESS_CONTEXT_WINDOW,
            "name" => 3,
            "calculation" => 5,
            _ => MAX_CONTEXT_WINDOW
        };

        foreach (var docWithLines in vectorDocumentsWithNumberedLines)
        {
            var lineGroups = GetConsecutiveLineGroups(
                answerExtractionResult.LineNumbers
                    .Where(lineNum => lineNum > 0 && lineNum <= docWithLines.NumberedLines.Count)
                    .ToList(),
                contextWindow
            );

            foreach (var group in lineGroups)
            {
                try
                {
                    if (ValidateContentRelevance(group, docWithLines, answerExtractionResult))
                    {
                        var reference = new AzureSearch.VectorDocumentReference
                        {
                            DocumentID = docWithLines.VectorDocument.DocumentID,
                            FileName = docWithLines.VectorDocument.FileName,
                            PageNumber = docWithLines.VectorDocument.PageNumber,
                            BoundingBoxes = group
                                .Select(lineNum => docWithLines.VectorDocument.BoundingBoxes[lineNum - 1])
                                .Where(box => !string.IsNullOrEmpty(box))
                                .ToList()
                        };

                        if (ValidateTypeSpecificBoundingBoxes(reference, answerExtractionResult.Type))
                        {
                            references.Add(reference);
                        }
                    }
                }
                catch (Exception ex)
                {
                    _logService.Error($"Error building reference for document {docWithLines.VectorDocument.DocumentID}: {ex.Message}");
                    continue;
                }
            }
        }

        return references;
    }

    private List<List<int>> GetConsecutiveLineGroups(List<int> lineNumbers, int contextWindow)
    {
        var groups = new List<List<int>>();
        if (!lineNumbers.Any()) return groups;

        lineNumbers.Sort();
        var currentGroup = new List<int> { lineNumbers[0] };

        for (int i = 1; i < lineNumbers.Count; i++)
        {
            if (lineNumbers[i] - lineNumbers[i - 1] <= contextWindow)
            {
                currentGroup.Add(lineNumbers[i]);
            }
            else
            {
                groups.Add(new List<int>(currentGroup));
                currentGroup = new List<int> { lineNumbers[i] };
            }
        }
        
        groups.Add(currentGroup);
        return groups;
    }

    private bool ValidateContentRelevance(
        List<int> lineGroup,
        AzureSearch.VectorDocumentWithNumberedLines doc,
        AzureSearch.AnswerExtractionResult result)
    {
        var groupContent = string.Join(" ",
            lineGroup.Select(lineNum => doc.NumberedLines[lineNum - 1].Line));

        switch (result.Type?.ToLower())
        {
            case "address":
                return ValidateAddressContent(groupContent, result.Answer);
            case "name":
                return ValidateNameContent(groupContent, result.Answer);
            case "calculation":
                return ValidateCalculationContent(groupContent, result.Answer);
            default:
                return ValidateGeneralContent(groupContent, result.Answer);
        }
    }

    private bool ValidateAddressContent(string content, string answer)
    {
        var normalizedContent = NormalizeAddress(content);
        var normalizedAnswer = NormalizeAddress(answer);

        var addressParts = normalizedAnswer.Split(new[] { ' ', ',', '\n' }, StringSplitOptions.RemoveEmptyEntries);

        var significantParts = addressParts.Where(part => part.Length > 2);
        return significantParts.All(part => normalizedContent.Contains(part));
    }

    private string NormalizeAddress(string address)
    {
        return address.ToLower()
            .Replace("street", "st")
            .Replace("avenue", "ave")
            .Replace("road", "rd")
            .Replace("drive", "dr")
            .Replace("lane", "ln")
            .Replace("boulevard", "blvd")
            .Replace("circle", "cir")
            .Replace("court", "ct")
            .Replace("suite", "ste")
            .Replace("apartment", "apt")
            .Replace("number", "no")
            .Replace(".", "")
            .Replace(",", " ")
            .Replace("  ", " ");
    }

    private bool ValidateNameContent(string content, string answer)
    {
        var normalizedContent = content.ToLower();
        var normalizedAnswer = answer.ToLower();

        var nameParts = normalizedAnswer.Split(new[] { ' ', '.', ',' }, StringSplitOptions.RemoveEmptyEntries);
        return nameParts.All(part => normalizedContent.Contains(part));
    }

    private bool ValidateCalculationContent(string content, string answer)
    {
        var contentNumbers = ExtractNumbers(content);
        var answerNumbers = ExtractNumbers(answer);

        if (!answerNumbers.All(num => contentNumbers.Contains(num)))
            return false;

        if (answerNumbers.Count > 1)
        {
            var lastNumber = answerNumbers.Last();
            var otherNumbers = answerNumbers.Take(answerNumbers.Count - 1);
            
            return IsValidCalculationResult(lastNumber, otherNumbers);
        }

        return true;
    }

    private bool IsValidCalculationResult(decimal result, IEnumerable<decimal> components)
    {
        var sum = components.Sum();
        var product = components.Aggregate(1M, (a, b) => a * b);
        var tolerance = 0.01M;

        return Math.Abs(result - sum) < tolerance ||
               Math.Abs(result - product) < tolerance ||
               components.Any(c => Math.Abs(result - c) < tolerance);
    }

    private bool ValidateGeneralContent(string content, string answer)
    {
        return answer.Split(' ')
            .Where(word => word.Length > 4)
            .Any(word => content.Contains(word, StringComparison.OrdinalIgnoreCase));
    }

    private HashSet<decimal> ExtractNumbers(string text)
    {
        var numbers = new HashSet<decimal>();
        var numberStrings = System.Text.RegularExpressions.Regex.Matches(text, @"[\d,]+\.?\d*")
            .Select(m => m.Value.Replace(",", ""));

        foreach (var numStr in numberStrings)
        {
            if (decimal.TryParse(numStr, out decimal num))
            {
                numbers.Add(num);
            }
        }

        return numbers;
    }

    private bool ValidateTypeSpecificBoundingBoxes(AzureSearch.VectorDocumentReference reference, string type)
    {
        if (string.IsNullOrEmpty(type)) return true;

        switch (type.ToLower())
        {
            case "address":
            case "name":
                return ValidateAlignedBoxes(reference.BoundingBoxes);
            case "calculation":
                return ValidateCalculationBoxes(reference.BoundingBoxes);
            default:
                return true;
        }
    }

    private bool ValidateAlignedBoxes(List<string> boundingBoxes)
    {
        if (boundingBoxes.Count <= 1) return true;

        try
        {
            var boxes = boundingBoxes.Select(box => 
                box.Split(',').Select(float.Parse).ToList()).ToList();

            var firstTop = boxes[0][1];
            var tolerance = (boxes[0][5] - boxes[0][1]) * 0.1f;

            return boxes.All(box => Math.Abs(box[1] - firstTop) <= tolerance);
        }
        catch
        {
            return false;
        }
    }

    private bool ValidateCalculationBoxes(List<string> boundingBoxes)
    {
        if (boundingBoxes.Count <= 1) return true;

        try
        {
            var boxes = boundingBoxes.Select(box =>
                box.Split(',').Select(float.Parse).ToList()).ToList();

            var firstLeft = boxes[0][0];
            var tolerance = (boxes[0][2] - boxes[0][0]) * 0.2f;

            return boxes.All(box => Math.Abs(box[0] - firstLeft) <= tolerance);
        }
        catch
        {
            return false;
        }
    }
}