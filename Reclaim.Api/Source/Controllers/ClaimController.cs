using Microsoft.AspNetCore.Mvc;
using Reclaim.Api.Services;

namespace Reclaim.Api.Controllers;

[RequireRole]
[ValidateModel]
public class ClaimController : BaseController
{
    private readonly ClaimService _claimService;
    private readonly ChatService _chatService;
    private readonly DocumentService _documentService;

    public ClaimController(ClaimService claimService, ChatService chatService, DocumentService documentService)
    {
        _claimService = claimService;
        _chatService = chatService;
        _documentService = documentService;
    }

    /// <summary>
    /// Retrieve all claims in the system, currently not paged or limited
    /// </summary>
    [HttpGet("claims")]
    [RequireRole]
    public async Task<List<Dtos.Claim>> GetClaims()
    {
        var claims = await _claimService.GetAll();
             
        return claims.Select(x => new Dtos.Claim(x)).ToList();
    }

    /// <summary>
    /// Retrieve a given claim
    /// </summary>
    /// <remarks>
    /// ErrorCode.ClaimDoesNotExist\
    /// ErrorCode.ClaimDoesNotExistForAccount
    /// </remarks>
    /// <param name="uniqueID" example="92f76104-dfbe-497d-aa35-754eeba93589">The claims's public unique ID</param>
    [HttpGet("claims/{uniqueID}")]
    [RequireRole]
    public async Task<Dtos.Claim> GetClaim(Guid uniqueID)
    {
        var claim = await _claimService.Get(uniqueID);

        return new Dtos.Claim(claim);
    }

    /// <summary>
    /// Create a new RAG chat for the given claim
    /// </summary>
    /// <remarks>
    /// ErrorCode.ClaimDoesNotExist\
    /// ErrorCode.ClaimDoesNotExistForAccount
    /// </remarks>
    /// <param name="uniqueID" example="92f76104-dfbe-497d-aa35-754eeba93589">The claims's public unique ID</param>
    [HttpPut("claims/{uniqueID}/chats")]
    [RequireRole]
    public async Task<Dtos.Chat> CreateChat(Guid uniqueID)
    {
        var claim = await _claimService.Get(uniqueID);
        var chat = await _chatService.Create(claim);

        return new Dtos.Chat(chat);
    }

    /// <summary>
    /// Get the message history for a given chat
    /// </summary>
    /// <remarks>
    /// ErrorCode.ChatDoesNotExist\
    /// ErrorCode.ChatNotAssociatedToClaim\
    /// </remarks>
    /// <param name="uniqueID" example="92f76104-dfbe-497d-aa35-754eeba93589">The claim's public unique ID</param>
    /// <param name="chatUniqueID" example="30237a53-2da3-4acd-b400-07630410dcec">The chat's public unique ID</param>
    [HttpGet("claims/{uniqueID}/chats/{chatUniqueID}")]
    [RequireRole]
    public async Task<Dtos.Chat> GetChat(Guid uniqueID, Guid chatUniqueID)
    {
        var chat = await _chatService.Get(chatUniqueID);

        return new Dtos.Chat(chat);
    }

    /// <summary>
    /// Query all the documents associated to a given claim using Azure OpenAI
    /// </summary>
    /// <remarks>
    /// ErrorCode.ChatDoesNotExist\
    /// ErrorCode.ChatNotAssociatedToClaim\
    /// ErrorCode.DocumentOpenAIQueryFailed
    /// </remarks>
    /// <param name="uniqueID" example="92f76104-dfbe-497d-aa35-754eeba93589">The claim's public unique ID</param>
    /// <param name="chatUniqueID" example="30237a53-2da3-4acd-b400-07630410dcec">The chat's public unique ID</param>
    /// <param name="question" example="What did the policyholder say about the hail damage?">A natural-language question to apply to all documents associated with this claim</param>
    [HttpPost("claims/{uniqueID}chats/{chatUniqueID}")]
    [RequireRole]
    public async Task<Dtos.Chat> QueryClaim(Guid uniqueID, Guid chatUniqueID, string question)
    {
        var chat = await _chatService.Get(chatUniqueID);

        var answer = await _chatService.SubmitQuestion(chat, question);
        var reloaded = await _chatService.Get(chatUniqueID);

        return new Dtos.Chat(chat);
    }

    /// <summary>
    /// Upload a single document for a given claim to blob storage
    /// </summary>
    /// <remarks>
    /// ErrorCode.ClaimDoesNotExist\
    /// ErrorCode.DocumentHashAlreadyExists\
    /// ErrorCode.DocumentUploadToAzureFailed\
    /// ErrorCode.DocumentEnumerationFromAzureFailed\
    /// ErrorCode.DocumentTypeNotSupported  
    /// </remarks>
    /// <param name="uniqueID" example="92f76104-dfbe-497d-aa35-754eeba93589">The claims's public unique ID</param>
    /// <param name="file" example="">A stream containing the file's contents</param>
    /// <param name="lastModifiedTimestamp" example="2016-01-01T00:00:00.0000000-00:00">The last modified time of the file (from the caller's local OS)</param>
    [HttpPut("claims/{uniqueID}/documents")]
    [RequireRole]
    public async Task<Dtos.Document> UploadDocument(Guid uniqueID, IFormFile file, DateTime lastModifiedTimestamp)
    {
        var claim = await _claimService.Get(uniqueID);
        var document = await _documentService.Ingest(file, lastModifiedTimestamp, claim);

        return new Dtos.Document(document);
    }

    /// <summary>
    /// Download a single document for a given claim from blob storage
    /// </summary>
    /// <remarks>
    /// ErrorCode.DocumentDoesNotExist\
    /// ErrorCode.DocumentDownloadFromAzureFailed
    /// </remarks>
    /// <param name="uniqueID" example="92f76104-dfbe-497d-aa35-754eeba93589">The claims's public unique ID</param>
    /// <param name="documentUniqueID" example="ff4c9fa3-5682-4380-ad07-e3f439a70ac0">The document's public unique ID</param>
    [HttpGet("claims/{uniqueID}/documents/{documentUniqueID}")]
    [RequireRole]
    public async Task<FileResult> DownloadDocument(Guid uniqueID, Guid documentUniqueID)
    {
        var claim = await _claimService.Get(uniqueID);
        var document = await _documentService.Get(documentUniqueID);
        
        var stream = await _documentService.Download(document);

        return File(stream.ToArray(), document.ContentType, document.Name, document.OriginatedTimestamp, Microsoft.Net.Http.Headers.EntityTagHeaderValue.Any);
    }

    /// <summary>
    /// Tombstone a single document, removing it from search space
    /// </summary>
    /// <remarks>
    /// ErrorCode.DocumentDoesNotExist
    /// </remarks>
    /// <param name="uniqueID" example="92f76104-dfbe-497d-aa35-754eeba93589">The claims's public unique ID</param>
    /// <param name="documentUniqueID" example="ff4c9fa3-5682-4380-ad07-e3f439a70ac0">The document's public unique ID</param>
    [HttpDelete("claims/{uniqueID}/documents/{documentUniqueID}")]
    [RequireRole]
    public async Task TombstoneDocument(Guid uniqueID, Guid documentUniqueID)
    {
        var document = await _documentService.Get(documentUniqueID);

        await _documentService.Tombstone(document);
    }
}