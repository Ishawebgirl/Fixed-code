using Reclaim.Api.Model;
using Microsoft.EntityFrameworkCore;
using Newtonsoft.Json;

namespace Reclaim.Api.Services;

public class ChatService : BaseService
{
    private readonly DatabaseContext _db;
    private readonly IHttpContextAccessor _httpContextAccessor;
    private readonly LogService _logService;
    private readonly SearchService _searchService;

    public ChatService(DatabaseContext db, IHttpContextAccessor httpContextAccessor, LogService logService, SearchService searchService)
    {
        _db = db;
        _httpContextAccessor = httpContextAccessor;
        _logService = logService;
        _searchService = searchService;
    }

    private IQueryable<Chat> GetQuery()
    {
        var (accountID, role) = base.GetCurrentAccountInfo(_httpContextAccessor);

        IQueryable<Chat> query = _db.Chats
            .Include(x => x.Claim)
            .Include(x => x.Messages)
            .ThenInclude(x => x.Citations)
            .ThenInclude(x => x.Document);

        switch (role)
        {
            case Role.Administrator:
                break;

            case Role.Customer:
                query = query.Where(x => x.Claim.Policy.Customer.AccountID == accountID);
                break;

            case Role.Investigator:
                query = query.Where(x => x.Claim.Investigator.AccountID == accountID);
                break;
        }

        return query;
    }

    public async Task<Chat> Get(Guid uniqueID)
    {
        var chat = await GetQuery()
            .FirstOrDefaultAsync(x => x.UniqueID == uniqueID);

        if (chat == null)
            throw new ApiException(ErrorCode.ChatDoesNotExistForAccount, $"Chat {uniqueID} does not exist or is not associated with the current account");

        return chat;
    }

    public async Task<Chat> Create(Claim claim)
    {
        var (accountID, role) = base.GetCurrentAccountInfo(_httpContextAccessor);

        var chat = new Chat { AccountID = accountID, ClaimID = claim.ID, StartedTimestamp = DateTime.UtcNow, Type = ChatType.ClaimQuery };

        _db.Chats.Add(chat);
        await _db.SaveChangesAsync();

        return chat;
    }

    public async Task<ChatMessage> SubmitQuestion(Chat chat, string question)
    {
        var userMessage = new ChatMessage { ChatID = chat.ID, ChatRole = ChatRole.User, Text = question, SubmittedTimestamp = _db.ContextTimestamp, ReceivedTimestamp = _db.ContextTimestamp, IsError = false };
        _db.ChatMessages.Add(userMessage);
        await _db.SaveChangesAsync();

        var assistantMessage = new ChatMessage { ChatID = chat.ID, ChatRole = ChatRole.Assistant, SubmittedTimestamp = _db.ContextTimestamp, ReceivedTimestamp = DateTime.UtcNow, IsError = false };

        try
        {
            var queryResult = await _searchService.Query(chat.Claim, chat, question);

            assistantMessage.Text = queryResult.Answer;

            foreach (var reference in queryResult.References)
            {
                var boundingBoxList = new List<string>();
                
                foreach (var boundingBox in reference.BoundingBoxes)
                {
                    // Get just the top left and bottom right coordinates of the bounding box
                    var split = boundingBox.Split(',');
                    boundingBoxList.Add($"{split[0]},{split[1]},{split[4]},{split[5]}");
                }

                var citation = new ChatMessageCitation
                {
                    DocumentID = reference.DocumentID,
                    PageNumber = reference.PageNumber,
                    BoundingBoxes = string.Join('|', boundingBoxList)
                };

                assistantMessage.Citations.Add(citation);
            }
        }
        catch (Exception ex)
        {
            await _logService.Add(ex);
            assistantMessage.IsError = true;

            if (ex is ApiException && ((ApiException)ex).ErrorCode == ErrorCode.DocumentOpenAIQueryNoResults)
                assistantMessage.Text = "I'm sorry, currently no documents are available to query.";
            else
                assistantMessage.Text = $"I'm sorry, I wasn't able to query the documents. {ex.Message}";
        }

        _db.ChatMessages.Add(assistantMessage);
        await _db.SaveChangesAsync();

        return assistantMessage;
    }
}
