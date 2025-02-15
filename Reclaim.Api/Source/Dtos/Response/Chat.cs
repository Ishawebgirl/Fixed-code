using Reclaim.Api.Model;
using System.ComponentModel.DataAnnotations;
using System.Diagnostics.CodeAnalysis;
using System.Xml;

namespace Reclaim.Api.Dtos;

public record Chat
{
    /// <example>5fe0302c-931d-423e-b955-bc43782e7a94</example>
    [Required]
    public Guid UniqueID { get; set; }

    /// <example>QE-28271-01</example>
    [Required]
    public string ClaimExternalID { get; set; }

    /// <example>ClaimQuery</example>
    [Required]
    public ChatType Type { get; set; }

    [Required]
    public List<ChatMessage> Messages { get; set; }

    [Required]
    public DateTime StartedTimestamp { get; set; }

    [SetsRequiredMembers]
    public Chat(Model.Chat model)
    {
        UniqueID = model.UniqueID;
        Type = model.Type;
        Messages = model.Messages.Select(x => new Dtos.ChatMessage(x)).ToList();
        StartedTimestamp = model.StartedTimestamp;
        ClaimExternalID = model.Claim.ExternalID;
    }
}

public record ChatMessage
{
    public record Citation
    {
        [Required]
        public Guid UniqueID { get; set; }

        [Required]
        public int PageNumber { get; set; }

        [Required]
        public string BoundingBoxes { get; set; }

        [Required]
        public string FileName { get; set; }

        [Required]
        public Guid DocumentUniqueID { get; set; }
    }

    [Required]
    public Guid UniqueID { get; set; }

    [Required]
    public ChatRole ChatRole { get; set; }

    [Required]
    public string Text { get; set; }
    
    [Required]
    public DateTime SubmittedTimestamp { get; set; }

    public DateTime? ReceivedTimestamp { get; set; }

    public List<Citation> Citations { get; set; }

    [SetsRequiredMembers]
    public ChatMessage(Model.ChatMessage model)
    {
        UniqueID = model.UniqueID;
        ChatRole = model.ChatRole;
        Text = model.Text;
        SubmittedTimestamp = model.SubmittedTimestamp;
        ReceivedTimestamp = model.ReceivedTimestamp;
        Citations = model.Citations.Select(x => new Citation { 
            UniqueID = x.UniqueID,
            PageNumber = x.PageNumber, 
            BoundingBoxes = x.BoundingBoxes, 
            FileName = x.Document.Name,
            DocumentUniqueID = x.Document.UniqueID
        }).ToList();
    }
}
