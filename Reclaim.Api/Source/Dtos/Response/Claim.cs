using Reclaim.Api.Model;
using System.ComponentModel.DataAnnotations;
using System.Diagnostics.CodeAnalysis;

namespace Reclaim.Api.Dtos;

public record Claim
{
    /// <example>5fe0302c-931d-423e-b955-bc43782e7a94</example>
    [Required]
    public Guid UniqueID { get; set; }

    /// <example>Fire</example>
    [Required]
    public ClaimType Type { get; set; }

    /// <example>Investigating</example>
    [Required]
    public ClaimStatus Status { get; set; }

    /// <example>Undecided</example>
    [Required]
    public ClaimDisposition Disposition { get; set; }

    /// <example>QE-28271-01</example>
    [Required]
    public string ExternalID { get; set; }

    /// <example>12200</example>
    public decimal? AmountSubmitted { get; set; }

    /// <example>6500</example>
    public decimal? AmountAdjusted { get; set; }

    /// <example>6500</example>
    public decimal? AmountPaid { get; set; }

    [Required]
    public DateOnly EventDate { get; set; }

    /// <example>10:00:00</example>
    public TimeOnly? EventTime { get; set; }
    public DateTime? IngestedTimestamp { get; set; }
    public DateTime? AdjudicatedTimestamp { get; set; }
    public DateTime? TombstonedTimestamp { get; set; }

    [Required]
    public List<Dtos.Document> Documents { get; set; }
    
    [Required]
    public Dtos.Policy Policy { get; set; }
    
    public Dtos.Investigator? Investigator { get; set; }

    [SetsRequiredMembers]
    public Claim(Model.Claim model, bool includeCustomer = true)
    {
        UniqueID = model.UniqueID;
        Type = model.Type;
        Status = model.Status;
        Disposition = model.Disposition;
        ExternalID = model.ExternalID;
        AmountSubmitted = model.AmountSubmitted;
        AmountAdjusted = model.AmountAdjusted;
        AmountPaid = model.AmountPaid;
        EventDate = model.EventDate;
        EventTime = model.EventTime;
        IngestedTimestamp = model.IngestedTimestamp;
        AdjudicatedTimestamp = model.AdjudicatedTimestamp;
        TombstonedTimestamp = model.TombstonedTimestamp;
        Documents = model.Documents.Select(x => new Dtos.Document(x)).ToList();
        Policy = new Dtos.Policy(model.Policy, includeCustomer);
        Investigator = model.Investigator != null ? new Dtos.Investigator(model.Investigator) : null;
    }
}
