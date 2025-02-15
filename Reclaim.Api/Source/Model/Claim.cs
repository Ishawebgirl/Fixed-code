using System;
using System.Collections.Generic;

 
namespace Reclaim.Api.Model;

public partial class Claim : Base
{
    public int ID { get; set; }

    public ClaimType Type { get; set; }

    public ClaimStatus Status { get; set; }

    public ClaimDisposition Disposition { get; set; }

    public int AccountID { get; set; }

    public int PolicyID { get; set; }

    public int? InvestigatorID { get; set; }

    public string ExternalID { get; set; } = null!;

    public decimal? AmountSubmitted { get; set; }

    public decimal? AmountAdjusted { get; set; }

    public decimal? AmountRequested { get; set; }

    public decimal? AmountPaid { get; set; }

    public DateOnly EventDate { get; set; }

    public TimeOnly? EventTime { get; set; }

    public DateTime IngestedTimestamp { get; set; }

    public DateTime? AdjudicatedTimestamp { get; set; }

    public DateTime? TombstonedTimestamp { get; set; }

    public Guid UniqueID { get; set; }



    public virtual Account Account { get; set; } = null!;

    public virtual ICollection<Chat> Chats { get; set; } = new List<Chat>();

    public virtual ICollection<Document> Documents { get; set; } = new List<Document>();

    public virtual Investigator? Investigator { get; set; }

    public virtual Policy Policy { get; set; } = null!;
}
