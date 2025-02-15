using System;
using System.Collections.Generic;

 
namespace Reclaim.Api.Model;

public partial class Document : Base
{
    public int ID { get; set; }

    public DocumentType Type { get; set; }

    public int? ClaimID { get; set; }

    public int AccountID { get; set; }

    public string Name { get; set; } = null!;

    public string Path { get; set; } = null!;

    public int Size { get; set; }

    public string Description { get; set; } = null!;

    public string? Summary { get; set; }

    public string Hash { get; set; } = null!;

    public DateTime? OriginatedTimestamp { get; set; }

    public DateTime UploadedTimestamp { get; set; }

    public DateTime? IngestedTimestamp { get; set; }

    public DateTime? SummarizedTimestamp { get; set; }

    public DateTime? TombstonedTimestamp { get; set; }

    public Guid UniqueID { get; set; }



    public virtual Account Account { get; set; } = null!;

    public virtual ICollection<ChatMessageCitation> ChatMessageCitations { get; set; } = new List<ChatMessageCitation>();

    public virtual Claim? Claim { get; set; }
}
